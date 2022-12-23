#pragma once

#include <madrona/span.hpp>
#include <madrona/query.hpp>

#include <madrona/state.hpp>

#include "mw_gpu/const.hpp"
#include "mw_gpu/worker_init.hpp"

#include <cuda/barrier>
#include <cuda/std/tuple>

namespace madrona {

#if 0
namespace mwGPU {

    struct ParallelFor {
        QueryRef *query;
    };

    struct ClearTmp {
        uint32_t archetypeID;
    };

    struct CompactArchetype {
        uint32_t archetypeID;
    };
    
    struct SortArchetypeSetup {
        uint32_t archetypeID;
        int32_t columnIDX;
        int32_t numPasses;
    };

    struct SortArchetype {
        uint32_t archetypeID;
    };
    
    struct RecycleEntities {
        int32_t recycleBase;
    };

    struct CustomData {
        void *ptr;
    };

    union {
        ParallelFor parallelFor;
        ClearTmp clearTmp;
        CompactArchetype compactArchetype;
        SortArchetypeSetup sortArchetypeSetup;
        SortArchetype sortArchetype;
        RecycleEntities recycleEntities;
        CustomData custom;
    };

}
#endif

struct NodeBase {
    uint32_t numDynamicInvocations;
};

class TaskGraph {
private:
    static inline constexpr uint32_t maxNodeDataBytes = 64;
    struct alignas(maxNodeDataBytes) NodeData {
        char userData[maxNodeDataBytes];
    };

    struct Node {
        uint32_t dataIDX;
        uint32_t fixedCount;
        uint32_t funcID;
        std::atomic_uint32_t curOffset;
        std::atomic_uint32_t numRemaining;
        std::atomic_uint32_t totalNumInvocations;
    };

public:
    struct NodeID {
        int32_t id;
    };

    struct DataID {
        int32_t id;
    };

    class Builder {
    public:
        Builder(uint32_t max_nodes,
                uint32_t max_node_datas,
                uint32_t max_num_dependencies);
        ~Builder();

        template <typename NodeT, typename... Args>
        NodeT & constructNodeData(Args &&...args);

        template <auto fn, typename NodeT>
        NodeID addNodeFn(const NodeT &node,
                         Span<const NodeID> dependencies,
                         int32_t fixed_num_invocations = 0);

        template <typename NodeT, int32_t count = 1, typename... Args>
        NodeID addOneOffNode(Span<const NodeID> dependencies,
                             Args && ...args);

        template <typename NodeT, typename... Args>
        NodeID addDynamicCountNode(Span<const NodeID> dependencies,
                                   Args && ...args);

        template <typename NodeT>
        inline NodeID addToGraph(Span<const NodeID> dependencies);

        void build(TaskGraph *out);

    private:
        NodeID registerNode(Node &&node,
                            Span<const NodeID> dependencies)

        struct StagedNode {
            Node node;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        StagedNode *staged_;
        uint32_t num_nodes_;
        NodeData *node_datas_;
        uint32_t num_datas_;
        NodeID *all_dependencies_;
        uint32_t num_dependencies_;
        uint32_t num_worlds_;
    };

    enum class WorkerState {
        Run,
        PartialRun,
        Loop,
        Exit,
    };

    TaskGraph(const TaskGraph &) = delete;
    ~TaskGraph();

    void init();

    WorkerState getWork(mwGPU::EntryData **entry_data,
                        uint32_t *run_func_id,
                        int32_t *run_offset);

    void finishWork();

    static inline WorldBase * getWorld(int32_t world_idx)
    {
        const auto &consts = mwGPU::GPUImplConsts::get();
        auto world_ptr = (char *)consts.worldDataAddr +
            world_idx * (int32_t)consts.numWorldDataBytes;

        return (WorldBase *)world_ptr;
    }

    struct BlockState;
private:
    TaskGraph(Node *nodes, uint32_t num_nodes);

    inline void setBlockState();
    inline uint32_t computeNumInvocations(NodeState &node);

    Node *sorted_nodes_;
    uint32_t num_nodes_;
    NodeData *node_datas_;
    std::atomic_uint32_t cur_node_idx_;
    cuda::barrier<cuda::thread_scope_device> init_barrier_;

friend class Builder;
};

template <typename ContextT, auto Fn, typename ...ComponentTs>
class ParallelForNode : DynamicCountNode<ContextT, Fn, ComponentTs...>> {
public:
    ParallelForNode();

    inline void run(int32_t invocation_idx);
    inline uint32_t numInvocations();

    static TaskGraph::NodeID addToGraph(TaskGraph::Builder &builder,
                                        Span<const NodeID> dependencies);

private:
    inline ContextT makeContext(WorldID world_id);

    QueryRef query_ref_;
};

struct ClearTmpNodeBase : NodeBase {
    ClearTmpNode(uint32_t archetype_id);

    void run(int32_t);

    static TaskGraph::NodeID addToGraph(TaskGraph::Builder &builder,
                                        Span<const NodeID> dependencies,
                                        uint32_t archetype_id);

    uint32_t archetypeID;
};

template <typename ArchetypeT>
struct ClearTmpNode : ClearTmpNodeBase {
    static TaskGraph::NodeID addToGraph(TaskGraph::Builder &builder,
                                        Span<const NodeID> dependencies);
};

struct RecycleEntitiesNode : NodeBase {
    RecycleEntitiesNode();

    void run(int32_t invocation_idx);
    uint32_t numInvocations();

    static TaskGraph::NodeID addToGraph(TaskGraph::Builder &builder,
                                        Span<const NodeID> dependencies);

    int32_t recycleBase;
};

struct ResetTmpAllocNode : NodeBase {
    void run(int32_t);

    static TaskGraph::NodeID addToGraph(TaskGraph::Builder &builder,
                                        Span<const NodeID> dependencies);
};

struct CompactArchetypeNodeBase : NodeBase {
    CompactArchetypeNode(uint32_t archetype_id);

    void run(int32_t invocation_idx);
    uint32_t numInvocations();

    uint32_t archetypeID;
};

template <typename ArchetypeT>
struct CompactArchetypeNode : CompactArchetypeNodeBase {
    static TaskGraph::NodeID addToGraph(TaskGraph::Builder &builder,
                                        Span<const NodeID> dependencies);
};

struct SortArchetypeNodeBase : NodeBase {
    SortArchetypeNodeBase(uint32_t archetype_id,
                          int32_t column_idx,
                          int32_t num_passes);

    void sortSetup(int32_t);
    void histogram(int32_t invocation_idx);

    TaskGraph::NodeID addToGraph(TaskGraph::Builder &builder,
                                 Span<const NodeID> dependencies,
                                 uint32_t archetype_id,
                                 int32_t component_id)

    uint32_t archetypeID;
    int32_t columnIDX;
    int32_t numPasses;
};

template <typename ArchetypeT, typename ComponentT>
struct SortArchetypeNode : SortArchetypeNodeBase {
    static TaskGraph::NodeID addToGraph(TaskGraph::Builder &builder,
                                        Span<const NodeID> dependencies);
};

}

#include "taskgraph.inl"
