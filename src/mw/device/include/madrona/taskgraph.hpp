#pragma once

#include <madrona/span.hpp>
#include <madrona/query.hpp>
#include <madrona/state.hpp>
#include <madrona/inline_array.hpp>

#include "mw_gpu/const.hpp"
#include "mw_gpu/worker_init.hpp"
#include "mw_gpu/megakernel_consts.hpp"

#include <cuda/barrier>
#include <cuda/std/tuple>

//#define LIMIT_ACTIVE_THREADS
// #define LIMIT_ACTIVE_BLOCKS
// #define FETCH_MULTI_INVOCATIONS

namespace madrona {

class TaskGraph;

struct NodeBase {
    uint32_t numDynamicInvocations;
};

class TaskGraph {
private:
    static inline constexpr uint32_t maxNodeDataBytes = 128;
    struct alignas(maxNodeDataBytes) NodeData {
        char userData[maxNodeDataBytes];
    };
    static_assert(sizeof(NodeData) == 128);

    struct Node {
        uint32_t dataIDX;
        uint32_t fixedCount;
        uint32_t funcID;
        uint32_t numChildren;
        uint32_t numThreadsPerInvocation;

        AtomicU32 curOffset;
        AtomicU32 numRemaining;
        AtomicU32 totalNumInvocations;
    };

public:
    struct NodeID {
        int32_t id;
    };

    struct DataID {
        int32_t id;
    };

    template <typename NodeT>
    struct TypedDataID : DataID {};

    class Builder {
    public:
        Builder(int32_t max_nodes,
                int32_t max_node_datas,
                int32_t max_num_dependencies,
                int32_t world_idx);
        ~Builder();

        template <typename NodeT, typename... Args>
        TypedDataID<NodeT> constructNodeData(Args &&...args);

        template <auto fn, typename NodeT>
        NodeID addNodeFn(TypedDataID<NodeT> data,
                         Span<const NodeID> dependencies,
                         Optional<NodeID> parent_node =
                             Optional<NodeID>::none(),
                         uint32_t fixed_num_invocations = 0,
                         uint32_t num_threads_per_invocation = 1);

        template <typename NodeT, int32_t count = 1, typename... Args>
        NodeID addOneOffNode(Span<const NodeID> dependencies,
                             Args && ...args);

        template <typename NodeT, typename... Args>
        NodeID addDynamicCountNode(Span<const NodeID> dependencies,
                                   uint32_t num_threads_per_invocation,
                                   Args && ...args);

        template <typename NodeT>
        inline NodeID addToGraph(Span<const NodeID> dependencies);

        template <typename NodeT>
        NodeT & getDataRef(TypedDataID<NodeT> data_id);

        void build(TaskGraph *out);

        inline int32_t worldIDX() const { return world_idx_; }

    private:
        template <typename NodeT>
        static void dynamicCountWrapper(NodeT *node, int32_t);

        NodeID registerNode(uint32_t data_idx,
                            uint32_t fixed_count,
                            uint32_t num_threads_per_invocation,
                            uint32_t func_id,
                            Span<const NodeID> dependencies,
                            Optional<NodeID> parent_node);

        struct StagedNode {
            Node node;
            int32_t parentID;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        StagedNode *staged_;
        int32_t num_nodes_;
        NodeData *node_datas_;
        int32_t num_datas_;
        NodeID *all_dependencies_;
        uint32_t num_dependencies_;
        int32_t world_idx_;
    };

    enum class WorkerState {
        Run,
        PartialRun,
        Loop,
        Exit,
    };

    TaskGraph(const TaskGraph &) = delete;

    ~TaskGraph();

    inline void execute();

    static inline WorldBase * getWorld(int32_t world_idx);

    template <typename ContextT>
    static ContextT makeContext(WorldID world_id);

    static void setupRenderer(Context &ctx, const void *renderer_inits,
                              int32_t world_idx);

    template <typename NodeT>
    NodeT & getNodeData(TypedDataID<NodeT> data_id);

private:
    template <typename ContextT, bool> struct WorldTypeExtract;

    TaskGraph(Node *nodes, uint32_t num_nodes, NodeData *node_datas,
              int32_t world_idx);

    Node *sorted_nodes_;
    uint32_t num_nodes_;
    NodeData *node_datas_;
    int32_t world_idx_;

friend class Builder;
};

template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
class CustomParallelForNode: public NodeBase {
public:
    CustomParallelForNode(int32_t world_idx);

    inline void run(const int32_t invocation_idx);

    static TaskGraph::NodeID addToGraph(
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);

private:
    QueryRef *query_ref_;
};

template <typename ContextT, auto Fn, typename ...ComponentTs>
using ParallelForNode =
    CustomParallelForNode<ContextT, Fn, 1, 1, ComponentTs...>;

struct ClearTmpNodeBase : NodeBase {
    ClearTmpNodeBase(uint32_t archetype_id);

    void run(int32_t world_idx);

    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies,
        uint32_t archetype_id);

    uint32_t archetypeID;
};

template <typename ArchetypeT>
struct ClearTmpNode : ClearTmpNodeBase {
    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies);
};

struct ResetTmpAllocNode : NodeBase {
    void run(int32_t world_idx);

    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies);
};

}

#include "taskgraph.inl"
