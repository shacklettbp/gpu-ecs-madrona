#pragma once

#include <madrona/mw_gpu/host_print.hpp>

namespace madrona {

namespace mwGPU {

template <typename NodeT, auto fn>
__attribute__((used, always_inline))
inline void userEntry(NodeBase *data_ptr, int32_t invocation_idx)
{
    auto node = (NodeT *)data_ptr;

    std::invoke(fn, node, invocation_idx);
}

template <typename NodeT, auto fn>
struct UserFuncIDBase {
    static uint32_t id;
};

template <typename NodeT,
          auto fn,
          decltype(userEntry<NodeT, fn>) = userEntry<NodeT, fn>>
struct UserFuncID : UserFuncIDBase<NodeT, fn> {};

}

template <typename ContextT, bool = false>
struct TaskGraph::WorldTypeExtract {
    using type = typename ContextT::WorldDataT;
};

template <bool ignore>
struct TaskGraph::WorldTypeExtract<Context, ignore> {
    using type = WorldBase;
};

template <typename NodeT, typename... Args> 
TaskGraph::TypedDataID<NodeT> TaskGraph::Builder::constructNodeData(
    Args && ...args)
{
    static_assert(sizeof(NodeT) <= maxNodeDataBytes);
    static_assert(alignof(NodeT) <= maxNodeDataBytes);

    int32_t data_idx = num_datas_++;
    new (&node_datas_[data_idx]) NodeT(std::forward<Args>(args)...);

    return TypedDataID<NodeT> {
        DataID { data_idx },
    };
}

template <auto fn, typename NodeT>
TaskGraph::NodeID TaskGraph::Builder::addNodeFn(
        TypedDataID<NodeT> data,
        Span<const NodeID> dependencies,
        Optional<NodeID> parent_node,
        uint32_t fixed_num_invocations,
        uint32_t num_threads_per_invocation)
{
    using namespace mwGPU;

    uint32_t func_id = mwGPU::UserFuncID<NodeT, fn>::id;

    return registerNode(uint32_t(data.id),
                        fixed_num_invocations,
                        num_threads_per_invocation,
                        func_id,
                        dependencies,
                        parent_node);
}

template <typename NodeT, int32_t count, typename... Args>
TaskGraph::NodeID TaskGraph::Builder::addOneOffNode(
    Span<const NodeID> dependencies,
    Args && ...args)
{
    auto data_id = constructNodeData<NodeT>(
        std::forward<Args>(args)...);
    return addNodeFn<&NodeT::run>(data_id, dependencies,
                                  Optional<NodeID>::none(), count);
}

template <typename NodeT>
void TaskGraph::Builder::dynamicCountWrapper(NodeT *node, int32_t)
{
    int32_t num_invocations = node->numInvocations();
    node->numDynamicInvocations = num_invocations;
}

template <typename NodeT, typename... Args>
TaskGraph::NodeID TaskGraph::Builder::addDynamicCountNode(
    Span<const NodeID> dependencies,
    uint32_t num_threads_per_invocation,
    Args && ...args)
{
    auto data_id = constructNodeData<NodeT>(
        std::forward<Args>(args)...);

    NodeID count_node = addNodeFn<&Builder::dynamicCountWrapper<NodeT>>(
        data_id, dependencies, Optional<NodeID>::none(), 1);

    return addNodeFn<&NodeT::run>(data_id, {count_node},
        Optional<NodeID>::none(), 0, num_threads_per_invocation);
}

template <typename NodeT>
TaskGraph::NodeID TaskGraph::Builder::addToGraph(
    Span<const NodeID> dependencies)
{
    return NodeT::addToGraph(*this, dependencies);
}

template <typename NodeT>
NodeT & TaskGraph::Builder::getDataRef(TypedDataID<NodeT> data_id)
{
    return *(NodeT *)node_datas_[data_id.id].userData;
}

WorldBase * TaskGraph::getWorld(int32_t world_idx)
{
    const auto &consts = mwGPU::GPUImplConsts::get();
    auto world_ptr = (char *)consts.worldDataAddr +
        world_idx * (int32_t)consts.numWorldDataBytes;

    return (WorldBase *)world_ptr;
}

template <typename ContextT>
ContextT TaskGraph::makeContext(WorldID world_id)
{
    using WorldDataT = typename WorldTypeExtract<ContextT>::type;

    auto world = TaskGraph::getWorld(world_id.idx);
    return ContextT((WorldDataT *)world, WorkerInit {
        world_id,
    });
}

template <typename NodeT>
NodeT & TaskGraph::getNodeData(TypedDataID<NodeT> data_id)
{
    return *(NodeT *)node_datas_[data_id.id].userData;
}

template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
CustomParallelForNode<ContextT, Fn, threads_per_invocation,
                      items_per_invocation, ComponentTs...>::
CustomParallelForNode(int32_t world_idx)
    : NodeBase {},
      query_ref_([world_idx]() {
          StateManager *state_mgr = 
              &((StateManager *)mwGPU::GPUImplConsts::get().stateManagerAddr)[world_idx];
          auto query = state_mgr->query<ComponentTs...>();

          QueryRef *query_ref = query.getSharedRef();
          query_ref->numReferences.fetch_add_relaxed(1);

          return query_ref;
      }())
{}


template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
void CustomParallelForNode<ContextT, Fn,
                           threads_per_invocation,
                           items_per_invocation,
                           ComponentTs...>::run(const int32_t invocation_idx)
{
    int32_t world_idx = invocation_idx;
    StateManager *state_mgr = 
        &((StateManager *)mwGPU::GPUImplConsts::get().stateManagerAddr)[world_idx];
    ContextT ctx = TaskGraph::makeContext<ContextT>(WorldID {world_idx});

    state_mgr->iterateArchetypesRaw<sizeof...(ComponentTs)>(query_ref_,
        [&](int32_t num_rows, auto ...raw_ptrs) {
            // The following should work, but doesn't in cuda 11.7 it seems
            // Need to put arguments in tuple for some reason instead
            //Fn(ctx, ((ComponentTs *)raw_ptrs)[tbl_offset] ...);
            
            cuda::std::tuple typed_ptrs {
                (ComponentTs *)raw_ptrs
                ...
            };
            
            std::apply([&](auto ...ptrs) {
                for (int32_t i = 0; i < num_rows; i++) {
                    Fn(ctx, ptrs[i] ...);
                }
            }, typed_ptrs);

            return false;
        });
}

template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
TaskGraph::NodeID CustomParallelForNode<ContextT, Fn,
                                        threads_per_invocation,
                                        items_per_invocation,
                                        ComponentTs...>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return builder.addOneOffNode<
        CustomParallelForNode<
            ContextT, Fn,
            threads_per_invocation,
            items_per_invocation,
            ComponentTs...>>(dependencies,
                             builder.worldIDX());
}

template <typename ArchetypeT>
TaskGraph::NodeID ClearTmpNode<ArchetypeT>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return ClearTmpNodeBase::addToGraph(builder, dependencies,
        TypeTracker::typeID<ArchetypeT>());
}

}
