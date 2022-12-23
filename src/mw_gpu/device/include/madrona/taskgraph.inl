#pragma once

namespace madrona {

namespace mwGPU {

static_assert(sizeof(NodeData) == 64);

template <typename NodeT, auto fn>
__attribute__((used, always_inline))
inline void userEntry(void *data_ptr, int32_t invocation_idx)
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
struct UserFuncID : FuncIDBase<NodeT, fn> {};

template <typename ContextT, bool = false>
struct WorldTypeExtract {
    using type = typename ContextT::WorldDataT;
};

template <bool ignore>
struct WorldTypeExtract<Context, ignore> {
    using type = WorldBase;
};

}

template <typename NodeT, typename... Args> 
NodeT & TaskGraph::Builder::constructNodeData(Args &&..args)
{
    static_assert(sizeof(NodeT) <= maxNodeDataBytes);
    static_assert(alignof(NodeT) <= maxNodeDataBytes);

    uint32_t data_idx = num_datas_++;
    NodeT *constructed = 
        new (&node_datas_[data_idx]) NodeT(std::forward<Args>(args)...);

    return *constructed;
}

template <auto fn, typename NodeT>
TaskGraph::NodeID TaskGraph::Builder::addNodeFn(
        const NodeT &node,
        Span<const NodeID> dependencies,
        int32_t fixed_num_invocations = 0)
{
    using namespace mwGPU;

    NodeDataT *data_ptr = (NodeDataT *)&node;

    int64_t data_offset = data_ptr - node_datas_;

    assert(data_offset >= 0 && data_offset < num_datas_);

    uint32_t func_id = mwGPU::UserFuncID<NodeT, fn>::id;

    Node node;
    node.dataIdx = uint32_t(data_offset);
    node.fixedCount = fixed_num_invocations;
    node.funcID = func_id;
    node.curOffset.store(0, std::memory_order_relaxed);
    node.numRemaining.store(0, std::memory_order_relaxed);
    node.totalNumInvocations.store(0, std::memory_order_relaxed);

    return registerNode(std::move(node), dependencies);
}

template <typename NodeT, int32_t count, typename... Args>
TaskGraph::NodeID TaskGraph::Builder::addOneOffNode(
    Span<const NodeID> dependencies,
    Args && ...args)
{
    auto &node = builder.constructNodeData<NodeT>(
        std::forward<Args>(args)...);
    return builder.addNodeFn<NodeT::run>(node, dependencies, count);
}

template <typename NodeT, typename... Args>
TaskGraph::NodeID TaskGraph::Builder::addDynamicCountNode(
    Span<const NodeID> dependencies,
    Args && ...args)
{
    auto count_wrapper = [](NodeT *node) {
        int32_t num_invocations = node->numInvocations();
        node->numDynamicInvocations = num_invocations;
    };

    auto &node_data = builder.constructNodeData<NodeT>();

    NodeID count_node =
        builder.addNodeFn<count_wrapper>(node, dependencies, 1);

    return builder.addNodeFn<NodeT::run>(node, {count_node});
}

template <typename NodeT>
TaskGraph::NodeID TaskGraph::Builder::addToGraph(
    Span<const NodeID> dependencies)
{
    return NodeT::addToGraph(*this, dependencies);
}

template <typename ContextT, auto Fn, typename ...ComponentTs>
ParallelForNode::ParallelForNode()
    : DynamicCountNode {},
      query_ref_([]() {
          auto query = mwGPU::getStateManager()->query<ComponentTs...>();
          QueryRef *query_ref = query.getSharedRef();
          query_ref->numReferences.fetch_add(1, std::memory_order_relaxed);

          return query_ref;
      }())
{}

template <typename ContextT, auto Fn, typename ...ComponentTs>
void ParallelForNode<ContextT, Fn, ComponentTs...>::run(int32_t invocation_idx)
{
    StateManager *state_mgr = mwGPU::getStateManager();

    int32_t cumulative_num_rows = 0;
    state_mgr->iterateArchetypesRaw<sizeof...(ComponentTs)>(query_ref_,
            [&](int32_t num_rows, WorldID *world_column,
                auto ...raw_ptrs) {
        int32_t tbl_offset = invocation_idx - cumulative_num_rows;
        cumulative_num_rows += num_rows;
        if (tbl_offset >= num_rows) {
            return false;
        }

        WorldID world_id = world_column[tbl_offset];

        // This entity has been deleted but not actually removed from the
        // table yet
        if (world_id.idx == -1) {
            return true;
        }

        ContextT ctx = makeContext(world_id);

        // The following should work, but doesn't in cuda 11.7 it seems
        // Need to put arguments in tuple for some reason instead
        //Fn(ctx, ((ComponentTs *)raw_ptrs)[tbl_offset] ...);

        cuda::std::tuple typed_ptrs {
            (ComponentTs *)raw_ptrs
            ...
        };

        std::apply([&](auto ...ptrs) {
            Fn(ctx, ptrs[tbl_offset] ...);
        }, typed_ptrs);

        return true;
    });
}

template <typename ContextT, auto Fn, typename ...ComponentTs>
uint32_t ParallelForNode<ContextT, Fn, ComponentTs...>::numInvocations()
{
    StateManager *state_mgr = mwGPU::getStateManager();
    return state_mgr->numMatchingEntities(query_ref_);
}

template <typename ContextT, auto Fn, typename ...ComponentTs>
ContextT ParallelForNode<ContextT, Fn, ComponentTs...>::makeContext(
        WorldID world_id)
{
    using WorldDataT = mwGPU::WorldTypeExtract<ContextT>::type;

    auto world = TaskGraph::getWorld(world_id.idx);
    return ContextT((WorldDataT *)world, WorkerInit {
        world_id,
    });
}


template <typename ContextT, auto Fn, typename ...ComponentTs>
TaskGraph::NodeID ParallelForNode<ContextT, Fn, ComponentTs...>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const NodeID> dependencies)
{
    return builder.addDynamicCountNode<
        ParallelForNode<ContextT, Fn, ComponentTs...>>(dependencies);
}

template <typename ArchetypeT>
TaskGraph::NodeID ClearTmpNode<ArchetypeT>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const NodeID> dependencies)
{
    return addToGraph(builder, dependencies,
                      TypeTracker::typeID<ArchetypeT>());
}

template <typename ArchetypeT>
TaskGraph::NodeID CompactArchetypeNode<ArchetypeT>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const NodeID> dependencies)
{
    return addToGraph(builder, dependencies,
                      TypeTracker::typeID<ArchetypeT>());
}

template <typename ArchetypeT, typename ComponentT>
TaskGraph::NodeID SortArchetypeNode<ArchetypeT>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const NodeID> dependencies)
{
    return addToGraph(builder, dependencies,
        TypeTracker::typeID<ArchetypeT>(),
        TypeTracker::typeID<ComponentT>());
}

}
