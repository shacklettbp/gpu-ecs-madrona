#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>
#include <madrona/mw_gpu/cu_utils.hpp>

#include "../render/interop.hpp"

namespace madrona::mwGPU {

#ifdef MADRONA_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-internal"
#endif
static inline __attribute__((always_inline)) void dispatch(
        uint32_t func_id,
        NodeBase *node_data,
        uint32_t invocation_offset);
#ifdef MADRONA_CLANG
#pragma clang diagnostic pop
#endif

}

namespace madrona {

void TaskGraph::execute()
{
   for (int node_idx = 0; node_idx < num_nodes_; node_idx++) {
        Node &cur_node = sorted_nodes_[node_idx];
        int32_t data_idx = cur_node.dataIDX;
        uint32_t func_id = cur_node.funcID;
        assert(cur_node.numThreadsPerInvocation == 1);

        NodeBase *node_data = (NodeBase *)node_datas_[data_idx].userData;
        mwGPU::dispatch(func_id, node_data, world_idx_);
    }
}

namespace mwGPU {

static inline __attribute__((always_inline)) void megakernelImpl()
{
    const int32_t world_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (world_idx >= GPUImplConsts::get().numWorlds) {
        return;
    }

    TaskGraph *taskgraph =
        &((TaskGraph *)GPUImplConsts::get().taskGraph)[world_idx];

    taskgraph->execute();
}

}

}
