#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/memory.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>
#include <madrona/mw_gpu/cu_utils.hpp>

#include "../render/interop.hpp"

namespace madrona {

TaskGraph::TaskGraph(Node *nodes,
                     uint32_t num_nodes,
                     NodeData *node_datas,
                     int32_t world_idx)
    : sorted_nodes_(nodes),
      num_nodes_(num_nodes),
      node_datas_(node_datas),
      world_idx_(world_idx)
{
}

TaskGraph::~TaskGraph()
{
    rawDealloc(sorted_nodes_);
}

void TaskGraph::setupRenderer(Context &ctx, const void *renderer_inits,
                              int32_t world_idx)
{
    const render::RendererInit &renderer_init =
        ((const render::RendererInit *)renderer_inits)[world_idx];

    render::RendererState::init(ctx, renderer_init);
}

}
