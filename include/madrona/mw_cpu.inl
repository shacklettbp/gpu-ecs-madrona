#pragma once

namespace madrona {

template <typename ContextT, typename WorldT, typename... InitTs>
template <typename... Args>
TaskGraphExecutor<ContextT, WorldT, InitTs...>::TaskGraphExecutor(
        const Config &cfg,
        const Args * ...user_init_ptrs)
    : ThreadPoolExecutor(cfg),
      world_contexts_(cfg.numWorlds),
      jobs_(cfg.numWorlds)
{
    auto ecs_reg = getECSRegistry();
    WorldT::registerTypes(ecs_reg);

    auto renderer_iface = getRendererInterface();

    for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
        std::array<void *, sizeof...(InitTs)> init_ptrs {
            (void *)&user_init_ptrs[i] ...,
            renderer_iface.has_value() ? (void *)&(*renderer_iface) : nullptr,
        };

        // FIXME: this is super ugly because WorkerInit
        // isn't available at the header level
        auto cb = [&](const WorkerInit &worker_init) {
            std::apply([&](auto ...ptrs) {
                new (&world_contexts_[i]) WorldContext(
                    worker_init, *(InitTs *)ptrs ...);
            }, init_ptrs);
        };

        using CBPtrT = decltype(&cb);

        auto wrapper = [](void *ptr_raw, const WorkerInit &worker_init) {
            (*(CBPtrT)ptr_raw)(worker_init);
        };

        ctxInit(wrapper, &cb, i);
        jobs_[i].fn = [](void *ptr) {
            stepWorld(ptr);
        };
        jobs_[i].data = &world_contexts_[i];
    }
}

template <typename ContextT, typename WorldT, typename... InitTs>
void TaskGraphExecutor<ContextT, WorldT, InitTs...>::run()
{
    ThreadPoolExecutor::run(jobs_.data(), jobs_.size());
}

template <typename ContextT, typename WorldT, typename... InitTs>
TaskGraphExecutor<ContextT, WorldT, InitTs...>::WorldContext::WorldContext(
        const WorkerInit &worker_init,
        const InitTs & ...world_inits)
    : ctx(&worldData, worker_init),
      worldData(ctx, world_inits...),
      taskgraph([this]() {
          TaskGraph::Builder builder(ctx);
          WorldT::setupTasks(builder);
          return builder.build();
      }())
{}

template <typename ContextT, typename WorldT, typename... InitTs>
void TaskGraphExecutor<ContextT, WorldT, InitTs...>::stepWorld(void *data_raw)
{
    auto world_ctx = (WorldContext *)data_raw;
    world_ctx->taskgraph.run(&world_ctx->ctx);
}

}