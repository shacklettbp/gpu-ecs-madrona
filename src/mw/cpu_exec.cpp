#include <madrona/mw_cpu.hpp>
#include "../core/worker_init.hpp"

#include "render/batch_renderer.hpp"

#if defined(MADRONA_LINUX) or defined(MADRONA_MACOS)
#include <unistd.h>
#endif

namespace madrona {

struct ThreadPoolExecutor::Impl {
    HeapArray<std::thread> workers;
    alignas(MADRONA_CACHE_LINE) AtomicI32 workerWakeup;
    alignas(MADRONA_CACHE_LINE) AtomicI32 mainWakeup;
    ThreadPoolExecutor::Job *currentJobs;
    uint32_t numJobs;
    alignas(MADRONA_CACHE_LINE) AtomicU32 nextJob;
    alignas(MADRONA_CACHE_LINE) AtomicU32 numFinished;
    StateManager stateMgr;
    HeapArray<StateCache> stateCaches;
    HeapArray<void *> exportPtrs;
    Optional<render::BatchRenderer> renderer;

    static Impl * make(const ThreadPoolExecutor::Config &cfg);
    ~Impl();
    void run(Job *jobs, CountT num_jobs);
    void workerThread(CountT worker_id);
};

static CountT getNumCores()
{
#if defined(MADRONA_MACOS)
    int os_num_threads = sysconf(_SC_NPROCESSORS_ONLN);

    if (os_num_threads <= 0) {
        FATAL("Failed to get number of concurrent threads");
    }

    return os_num_threads;
#elif defined(MADRONA_LINUX)
    cpu_set_t cpuset;
    pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    CountT num_active_threads = CPU_COUNT(&cpuset);
    if (num_active_threads <= 0) {
        FATAL("Failed to get number of concurrent threads");
    }

    return num_active_threads;
#else
    STATIC_UNIMPLEMENTED();
#endif

}

static inline void pinThread([[maybe_unused]] CountT worker_id)
{
#ifdef MADRONA_LINUX
    cpu_set_t cpu_set;
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);

    const int max_threads = CPU_COUNT(&cpu_set);

    if (worker_id > max_threads) [[unlikely]] {
        FATAL("Tried setting thread affinity to %d when %d is max",
              worker_id, max_threads);
    }

    cpu_set_t worker_set;
    CPU_ZERO(&worker_set);
    
    // This is needed in case there was already cpu masking via
    // a different call to setaffinity or via cgroup (SLURM)
    for (CountT thread_idx = 0, available_threads = 0;
         thread_idx < (CountT)CPU_SETSIZE; thread_idx++) {
        if (CPU_ISSET(thread_idx, &cpu_set)) {
            if ((available_threads++) == worker_id) {
                CPU_SET(thread_idx, &worker_set);

                break;
            }
        }
    }

    int res = pthread_setaffinity_np(pthread_self(),
                                     sizeof(worker_set),
                                     &worker_set);

    if (res != 0) {
        FATAL("Failed to set thread affinity to %d", worker_id);
    }
#endif
}

static Optional<render::BatchRenderer> makeRenderer(
    const ThreadPoolExecutor::Config &cfg)
{
    if (cfg.cameraMode == render::CameraMode::None) {
        return Optional<render::BatchRenderer>::none();
    }

    return Optional<render::BatchRenderer>::make(render::BatchRenderer::Config {
        .gpuID = cfg.renderGPUID,
        .renderWidth = cfg.renderWidth,
        .renderHeight = cfg.renderHeight,
        .numWorlds = cfg.numWorlds,
        .maxViewsPerWorld = cfg.maxViewsPerWorld,
        .maxInstancesPerWorld = cfg.maxInstancesPerWorld,
        .maxObjects = cfg.maxObjects,
        .cameraMode = cfg.cameraMode,
        .inputMode = render::BatchRenderer::InputMode::CPU,
    });
}

ThreadPoolExecutor::Impl * ThreadPoolExecutor::Impl::make(
    const ThreadPoolExecutor::Config &cfg)
{
    Impl *impl = new Impl {
        .workers = HeapArray<std::thread>(
            cfg.numWorkers == 0 ? getNumCores() : cfg.numWorkers),
        .workerWakeup = 0,
        .mainWakeup = 0,
        .currentJobs = nullptr,
        .numJobs = 0,
        .nextJob = 0,
        .numFinished = 0,
        .stateMgr = StateManager(cfg.numWorlds),
        .stateCaches = HeapArray<StateCache>(cfg.numWorlds),
        .exportPtrs = HeapArray<void *>(cfg.numExportedBuffers),
        .renderer = makeRenderer(cfg),
    };

    for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
        impl->stateCaches.emplace(i);
    }

    for (CountT i = 0; i < impl->workers.size(); i++) {
        impl->workers.emplace(i, [](Impl *impl, CountT i) {
            impl->workerThread(i);
        }, impl, i);
    }

    return impl;
}

ThreadPoolExecutor::ThreadPoolExecutor(const Config &cfg)
    : impl_(Impl::make(cfg))
{}

ThreadPoolExecutor::Impl::~Impl()
{
    workerWakeup.store_release(-1);
    workerWakeup.notify_all();

    for (CountT i = 0; i < workers.size(); i++) {
        workers[i].join();
    }
}

ThreadPoolExecutor::~ThreadPoolExecutor() = default;

void ThreadPoolExecutor::Impl::run(Job *jobs, CountT num_jobs)
{
    stateMgr.copyInExportedColumns();

    currentJobs = jobs;
    numJobs = uint32_t(num_jobs);
    nextJob.store_relaxed(0);
    numFinished.store_relaxed(0);
    workerWakeup.store_release(1);
    workerWakeup.notify_all();

    mainWakeup.wait<sync::acquire>(0);
    mainWakeup.store_relaxed(0);

    stateMgr.copyOutExportedColumns();

    if (renderer.has_value()) {
        renderer->render();
    }
}

void ThreadPoolExecutor::run(Job *jobs, CountT num_jobs)
{
    impl_->run(jobs, num_jobs);
}

CountT ThreadPoolExecutor::loadObjects(Span<const imp::SourceObject> objs)
{
    return impl_->renderer->loadObjects(objs);
}

uint8_t * ThreadPoolExecutor::rgbObservations() const
{
    return impl_->renderer->rgbPtr();
}

float * ThreadPoolExecutor::depthObservations() const
{
    return impl_->renderer->depthPtr();
}

void * ThreadPoolExecutor::getExported(CountT slot) const
{
    return impl_->exportPtrs[slot];
}

void ThreadPoolExecutor::initializeContexts(
    Context & (*init_fn)(void *, const WorkerInit &, CountT),
    void *init_data, CountT num_worlds)
{
    render::WorldGrid render_grid(num_worlds, 220.f /* FIXME */);

    for (CountT world_idx = 0; world_idx < num_worlds; world_idx++) {
        WorkerInit worker_init {
            &impl_->stateMgr,
            &impl_->stateCaches[world_idx],
            uint32_t(world_idx),
        };

        Context &ctx = init_fn(init_data, worker_init, world_idx);

        if (impl_->renderer.has_value()) {
            render::RendererInit renderer_init {
                impl_->renderer->getInterface(),
                render_grid.getOffset(world_idx),
            };

            render::RendererState::init(ctx, renderer_init);
        }
    }
}

ECSRegistry ThreadPoolExecutor::getECSRegistry()
{
    return ECSRegistry(&impl_->stateMgr, impl_->exportPtrs.data());
}

void ThreadPoolExecutor::initExport()
{
    impl_->stateMgr.copyOutExportedColumns();
}

void ThreadPoolExecutor::Impl::workerThread(CountT worker_id)
{
    pinThread(worker_id);

    while (true) {
        workerWakeup.wait<sync::relaxed>(0);
        int32_t ctrl = workerWakeup.load_acquire();

        if (ctrl == 0) {
            continue;
        } else if (ctrl == -1) {
            break;
        }

        while (true) {
            uint32_t job_idx = nextJob.fetch_add_relaxed(1);

            if (job_idx == numJobs) {
                workerWakeup.store_relaxed(0);
            }

            assert(job_idx < 0xFFFF'FFFF);

            if (job_idx >= numJobs) {
                break;
            }

            currentJobs[job_idx].fn(currentJobs[job_idx].data);

            // This has to be acq_rel so the finishing thread has seen
            // all the other threads' effects
            uint32_t prev_finished =
                numFinished.fetch_add_acq_rel(1);

            if (prev_finished == numJobs - 1) {
                mainWakeup.store_release(1);
                mainWakeup.notify_one();
            }
        }
    }
}

}
