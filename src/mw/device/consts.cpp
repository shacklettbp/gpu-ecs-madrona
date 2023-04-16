#include <madrona/mw_gpu/const.hpp>

#include <madrona/taskgraph.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#include <madrona/mw_gpu/tracing.hpp>

extern "C" {
__constant__ madrona::mwGPU::GPUImplConsts madronaMWGPUConsts;
}


extern "C" __global__ void madronaMWGPUComputeConstants(
    uint32_t num_worlds,
    uint32_t num_exports,
    uint32_t num_world_data_bytes,
    uint32_t world_data_alignment,
    madrona::mwGPU::GPUImplConsts *out_constants,
    size_t *job_system_buffer_size)
{
    using namespace madrona;
    using namespace madrona::mwGPU;

    uint64_t total_bytes = sizeof(TaskGraph) * num_worlds;

    uint64_t state_mgr_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(StateManager));

    total_bytes = state_mgr_offset + sizeof(StateManager) * num_worlds;

    uint64_t world_data_offset =
        utils::roundUp(total_bytes, (uint64_t)world_data_alignment);

    uint64_t total_world_bytes =
        (uint64_t)num_world_data_bytes * (uint64_t)num_worlds;

    total_bytes = world_data_offset + total_world_bytes;

    uint64_t host_allocator_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(mwGPU::HostAllocator));

    total_bytes = host_allocator_offset + sizeof(mwGPU::HostAllocator);

    uint64_t host_print_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(mwGPU::HostPrint));

    total_bytes = host_print_offset + sizeof(mwGPU::HostPrint);

    uint64_t tmp_allocator_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(TmpAllocator));

    total_bytes = tmp_allocator_offset + sizeof(TmpAllocator);

    uint64_t device_tracing_offset = utils::roundUp(
        total_bytes, (uint64_t)alignof(mwGPU::DeviceTracing));

    total_bytes = device_tracing_offset + sizeof(mwGPU::DeviceTracing);

    uint64_t export_ptrs_offset = utils::roundUp(
        total_bytes, (uint64_t)alignof(void *));

    total_bytes = export_ptrs_offset + sizeof(void *) * (uint64_t)num_worlds;

    uint64_t export_counters_offset = utils::roundUp(
        total_bytes, (uint64_t)alignof(uint32_t));

    total_bytes = export_counters_offset + sizeof(uint32_t) * (uint64_t)num_worlds;

    *out_constants = GPUImplConsts {
        /*.jobSystemAddr = */                 (void *)0ul,
        /* .taskGraph = */                    (void *)0ul,
        /* .stateManagerAddr = */             (void *)state_mgr_offset,
        /* .worldDataAddr =  */               (void *)world_data_offset,
        /* .hostAllocatorAddr = */            (void *)host_allocator_offset,
        /* .hostPrintAddr = */                (void *)host_print_offset,
        /* .tmpAllocatorAddr */               (void *)tmp_allocator_offset,
        /* .deviceTracingAddr = */            (void *)device_tracing_offset,
        /* .exportPointers = */               (void **)export_ptrs_offset,
        /* .exportCounts = */                 (uint32_t *)export_counters_offset,
        /* .numWorldDataBytes = */            num_world_data_bytes,
        /* .numWorlds = */                    num_worlds,
        /* .jobGridsOffset = */               (uint32_t)0,
        /* .jobListOffset = */                (uint32_t)0,
        /* .maxJobsPerGrid = */               0,
        /* .sharedJobTrackerOffset = */       (uint32_t)0,
        /* .userJobTrackerOffset = */         (uint32_t)0,
    };

    *job_system_buffer_size = total_bytes;
}

extern "C" __global__ void madronaMWGPUSetupExports(
    uint32_t num_exports,
    cuda::barrier<cuda::thread_scope_device> *barrier)
{
    //using namespace madrona;
    //using namespace madrona::mwGPU;

    //uint32_t num_worlds = GPUImplConsts::get().numWorlds;
}
