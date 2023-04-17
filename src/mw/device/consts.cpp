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

    uint64_t host_print_offset =
        utils::roundUp(total_bytes, (uint64_t)alignof(mwGPU::HostPrint));

    total_bytes = host_print_offset + sizeof(mwGPU::HostPrint);

    uint64_t export_ptrs_offset = utils::roundUp(
        total_bytes, (uint64_t)alignof(void *));

    total_bytes = export_ptrs_offset + sizeof(void *) * (uint64_t)num_exports;

    uint64_t export_counters_offset = utils::roundUp(
        total_bytes, (uint64_t)alignof(uint32_t));

    total_bytes = export_counters_offset + sizeof(uint32_t) * (uint64_t)num_exports;

    *out_constants = GPUImplConsts {
        .jobSystemAddr =                (void *)0ul,
        .taskGraph =                    (void *)0ul,
        .stateManagerAddr =             (void *)state_mgr_offset,
        .worldDataAddr =                (void *)world_data_offset,
        .hostAllocatorAddr =            (void *)0ul,
        .hostPrintAddr =                (void *)host_print_offset,
        .tmpAllocatorAddr =             (void *)0ul,
        .deviceTracingAddr =            (void *)0ul,
        .exportPointers =               (void **)export_ptrs_offset,
        .exportCounts =                 (uint32_t *)export_counters_offset,
        .numWorldDataBytes =            num_world_data_bytes,
        .numWorlds =                    num_worlds,
        .jobGridsOffset =               (uint32_t)0,
        .jobListOffset =                (uint32_t)0,
        .maxJobsPerGrid =               0,
        .sharedJobTrackerOffset =       (uint32_t)0,
        .userJobTrackerOffset =         (uint32_t)0,
    };

    *job_system_buffer_size = total_bytes;
}

extern "C" __global__ void madronaMWGPUExportBarrierSetup(
    uint32_t num_exports,
    cuda::barrier<cuda::thread_scope_device> *barrier)
{
    using namespace madrona;
    using namespace madrona::mwGPU;

    assert(sizeof(cuda::barrier<cuda::thread_scope_device>) <= 64);

    const int32_t num_worlds = GPUImplConsts::get().numWorlds;
    int32_t blocks_per_export = utils::divideRoundUp(num_worlds, 8);

    new (barrier) cuda::barrier<cuda::thread_scope_device>(
        blocks_per_export * num_exports);
}

extern "C" __global__ void madronaMWGPUExportCopyIn(
    uint32_t num_exports)
{
    using namespace madrona;
    using namespace madrona::mwGPU;

    const int32_t num_worlds = GPUImplConsts::get().numWorlds;

    constexpr int32_t threads_per_world = 32;
    const int32_t lane_idx = threadIdx.x % 32;

    int32_t world_idx =
        (blockIdx.x * blockDim.x + threadIdx.x) / threads_per_world;

    int32_t export_idx = blockIdx.y;

    if (world_idx >= num_worlds || export_idx >= num_exports) {
        return;
    }

    StateManager *state_mgr = 
        &((StateManager *)mwGPU::GPUImplConsts::get().stateManagerAddr)[world_idx];

    int32_t export_offset = state_mgr->exportedData[export_idx].offset;
    int32_t num_rows_export = state_mgr->getExportNumRows(export_idx);

    char *dst_export_ptr = (char *)state_mgr->getExportColumnPtr(export_idx);
    uint32_t export_column_size = state_mgr->getExportColumnSize(export_idx);

    char *src_export_ptr =
        (char *)GPUImplConsts::get().exportPointers[export_idx];

    for (int32_t i = 0; i < num_rows_export; i += 32) {
        int32_t idx = i + lane_idx;

        if (idx >= num_rows_export) {
            continue;
        }

        memcpy(dst_export_ptr + export_column_size * idx,
               src_export_ptr + export_column_size * (export_offset + idx), 
               export_column_size);
    }
}

extern "C" __global__ void madronaMWGPUExportBlockSums(
    uint32_t num_exports,
    uint32_t *block_sums)
{
    __shared__ uint32_t prefix_smem[8];

    using namespace madrona;
    using namespace madrona::mwGPU;

    const int32_t num_worlds = GPUImplConsts::get().numWorlds;

    constexpr int32_t threads_per_world = 32;
    const int32_t warp_idx = threadIdx.x / 32;
    const int32_t lane_idx = threadIdx.x % 32;

    int32_t world_idx =
        (blockIdx.x * blockDim.x + threadIdx.x) / threads_per_world;

    int32_t export_idx = blockIdx.y;

    if (world_idx >= num_worlds || export_idx >= num_exports) {
        if (lane_idx == 0) {
            prefix_smem[warp_idx] = 0;
        }

        return;
    }

    StateManager *state_mgr = 
        &((StateManager *)mwGPU::GPUImplConsts::get().stateManagerAddr)[world_idx];

    int32_t num_rows_export = state_mgr->getExportNumRows(export_idx);

    if (lane_idx == 0) {
        prefix_smem[warp_idx] = num_rows_export;
    }

    __syncthreads();

    int32_t blocks_per_export = utils::divideRoundUp(num_worlds, 8);
    int32_t block_slot = blockIdx.x + export_idx * blocks_per_export;

    if (warp_idx == 0 && lane_idx == 0) {
        int32_t local_sum = 0;
        for (int32_t i = 0; i < 8; i++) {
            int32_t cur = prefix_smem[i];
            local_sum += cur;
        }

        block_sums[block_slot] = local_sum;
    }
}

extern "C" __global__ void madronaMWGPUExportCopyOut(
    uint32_t num_exports,
    uint32_t *block_sums)
{
    __shared__ uint32_t prefix_smem[8];

    using namespace madrona;
    using namespace madrona::mwGPU;

    const int32_t num_worlds = GPUImplConsts::get().numWorlds;

    constexpr int32_t threads_per_world = 32;
    const int32_t warp_idx = threadIdx.x / 32;
    const int32_t lane_idx = threadIdx.x % 32;

    int32_t world_idx =
        (blockIdx.x * blockDim.x + threadIdx.x) / threads_per_world;

    int32_t export_idx = blockIdx.y;

    if (world_idx >= num_worlds || export_idx >= num_exports) {
        if (lane_idx == 0) {
            prefix_smem[warp_idx] = 0;
        }

        return;
    }

    StateManager *state_mgr = 
        &((StateManager *)mwGPU::GPUImplConsts::get().stateManagerAddr)[world_idx];

    int32_t num_rows_export = state_mgr->getExportNumRows(export_idx);

    if (lane_idx == 0) {
        prefix_smem[warp_idx] = num_rows_export;
    }

    __syncthreads();

    int32_t blocks_per_export = utils::divideRoundUp(num_worlds, 8);
    int32_t block_slot = blockIdx.x + export_idx * blocks_per_export;

    if (warp_idx == 0 && lane_idx == 0) {
        uint32_t prev_block_sum = 0;
        int32_t sum_start = export_idx * blocks_per_export;

        for (int32_t i = sum_start; i < block_slot; i++) {
            prev_block_sum += block_sums[i];
        }

        int32_t local_sum = prev_block_sum;
        for (int32_t i = 0; i < 8; i++) {
            int32_t cur = prefix_smem[i];
            prefix_smem[i] = local_sum;
            local_sum += cur;
        }
    }

    __syncthreads();

    int32_t export_offset = prefix_smem[warp_idx];

    if (lane_idx == 0) {
        state_mgr->exportedData[export_idx].offset = export_offset;
    }

    char *src_export_ptr = (char *)state_mgr->getExportColumnPtr(export_idx);
    uint32_t export_column_size = state_mgr->getExportColumnSize(export_idx);

    char *dst_export_ptr =
        (char *)GPUImplConsts::get().exportPointers[export_idx];

    for (int32_t i = 0; i < num_rows_export; i += 32) {
        int32_t idx = i + lane_idx;

        if (idx >= num_rows_export) {
            continue;
        }

        memcpy(dst_export_ptr + export_column_size * (export_offset + idx),
               src_export_ptr + export_column_size * idx, 
               export_column_size);
    }
}
