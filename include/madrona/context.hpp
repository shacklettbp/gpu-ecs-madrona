/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/job.hpp>
#include <madrona/ecs.hpp>
#include <madrona/state.hpp>
#include <madrona/io.hpp>

namespace madrona {

class Context {
public:
    Context(WorldBase *world_data, const WorkerInit &init);
    Context(const Context &) = delete;

    AllocContext mem;

    // Registration
    template <typename ComponentT>
    void registerComponent();

    template <typename ArchetypeT>
    void registerArchetype();

    // State
    template <typename ArchetypeT>
    inline ArchetypeRef<ArchetypeT> archetype();

    inline Loc getLoc(Entity e) const;

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntity(Transaction &txn, Args && ...args);

    template <typename ArchetypeT, typename... Args>
    inline Entity makeEntityNow(Args && ...args);

    inline void destroyEntity(Transaction &txn, Entity e);

    inline void destroyEntityNow(Entity e);

    template <typename ArchetypeT>
    inline Loc makeTemporary();

    template <typename ComponentT>
    inline ResultRef<ComponentT> get(Entity e);

    template <typename ComponentT>
    inline ComponentT & getUnsafe(Entity e);

    template <typename ComponentT>
    inline ComponentT & getUnsafe(int32_t e_id);

    template <typename ComponentT>
    inline ComponentT & getUnsafe(Loc l);

    // FIXME: remove
    template <typename ArchetypeT, typename ComponentT>
    ComponentT & getComponent(Entity e);

    template <typename ComponentT>
    ComponentT & getDirect(int32_t column_idx, Loc loc);

    template <typename SingletonT>
    SingletonT & getSingleton();

    template <typename ArchetypeT>
    inline void clearArchetype();

    template <typename ArchetypeT>
    inline void clearTemporaries();

    template <typename... ComponentTs>
    inline Query<ComponentTs...> query();

    template <typename... ComponentTs, typename Fn>
    inline void forEach(const Query<ComponentTs...> &query, Fn &&fn);

    template <typename... ComponentTs>
    inline uint32_t numMatches(const Query<ComponentTs...> &query);

    // Jobs
    template <typename Fn, typename... DepTs>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        DepTs && ... dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
                         bool is_child = true,
                         DepTs && ... dependencies);

    // FIXME: currently this function requires that the query reference
    // is valid at least until the returned job is completed.
    template <typename... ComponentTs, typename Fn, typename... DepTs>
    inline JobID parallelFor(const Query<ComponentTs...> &query, Fn &&fn,
                             bool is_child = true,
                             DepTs && ... dependencies);

    template <typename Fn, typename... DepTs>
    inline JobID ioRead(const char *path, Fn &&fn, bool is_child = true,
                        DepTs && ... dependencies);

    inline void * tmpAlloc(uint64_t num_bytes);

    // FIXME: this doesn't belong here
    inline void resetTmpAlloc();

#ifdef MADRONA_USE_JOB_SYSTEM
    inline JobID currentJobID() const;
#endif


#ifdef MADRONA_MW_MODE
    inline WorldID worldID() const;
#endif

    inline WorldBase & data() { return *data_; }

protected:
    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitImpl(Fn &&fn, bool is_child, DepTs && ... dependencies);

    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitNImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                             DepTs && ... dependencies);

    template <typename ContextT, typename... ComponentTs, typename Fn,
              typename... DepTs>
    inline JobID parallelForImpl(const Query<ComponentTs...> &query, Fn &&fn,
                                 bool is_child, DepTs && ... dependencies);

    WorldBase *data_;

private:
    template <typename ContextT, typename Fn, typename... DepTs>
    inline JobID submitNImpl(Fn &&fn, uint32_t num_invocations, JobID parent_id,
                             DepTs && ... dependencies);

#ifdef MADRONA_USE_JOB_SYSTEM
    JobManager * const job_mgr_;
    StateManager * const state_mgr_;
    StateCache * const state_cache_;
    IOManager * const io_mgr_;
    const int worker_idx_;
    JobID cur_job_id_;
#endif
    StateManager * const state_mgr_;
    StateCache * const state_cache_;
#ifdef MADRONA_MW_MODE
    uint32_t cur_world_id_;
#endif

friend class JobManager;
};

}

#include "context.inl"
