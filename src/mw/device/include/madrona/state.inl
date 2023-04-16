#pragma once

#include <madrona/utils.hpp>

#include <madrona/mw_gpu/host_print.hpp>

namespace madrona {

template <typename ComponentT>
void ECSRegistry::registerComponent()
{
    state_mgr_->registerComponent<ComponentT>();
}

template <typename ArchetypeT>
void ECSRegistry::registerArchetype()
{
    state_mgr_->registerArchetype<ArchetypeT>();
}

template <typename ArchetypeT>
void ECSRegistry::registerFixedSizeArchetype(CountT max_num_entities)
{
    // FIXME
    (void)max_num_entities;
    state_mgr_->registerArchetype<ArchetypeT>();
}

template <typename SingletonT>
void ECSRegistry::registerSingleton()
{
    state_mgr_->registerSingleton<SingletonT>();
}

template <typename ComponentT>
ComponentID StateManager::registerComponent()
{
    uint32_t id = TypeTracker::registerType<ComponentT>(
        &StateManager::num_components_);

    registerComponent(id, alignof(ComponentT), sizeof(ComponentT));

    return ComponentID {
        id,
    };
}

template <typename ArchetypeT, typename ComponentT>
void ECSRegistry::exportColumn(int32_t slot)
{
    state_mgr_->exportColumn<ArchetypeT, ComponentT>(
        slot, export_column_sizes_);
}

template <typename SingletonT>
void ECSRegistry::exportSingleton(int32_t slot)
{
    state_mgr_->exportSingleton<SingletonT>(
        slot, export_column_sizes_);
}

template <template <typename...> typename T, typename ...ComponentTs>
struct StateManager::RegistrationHelper<T<ComponentTs...>> {
    using ArchetypeT = T<ComponentTs...>;
    static_assert(std::is_same_v<ArchetypeT, Archetype<ComponentTs...>>);

    template <typename ComponentT>
    static void registerColumnIndex(uint32_t *idx)
    {
        using LookupT = typename ArchetypeRef<ArchetypeT>::
            template ComponentLookup<ComponentT>;

        TypeTracker::registerType<LookupT>(idx);
    }

    static std::array<ComponentID, sizeof...(ComponentTs)>
        registerArchetypeComponents()
    {
        uint32_t column_idx = user_component_offset_;

        ( registerColumnIndex<ComponentTs>(&column_idx), ... );

        std::array archetype_components {
            ComponentID { TypeTracker::typeID<ComponentTs>() }
            ...
        };

        return archetype_components;
    }

};

template <typename ArchetypeT>
ArchetypeID StateManager::registerArchetype()
{
    uint32_t archetype_id = TypeTracker::registerType<ArchetypeT>(
        &StateManager::num_archetypes_);

    using Base = typename ArchetypeT::Base;

    auto archetype_components =
        RegistrationHelper<Base>::registerArchetypeComponents();

    registerArchetype(archetype_id, archetype_components.data(),
                      archetype_components.size());

    return ArchetypeID {
        archetype_id,
    };
}

template <typename SingletonT>
void StateManager::registerSingleton()
{
    using ArchetypeT = SingletonArchetype<SingletonT>;

    registerComponent<SingletonT>();
    registerArchetype<ArchetypeT>();

    makeEntityNow<ArchetypeT>(WorldID {});
}

template <typename SingletonT>
SingletonT & StateManager::getSingleton(WorldID)
{
    SingletonT *col = getSingletonColumn<SingletonT>();
    return col[0];
}

template <typename... ComponentTs>
Query<ComponentTs...> StateManager::query()
{
    std::array component_ids {
        TypeTracker::typeID<ComponentTs>()
        ...
    };

    QueryRef *ref = &query_refs_[num_queries_++];

    if (ref->numReferences.load_acquire() == 0) {
        makeQuery(component_ids.data(), component_ids.size(), ref);
    }

    return Query<ComponentTs...>(true, ref);
}

template <typename Fn, int32_t... Indices>
void StateManager::iterateArchetypesRawImpl(QueryRef *query_ref, Fn &&fn,
        std::integer_sequence<int32_t, Indices...>)
{

    uint32_t *query_values = &query_data_[query_ref->offset];
    int32_t num_archetypes = query_ref->numMatchingArchetypes;

    for (int i = 0; i < num_archetypes; i++) {
        uint32_t archetype_idx = *query_values;
        query_values += 1;

        Table &tbl = archetypes_[archetype_idx]->tbl;

        bool early_out = fn(tbl.numRows,
            tbl.columns[query_values[Indices]] ...);

        if (early_out) {
            return;
        }

        query_values += sizeof...(Indices);
    }
}

template <int32_t num_components, typename Fn>
void StateManager::iterateArchetypesRaw(QueryRef *query_ref, Fn &&fn)
{
    using IndicesWrapper =
        std::make_integer_sequence<int32_t, num_components>;

    iterateArchetypesRawImpl(query_ref, std::forward<Fn>(fn),
                             IndicesWrapper());
}

uint32_t StateManager::numMatchingEntities(QueryRef *query_ref)
{
    uint32_t *query_values = &query_data_[query_ref->offset];
    int32_t num_archetypes = query_ref->numMatchingArchetypes;
    int32_t num_components = query_ref->numComponents;

    uint32_t total_rows = 0;
    for (int i = 0; i < num_archetypes; i++) {
        uint32_t archetype_idx = *query_values;

        Table &tbl = archetypes_[archetype_idx]->tbl;

        total_rows += tbl.numRows;

        query_values += 1 + num_components;
    }

    return total_rows;
}

template <typename ArchetypeT>
Entity StateManager::makeEntityNow(WorldID world_id)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

    return makeEntityNow(world_id, archetype_id);
}

template <typename ArchetypeT>
Loc StateManager::makeTemporary(WorldID world_id)
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

    return makeTemporary(world_id, archetype_id);
}

template <typename ArchetypeT>
void StateManager::clearTemporaries()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    clearTemporaries(archetype_id);
}

inline Loc StateManager::getLoc(Entity e) const
{
    const EntityStore::EntitySlot &slot = entity_store_.entities[e.id];
    return slot.loc;
}

template <typename ComponentT>
ComponentT & StateManager::getUnsafe(Entity e)
{
    const EntityStore::EntitySlot &slot = entity_store_.entities[e.id];

    assert(slot.gen == e.gen);

    return getUnsafe<ComponentT>(slot.loc);
}

template <typename ComponentT>
ComponentT & StateManager::getUnsafe(Loc loc)

{
    auto &archetype = *archetypes_[loc.archetype];
    uint32_t component_id = TypeTracker::typeID<ComponentT>();
    auto col_idx = archetype.columnLookup.lookup(component_id);
    assert(col_idx.has_value());

    Table &tbl = archetype.tbl;

    return ((ComponentT *)(tbl.columns[*col_idx]))[loc.row];
}

template <typename ComponentT>
ResultRef<ComponentT> StateManager::get(Entity e)
{
    const EntityStore::EntitySlot &slot = entity_store_.entities[e.id];
    if (slot.gen != e.gen) {
        return ResultRef<ComponentT>(nullptr);
    }

    return get<ComponentT>(slot.loc);
}

template <typename ComponentT>
ResultRef<ComponentT> StateManager::get(Loc loc)
{
    auto &archetype = *archetypes_[loc.archetype];
    uint32_t component_id = TypeTracker::typeID<ComponentT>();
    auto col_idx = archetype.columnLookup.lookup(component_id);

    if (!col_idx.has_value()) {
        return ResultRef<ComponentT>(nullptr);
    }

    assert(col_idx.has_value());

    Table &tbl = archetype.tbl;

    return ResultRef<ComponentT>(
        ((ComponentT *)(tbl.columns[*col_idx])) + loc.row);
}

template <typename ComponentT>
ComponentT & StateManager::getDirect(int32_t column_idx, Loc loc)
{
    return ((ComponentT *)getArchetypeColumn(
        loc.archetype, column_idx))[loc.row];
}

template <typename ArchetypeT, typename ComponentT>
ComponentT * StateManager::getArchetypeComponent()
{
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();
    uint32_t component_id = TypeTracker::typeID<ComponentT>();

    return (ComponentT *)getArchetypeComponent(archetype_id, component_id);
}

void * StateManager::getArchetypeComponent(uint32_t archetype_id,
                                           uint32_t component_id)
{
    auto &archetype = *archetypes_[archetype_id];
    int32_t col_idx = getArchetypeColumnIndex(archetype_id, component_id);
    return archetype.tbl.columns[col_idx];
}

int32_t StateManager::getArchetypeColumnIndex(uint32_t archetype_id,
                                              uint32_t component_id)
{
    auto &archetype = *archetypes_[archetype_id];
    return *archetype.columnLookup.lookup(component_id);
}

void * StateManager::getArchetypeColumn(uint32_t archetype_id,
                                        int32_t column_idx)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.tbl.columns[column_idx];
}

uint32_t StateManager::getArchetypeColumnBytesPerRow(uint32_t archetype_id,
                                                     int32_t column_idx)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.tbl.columnSizes[column_idx];
}

int32_t StateManager::getArchetypeNumColumns(uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];
    return archetype.tbl.numColumns;
}

void StateManager::remapEntity(Entity e, int32_t row_idx)
{
    entity_store_.entities[e.id].loc.row = row_idx;
}

template <typename SingletonT>
SingletonT * StateManager::getSingletonColumn()
{
    using ArchetypeT = SingletonArchetype<SingletonT>;
    uint32_t archetype_id = TypeTracker::typeID<ArchetypeT>();

    // Abuse the fact that the singleton only has one component that is going
    // to be in column 1
    
    Table &tbl = archetypes_[archetype_id]->tbl;
    return (SingletonT *)tbl.columns[1];
}

bool StateManager::archetypeNeedsSort(uint32_t archetype_id) const
{
    return archetypes_[archetype_id]->needsSort;
}

void StateManager::archetypeClearNeedsSort(uint32_t archetype_id)
{
    archetypes_[archetype_id]->needsSort = false;
}

}
