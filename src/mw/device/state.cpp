/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/state.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>

namespace madrona {

static MADRONA_NO_INLINE void growTable(Table &tbl, int32_t row)
{
    using namespace mwGPU;

    int32_t cur_num_rows = tbl.numAllocatedRows;
    int32_t new_num_rows = cur_num_rows * 2;

    if (new_num_rows - cur_num_rows > 10000) {
        new_num_rows = cur_num_rows + 10000; 
    }

    for (int32_t i = 0; i < tbl.numColumns; i++) {
        void *cur_col = tbl.columns[i];
        uint64_t column_bytes_per_row = tbl.columnSizes[i];

        uint64_t cur_num_bytes =
            cur_num_rows * column_bytes_per_row;
        
        uint64_t new_num_bytes =
            new_num_rows * column_bytes_per_row;

        void *new_col = rawAlloc(new_num_bytes);

        memcpy(new_col, cur_col, cur_num_bytes);

        rawDealloc(cur_col);

        tbl.columns[i] = new_col;
    }

    tbl.numAllocatedRows = new_num_rows;
}

ECSRegistry::ECSRegistry(StateManager &state_mgr, void **export_ptr)
    : state_mgr_(&state_mgr),
      export_ptr_(export_ptr)
{}

StateManager::StateManager(uint32_t)
{
    using namespace mwGPU;

#pragma unroll(1)
    for (int32_t i = 0; i < (int32_t)max_components_; i++) {
        Optional<TypeInfo>::noneAt(&components_[i]);
    }

    // Without disabling unrolling, this loop 100x's compilation time for
    // this file.... Optional must be really slow.
#pragma unroll(1)
    for (int32_t i = 0; i < (int32_t)max_archetypes_; i++) {
        Optional<ArchetypeStore>::noneAt(&archetypes_[i]);
    }

#pragma unroll(1)
    for (int32_t i = 0; i < EntityStore::maxNumEntities; i++) {
        entity_store_.entities[i].gen = 0;
    }

    entity_store_.numFreeEntities = EntityStore::maxNumEntities;

#pragma unroll(1)
    for (int32_t i = 0; i < EntityStore::maxNumEntities / 64; i++) {
        entity_store_.freeEntities[i] = 0xFFFF'FFFF;
    }
    entity_store_.numFreeEntities = EntityStore::maxNumEntities;

    registerComponent<Entity>();

    num_queries_ = 0;
    tmp_alloc_head_ = nullptr;
}

void StateManager::registerComponent(uint32_t id, uint32_t alignment,
                                     uint32_t num_bytes)
{
    components_[id].emplace(TypeInfo {
        /* .alignment = */ alignment,
        /* .numBytes = */  num_bytes,
    });
}

StateManager::ArchetypeStore::ArchetypeStore(uint32_t offset,
                                             uint32_t num_user_components,
                                             uint32_t num_columns,
                                             TypeInfo *type_infos,
                                             IntegerMapPair *lookup_input)
    : componentOffset(offset),
      numUserComponents(num_user_components),
      tbl(),
      columnLookup(lookup_input, num_columns),
      needsSort(false)
{
    using namespace mwGPU;

    tbl.numColumns = num_columns;

    for (int i = 0 ; i < (int)num_columns; i++) {
        uint64_t col_bytes = (uint64_t)type_infos[i].numBytes;

        tbl.columns[i] = rawAlloc(col_bytes);
        tbl.columnSizes[i] = col_bytes;
    }

    tbl.numRows = 0;
    tbl.numAllocatedRows = 1;
}

void StateManager::registerArchetype(uint32_t id, ComponentID *components,
                                     uint32_t num_user_components)
{
    uint32_t offset = archetype_component_offset_;
    archetype_component_offset_ += num_user_components;

    uint32_t num_total_components = num_user_components + 1;

    std::array<TypeInfo, max_archetype_components_> type_infos;
    std::array<IntegerMapPair, max_archetype_components_> lookup_input;

    TypeInfo *type_ptr = type_infos.data();
    IntegerMapPair *lookup_input_ptr = lookup_input.data();

    // Add entity column as first column of every table
    *type_ptr = *components_[0];
    type_ptr++;

    *lookup_input_ptr = {
        TypeTracker::typeID<Entity>(),
        0,
    };
    lookup_input_ptr++;

    for (int i = 0; i < (int)num_user_components; i++) {
        ComponentID component_id = components[i];
        assert(component_id.id != TypeTracker::unassignedTypeID);
        archetype_components_[offset + i] = component_id.id;

        type_ptr[i] = *components_[component_id.id];

        lookup_input_ptr[i] = IntegerMapPair {
            /* .key = */   component_id.id,
            /* .value = */ (uint32_t)i + user_component_offset_,
        };
    }

    archetypes_[id].emplace(offset, num_user_components,
                            num_total_components,
                            type_infos.data(),
                            lookup_input.data());
}

void StateManager::makeQuery(const uint32_t *components,
                             uint32_t num_components,
                             QueryRef *query_ref)
{
    uint32_t query_offset = query_data_offset_;

    uint32_t num_matching_archetypes = 0;
    for (int32_t archetype_idx = 0; archetype_idx < (int32_t)num_archetypes_;
         archetype_idx++) {
        auto &archetype = *archetypes_[archetype_idx];

        bool has_components = true;
        for (int component_idx = 0; component_idx < (int)num_components; 
             component_idx++) {
            uint32_t component = components[component_idx];
            if (component == TypeTracker::typeID<Entity>()) {
                continue;
            }

            if (!archetype.columnLookup.exists(component)) {
                has_components = false;
                break;
            }
        }

        if (!has_components) {
            continue;
        }


        num_matching_archetypes += 1;
        query_data_[query_data_offset_++] = uint32_t(archetype_idx);

        for (int32_t component_idx = 0;
             component_idx < (int32_t)num_components; component_idx++) {
            uint32_t component = components[component_idx];
            assert(component != TypeTracker::unassignedTypeID);
            if (component == TypeTracker::typeID<Entity>()) {
                query_data_[query_data_offset_++] = 0;
            } else {
                query_data_[query_data_offset_++] = 
                    archetype.columnLookup[component];
            }
        }
    }

    assert(query_data_offset_ < 512);

    query_ref->offset = query_offset;
    query_ref->numMatchingArchetypes = num_matching_archetypes;
    query_ref->numComponents = num_components;
}

static inline int32_t getEntitySlot(EntityStore &entity_store)
{
    assert(entity_store.numFreeEntities >= 1);

    int32_t free_idx;
    for (int32_t i = 0; i < EntityStore::maxNumEntities / 32; i++) {
        uint32_t free_mask = entity_store.freeEntities[i];
        if (free_mask == 0) continue;

        int32_t free_offset = __ffs(free_mask);
        entity_store.freeEntities[i] = free_mask & ~(1 << (free_offset - 1));

        free_idx = i * 32 + free_offset;

        break;
    }

    entity_store.numFreeEntities -= 1;

    return free_idx;
}

Entity StateManager::makeEntityNow(WorldID world_id, uint32_t archetype_id)
{
    auto &archetype = *archetypes_[archetype_id];
    archetype.needsSort = true;
    Table &tbl = archetype.tbl;

    int32_t row = tbl.numRows++;

    if (row >= tbl.numAllocatedRows) {
        growTable(tbl, row);
    }

    Loc loc {
        archetype_id,
        row,
    };

    int32_t entity_slot_idx = getEntitySlot(entity_store_);

    EntityStore::EntitySlot &entity_slot =
        entity_store_.entities[entity_slot_idx];

    entity_slot.loc = loc;
    entity_slot.gen += 1;

    // FIXME: proper entity mapping on GPU
    Entity e {
        entity_slot.gen,
        entity_slot_idx,
    };

    Entity *entity_column = (Entity *)tbl.columns[0];

    entity_column[row] = e;

    return e;
}

Loc StateManager::makeTemporary(WorldID world_id,
                                uint32_t archetype_id)
{
    Table &tbl = archetypes_[archetype_id]->tbl;

    int32_t row = tbl.numRows++;

    if (row >= tbl.numAllocatedRows) {
        growTable(tbl, row);
    }

    Loc loc {
        archetype_id,
        row,
    };

    return loc;
}

void StateManager::destroyEntityNow(Entity e)
{
    EntityStore::EntitySlot &entity_slot =
        entity_store_.entities[e.id];
    entity_slot.gen++;

    Loc loc = entity_slot.loc;

    int32_t mask_idx = e.id / 32;
    int32_t mask_offset = e.id % 32;
    entity_store_.freeEntities[mask_idx] |= 1 << mask_offset;
    entity_store_.numFreeEntities += 1;

    Table &tbl = archetypes_[loc.archetype]->tbl;
    int32_t last_row = --tbl.numRows;
    if (loc.row == last_row) {
        return;
    }

    for (int32_t i = 0; i < tbl.numColumns; i++) {
        void *cur_col = tbl.columns[i];
        uint64_t column_bytes_per_row = tbl.columnSizes[i];

        memcpy((char *)cur_col + column_bytes_per_row * loc.row,
               (char *)cur_col + column_bytes_per_row * last_row,
               column_bytes_per_row);
    }
}

void StateManager::clearTemporaries(uint32_t archetype_id)
{
    Table &tbl = archetypes_[archetype_id]->tbl;
    tbl.numRows = 0;
}

void StateManager::resizeArchetype(uint32_t archetype_id, int32_t num_rows)
{
    archetypes_[archetype_id]->tbl.numRows = num_rows;
}

int32_t StateManager::numArchetypeRows(uint32_t archetype_id) const
{
    return archetypes_[archetype_id]->tbl.numRows;
}

}
