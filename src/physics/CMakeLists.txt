set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona")

set(MADRONA_PHYSICS_SRCS
    ${INC_DIR}/physics.hpp ${INC_DIR}/physics.inl physics.cpp geometry.cpp narrowphase.cpp broadphase.cpp
)

add_library(madrona_physics STATIC
    ${MADRONA_PHYSICS_SRCS}
)

target_link_libraries(madrona_physics
    PUBLIC
        madrona_core
)

add_library(madrona_mw_physics STATIC
    ${MADRONA_PHYSICS_SRCS}
)

target_link_libraries(madrona_mw_physics
    PUBLIC
        madrona_mw_core
)

add_library(madrona_physics_assets STATIC
    ${INC_DIR}/physics_assets.hpp physics_assets.cpp geometry.cpp
)

target_link_libraries(madrona_physics_assets PRIVATE
    madrona_common madrona_importer
)

if (TARGET madrona_cuda)
    target_link_libraries(madrona_physics_assets PRIVATE
        madrona_cuda)
endif ()
