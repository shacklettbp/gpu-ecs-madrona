#pragma once

#include <madrona/physics.hpp>

namespace madrona {
namespace phys {

class PhysicsLoader {
public:
    enum class StorageType {
        CPU,
        CUDA,
    };

    PhysicsLoader(StorageType storage_type, CountT max_objects);
    ~PhysicsLoader();
    PhysicsLoader(PhysicsLoader &&o);

    struct LoadedHull {
        math::AABB aabb;
        geometry::HalfEdgeMesh collisionMesh;
    };

    LoadedHull loadHullFromDisk(const char *obj_path);

    CountT loadObjects(const RigidBodyMetadata *metadatas,
                        const math::AABB *aabbs,
                        const CollisionPrimitive *primitives,
                        CountT num_objs);


    ObjectManager & getObjectManager();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
