// Tracing for the integrator: hardware ray queries against the scene's TLAS
// (binding 14) when built with RAY_QUERY=1, else the software BVH. Both give
// the same Hit: t, the triangle slot (the BLAS is built from the reordered
// indices, so its primitive index is the slot) and the barycentrics of
// vertices 1 and 2. The triangles' instance has mask 1 and the splats' (a
// Blas of boxes, splats.glsl) mask 2, so triangle queries never see splats.

#if RAY_QUERY
#extension GL_EXT_ray_query : require

struct Hit {
    float t;
    uint triangle;
    vec2 uv;
};

layout(set = 0, binding = 14) uniform accelerationStructureEXT scene_tlas;

// One query object for every trace of a kernel: they never overlap, and on
// Metal each declared query is thread state held for the whole kernel.
rayQueryEXT query;

Hit trace_closest(vec3 origin, vec3 direction, float t_max) {
    rayQueryInitializeEXT(query, scene_tlas, gl_RayFlagsOpaqueEXT, 0x01u, origin, 0.0, direction, t_max);
    while (rayQueryProceedEXT(query)) {
    }
    if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionTriangleEXT)
        return Hit(t_max, 0xffffffffu, vec2(0.0));
    return Hit(rayQueryGetIntersectionTEXT(query, true), uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)),
               rayQueryGetIntersectionBarycentricsEXT(query, true));
}

bool trace_any(vec3 origin, vec3 direction, float t_max) {
    rayQueryInitializeEXT(query, scene_tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0x01u, origin, 0.0, direction, t_max);
    while (rayQueryProceedEXT(query)) {
    }
    return rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT;
}
#else
#include "bvh.glsl"
#endif
