// The compiled scene's buffers (src/render/scene.lucb), bindings 0..2 of every
// tracing kernel: BVH nodes, world-space positions (x, y, z per point) and three
// point indices per triangle in leaf order.

struct Node {
    vec3 lo;
    uint first;   // interior: first child of the pair; leaf: first triangle
    vec3 hi;
    uint count;   // 0: interior; else triangles in the leaf
};

layout(set = 0, binding = 0, std430) readonly buffer Nodes { Node nodes[]; };
layout(set = 0, binding = 1, std430) readonly buffer Positions { float positions[]; };
layout(set = 0, binding = 2, std430) readonly buffer Indices { uint indices[]; };
