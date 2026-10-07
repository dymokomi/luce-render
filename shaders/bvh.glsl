// Closest-hit traversal of luce-render's binary BVH (src/render/bvh.lucb), the
// software path: a short stack, nearer child first, Möller–Trumbore triangles.
// The including kernel declares the buffers `nodes`, `positions` and `indices`.

struct Hit {
    float t;        // distance along the (unnormalized) direction; t_max when missed
    uint triangle;  // slot in the reordered index list; 0xffffffff when missed
    vec2 uv;        // barycentrics of the second and third vertices
};

vec3 point_at(uint index) {
    return vec3(positions[index * 3u], positions[index * 3u + 1u], positions[index * 3u + 2u]);
}

// The entry distance of the ray into the box, or a value past t_max when it misses.
float box_entry(vec3 lo, vec3 hi, vec3 origin, vec3 inverse, float t_max) {
    vec3 a = (lo - origin) * inverse;
    vec3 b = (hi - origin) * inverse;
    vec3 near = min(a, b);
    vec3 far = max(a, b);
    float enter = max(max(near.x, near.y), max(near.z, 0.0));
    float leave = min(min(far.x, far.y), min(far.z, t_max));
    return enter <= leave ? enter : 3.0e38;
}

bool triangle_hit(uint triangle, vec3 origin, vec3 direction, inout Hit hit) {
    vec3 p0 = point_at(indices[triangle * 3u]);
    vec3 e1 = point_at(indices[triangle * 3u + 1u]) - p0;
    vec3 e2 = point_at(indices[triangle * 3u + 2u]) - p0;
    vec3 p = cross(direction, e2);
    float determinant = dot(e1, p);
    if (abs(determinant) < 1e-20) return false;
    float inverse = 1.0 / determinant;
    vec3 s = origin - p0;
    float u = dot(s, p) * inverse;
    if (u < 0.0 || u > 1.0) return false;
    vec3 q = cross(s, e1);
    float v = dot(direction, q) * inverse;
    if (v < 0.0 || u + v > 1.0) return false;
    float t = dot(e2, q) * inverse;
    if (t <= 0.0 || t >= hit.t) return false;
    hit.t = t;
    hit.triangle = triangle;
    hit.uv = vec2(u, v);
    return true;
}

#define BVH_STACK 64

Hit trace_closest(vec3 origin, vec3 direction, float t_max) {
    Hit hit = Hit(t_max, 0xffffffffu, vec2(0.0));
    // Axis-parallel rays: a tiny component instead of zero keeps the slab test
    // finite (Metal compiles with fast math, where infinities are not reliable).
    vec3 safe = mix(direction, sign(direction) * 1e-20 + vec3(equal(direction, vec3(0.0))) * 1e-20, lessThan(abs(direction), vec3(1e-20)));
    vec3 inverse = 1.0 / safe;
    uint stack[BVH_STACK];
    uint depth = 0u;
    uint node = 0u;
    if (box_entry(nodes[0].lo, nodes[0].hi, origin, inverse, hit.t) > hit.t) return hit;
    while (true) {
        Node current = nodes[node];
        if (current.count > 0u) {
            for (uint i = 0u; i < current.count; i++)
                triangle_hit(current.first + i, origin, direction, hit);
        } else {
            uint left = current.first;
            uint right = left + 1u;
            float a = box_entry(nodes[left].lo, nodes[left].hi, origin, inverse, hit.t);
            float b = box_entry(nodes[right].lo, nodes[right].hi, origin, inverse, hit.t);
            bool take_left = a <= hit.t;
            bool take_right = b <= hit.t;
            if (take_left && take_right) {
                // Visit the nearer first; the farther waits on the stack.
                uint near_child = a <= b ? left : right;
                uint far_child = a <= b ? right : left;
                if (depth < BVH_STACK) stack[depth++] = far_child;
                node = near_child;
                continue;
            }
            if (take_left) { node = left; continue; }
            if (take_right) { node = right; continue; }
        }
        if (depth == 0u) break;
        node = stack[--depth];
    }
    return hit;
}
