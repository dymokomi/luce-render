// The light tree (src/render/light_tree.lucb): choosing an emitter for a light
// sample by pbrt-v4's light-bounds importance, the probability of having chosen
// a given emitter (for MIS, from its trail), and samples of emissive triangles.
// Needs lights.glsl and openpbr.glsl before it.

layout(set = 0, binding = 15, std430) readonly buffer Emitters { uint emitters[]; };

#define K_TREE 8            // x tree nodes, y their first vec4 in the lights buffer, z chance of a distant light, w emitters
#define K_EMITTERS 10       // x emitter records' first u32, y the triangle map's first u32
#define LEAF_NODE 0x80000000u
#define EMITTER_TRIANGLE 0x80000000u

float safe_sqrt(float x) { return sqrt(max(0.0, x)); }

// How much node `node` may light point p with normal n (either side): power
// over distance², reduced by how far p lies outside the node's emission cone and
// how obliquely n faces it, each widened by the angle the bounds subtend.
float tree_importance(uint node, vec3 p, vec3 n) {
    uint base = uint(constants[K_TREE].y) + node * 3u;
    vec4 a = lights[base];
    vec4 b = lights[base + 1u];
    vec4 c = lights[base + 2u];
    if (a.w <= 0.0) return 0.0;
    vec3 center = 0.5 * (a.xyz + b.xyz);
    vec3 offset = p - center;
    float radius = 0.5 * length(b.xyz - a.xyz);
    float distance2 = max(dot(offset, offset), radius);
    vec3 wi = dot(offset, offset) > 0.0 ? normalize(offset) : c.xyz;
    float cos_w = dot(c.xyz, wi);
    float sin_w = safe_sqrt(1.0 - cos_w * cos_w);
    float cos_b = dot(offset, offset) < radius * radius ? -1.0 : safe_sqrt(1.0 - radius * radius / dot(offset, offset));
    float sin_b = safe_sqrt(1.0 - cos_b * cos_b);
    float cos_o = b.w;
    float sin_o = safe_sqrt(1.0 - cos_o * cos_o);
    // theta_x = max(0, theta_w - theta_o), then theta_p = max(0, theta_x - theta_b).
    float cos_x = cos_w > cos_o ? 1.0 : cos_w * cos_o + sin_w * sin_o;
    float sin_x = cos_w > cos_o ? 0.0 : sin_w * cos_o - cos_w * sin_o;
    float cos_p = cos_x > cos_b ? 1.0 : cos_x * cos_b + sin_x * sin_b;
    if (cos_p <= c.w) return 0.0;
    float importance = a.w * cos_p / distance2;
    // The receiver: how squarely n faces the node, by its bounds.
    float cos_i = abs(dot(wi, n));
    float sin_i = safe_sqrt(1.0 - cos_i * cos_i);
    float cos_pi = cos_i > cos_b ? 1.0 : cos_i * cos_b + sin_i * sin_b;
    return max(importance * cos_pi, 0.0);
}

// Descend from the root by importance; the emitter and the chance of it.
bool tree_sample(float u, vec3 p, vec3 n, out uint emitter, out float pdf) {
    pdf = 1.0;
    emitter = 0u;
    if (constants[K_TREE].x < 1.0) return false;
    uint node = 0u;
    for (uint depth = 0u; depth < 64u; depth++) {
        uint link = emitters[node * 2u];
        if ((emitters[node * 2u + 1u] & LEAF_NODE) != 0u) {
            emitter = link;
            return pdf > 0.0;
        }
        float left = tree_importance(link, p, n);
        float right = tree_importance(link + 1u, p, n);
        if (left + right <= 0.0) return false;
        float chance = left / (left + right);
        if (u < chance) {
            u = min(u / chance, 0.99999994);
            pdf *= chance;
            node = link;
        } else {
            u = min((u - chance) / (1.0 - chance), 0.99999994);
            pdf *= 1.0 - chance;
            node = link + 1u;
        }
    }
    return false;
}

// The chance tree_sample chooses `emitter` at p with normal n, from its trail.
float tree_pdf(uint emitter, vec3 p, vec3 n) {
    uint record = uint(constants[K_EMITTERS].x) + emitter * 4u;
    uint trail = emitters[record + 1u];
    uint depth = emitters[record + 2u];
    uint node = 0u;
    float pdf = 1.0;
    for (uint level = 0u; level < depth; level++) {
        uint link = emitters[node * 2u];
        float left = tree_importance(link, p, n);
        float right = tree_importance(link + 1u, p, n);
        if (left + right <= 0.0) return 0.0;
        uint bit = (trail >> level) & 1u;
        pdf *= bit == 0u ? left / (left + right) : right / (left + right);
        node = link + bit;
    }
    return pdf;
}

uint emitter_code(uint emitter) { return emitters[uint(constants[K_EMITTERS].x) + emitter * 4u]; }

// The emitter of a triangle (NONE when its material does not emit).
uint triangle_emitter(uint triangle) { return emitters[uint(constants[K_EMITTERS].y) + triangle]; }

vec3 tree_point(uint index) {
    return vec3(positions[index * 3u], positions[index * 3u + 1u], positions[index * 3u + 2u]);
}

// A point on emissive triangle `triangle` uniformly by area, seen from p: its
// direction, distance and solid-angle pdf, and the emitted radiance (two-sided).
bool sample_triangle(uint triangle, vec3 p, vec2 u, vec4 lambda, out LightSample s, out Spec radiance) {
    vec3 a = tree_point(indices[triangle * 3u]);
    vec3 b = tree_point(indices[triangle * 3u + 1u]);
    vec3 c = tree_point(indices[triangle * 3u + 2u]);
    float r = sqrt(u.x);
    vec3 q = a * (1.0 - r) + b * (r * (1.0 - u.y)) + c * (r * u.y);
    vec3 cross_ab = cross(b - a, c - a);
    float area = 0.5 * length(cross_ab);
    vec3 d = q - p;
    float d2 = dot(d, d);
    if (area <= 0.0 || d2 <= 0.0) return false;
    s.distance = sqrt(d2);
    s.direction = d / s.distance;
    float cos_light = abs(dot(s.direction, cross_ab)) / (2.0 * area);
    if (cos_light <= 0.0) return false;
    s.pdf = d2 / (cos_light * area);
    s.cos_light = cos_light;
    s.delta = false;
    radiance = material_color(shading[triangle].w, 6u, lambda);
    return true;
}

// The solid-angle pdf of sampling emissive triangle `triangle` hit at distance
// t along d (unit) from o.
float triangle_hit_pdf(uint triangle, vec3 d, float t) {
    vec3 a = tree_point(indices[triangle * 3u]);
    vec3 cross_ab = cross(tree_point(indices[triangle * 3u + 1u]) - a, tree_point(indices[triangle * 3u + 2u]) - a);
    float area = 0.5 * length(cross_ab);
    float cos_light = abs(dot(d, cross_ab)) / max(1e-20, 2.0 * area);
    return t * t / max(1e-20, cos_light * area);
}
