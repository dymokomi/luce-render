// UsdLux lights: sampling for next-event estimation, hits for BSDF rays, and the
// pdf MIS needs at those hits. Records are LIGHT_STRIDE vec4s
// (src/render/lights.lucb):
//   0  position, type (0 rect, 1 disk, 2 sphere, 3 distant)
//   1  u half-extent vector (rect, disk), radius (sphere)
//   2  v half-extent vector, cos of the half angle (distant)
//   3  emission direction (local -Z), area
//   4  emission spectrum fit (c0, c1, c2, scale)
//   5  emission ACEScg, flags (1 visible to camera, 2 blackbody)
//   6  kelvin, share among distant lights, cumulative share, emitter (lights
//      that have a place are emitters of the light tree, lighttree.glsl)
// Radiance already holds intensity, exposure and normalization; a distant light's
// is its irradiance over its solid angle (all of it when the angle is 0: a delta).

#define LIGHT_STRIDE 7u
#define LIGHT_RECT 0u
#define LIGHT_DISK 1u
#define LIGHT_SPHERE 2u
#define LIGHT_DISTANT 3u

vec4 light_at(uint light, uint row) { return lights[light * LIGHT_STRIDE + row]; }
uint light_type(uint light) { return uint(light_at(light, 0u).w); }
bool light_visible(uint light) { return (uint(light_at(light, 5u).w) & 1u) != 0u; }
// The chance a light sample picks distant light `light` (others: the tree's).
float light_select_pdf(uint light) { return constants[8].z * light_at(light, 6u).y; }
uint light_emitter(uint light) { return uint(light_at(light, 6u).w); }

Spec light_radiance(uint light, vec4 lambda) {
    vec4 rgb = light_at(light, 5u);
    Spec radiance = spec_of(light_at(light, 4u), rgb.xyz, lambda);
#if SPECTRAL
    if ((uint(rgb.w) & 2u) != 0u) {
        float kelvin = light_at(light, 6u).x;
        for (int i = 0; i < 4; i++) radiance[i] *= blackbody(lambda[i], kelvin);
    }
#endif
    return radiance;
}

// A distant light chosen in proportion to its power; `pdf` its share among
// the distant lights.
uint pick_distant(float u, out float pdf) {
    uint count = uint(constants[K_INFO].x);
    uint chosen = NONE;
    for (uint light = 0u; light < count; light++) {
        if (light_type(light) != 3u) continue;
        chosen = light;
        if (u < light_at(light, 6u).z) break;
    }
    pdf = chosen == NONE ? 0.0 : light_at(chosen, 6u).y;
    return chosen;
}

struct LightSample {
    vec3 direction;   // unit, from the shaded point toward the light
    float distance;   // INFINITY for distant lights
    float pdf;        // solid angle (1 for a delta)
    bool delta;
};

// A point on the unit disk, Shirley and Chiu's concentric map.
vec2 concentric(vec2 u) {
    vec2 o = 2.0 * u - 1.0;
    if (o.x == 0.0 && o.y == 0.0) return vec2(0.0);
    float r, theta;
    if (abs(o.x) > abs(o.y)) { r = o.x; theta = (PI / 4.0) * (o.y / o.x); }
    else { r = o.y; theta = (PI / 2.0) - (PI / 4.0) * (o.x / o.y); }
    return r * vec2(cos(theta), sin(theta));
}

// A direction in the cone of `cos_max` around unit `axis`, uniformly.
vec3 sample_cone(vec3 axis, float cos_max, vec2 u) {
    float c = 1.0 - u.x * (1.0 - cos_max);
    float s = sqrt(max(0.0, 1.0 - c * c));
    float phi = 2.0 * PI * u.y;
    vec3 t, b;
    basis(axis, t, b);
    return normalize(t * (s * cos(phi)) + b * (s * sin(phi)) + axis * c);
}

// The nearer positive hit of a ray (unit direction) with a sphere, or INFINITY.
float sphere_hit(vec3 center, float radius, vec3 o, vec3 d) {
    vec3 oc = o - center;
    float b = dot(oc, d);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return INFINITY;
    h = sqrt(h);
    float t = -b - h;
    if (t > 0.0) return t;
    t = -b + h;
    return t > 0.0 ? t : INFINITY;
}

bool sample_light(uint light, vec3 p, vec2 u, out LightSample s) {
    vec4 row0 = light_at(light, 0u);
    vec4 row1 = light_at(light, 1u);
    vec4 row2 = light_at(light, 2u);
    vec4 row3 = light_at(light, 3u);
    uint type = uint(row0.w);
    s.delta = false;
    if (type == LIGHT_RECT || type == LIGHT_DISK) {
        vec2 a = type == LIGHT_RECT ? 2.0 * u - 1.0 : concentric(u);
        vec3 q = row0.xyz + a.x * row1.xyz + a.y * row2.xyz;
        vec3 d = q - p;
        float d2 = dot(d, d);
        s.distance = sqrt(d2);
        s.direction = d / s.distance;
        float cos_light = -dot(s.direction, row3.xyz);
        if (cos_light <= 0.0 || row3.w <= 0.0) return false;
        s.pdf = d2 / (cos_light * row3.w);
        return true;
    }
    if (type == LIGHT_SPHERE) {
        vec3 to_center = row0.xyz - p;
        float d2 = dot(to_center, to_center);
        float r = row1.w;
        if (d2 <= r * r) return false;
        float cos_max = sqrt(max(0.0, 1.0 - r * r / d2));
        s.direction = sample_cone(to_center / sqrt(d2), cos_max, u);
        s.distance = sphere_hit(row0.xyz, r, p, s.direction);
        if (s.distance >= INFINITY) return false;
        s.pdf = 1.0 / (2.0 * PI * (1.0 - cos_max));
        return true;
    }
    // Distant: toward the light is against its emission direction.
    vec3 axis = -row3.xyz;
    float cos_half = row2.w;
    s.distance = INFINITY;
    if (cos_half >= 1.0 - 1e-7) {
        s.direction = axis;
        s.pdf = 1.0;
        s.delta = true;
        return true;
    }
    s.direction = sample_cone(axis, cos_half, u);
    s.pdf = 1.0 / (2.0 * PI * (1.0 - cos_half));
    return true;
}

// Where a ray (unit direction) meets the light's emitting side first, or INFINITY.
// Lights are seen only from the side they shine to; distant lights are never hit.
float intersect_light(uint light, vec3 o, vec3 d, float t_max) {
    vec4 row0 = light_at(light, 0u);
    uint type = uint(row0.w);
    if (type == LIGHT_SPHERE) {
        float t = sphere_hit(row0.xyz, light_at(light, 1u).w, o, d);
        return t < t_max ? t : INFINITY;
    }
    if (type == LIGHT_DISTANT) return INFINITY;
    vec3 u = light_at(light, 1u).xyz;
    vec3 v = light_at(light, 2u).xyz;
    vec3 n = light_at(light, 3u).xyz;
    float facing = dot(d, n);
    if (facing >= 0.0) return INFINITY;
    float t = dot(row0.xyz - o, n) / facing;
    if (t <= 0.0 || t >= t_max) return INFINITY;
    vec3 q = o + t * d - row0.xyz;
    float a = dot(q, u) / dot(u, u);
    float b = dot(q, v) / dot(v, v);
    bool inside = type == LIGHT_RECT ? (abs(a) <= 1.0 && abs(b) <= 1.0) : (a * a + b * b <= 1.0);
    return inside ? t : INFINITY;
}

// The solid-angle pdf of sampling the light hit at distance `t` along `d` from
// `o` (the selection's chance not included).
float light_hit_pdf(uint light, vec3 o, vec3 d, float t) {
    vec4 row0 = light_at(light, 0u);
    uint type = uint(row0.w);
    float pdf;
    if (type == LIGHT_SPHERE) {
        vec3 to_center = row0.xyz - o;
        float r = light_at(light, 1u).w;
        float cos_max = sqrt(max(0.0, 1.0 - r * r / dot(to_center, to_center)));
        pdf = 1.0 / (2.0 * PI * max(1e-12, 1.0 - cos_max));
    } else if (type == LIGHT_DISTANT) {
        pdf = 1.0 / (2.0 * PI * max(1e-12, 1.0 - light_at(light, 2u).w));
    } else {
        vec4 row3 = light_at(light, 3u);
        float cos_light = max(1e-12, -dot(d, row3.xyz));
        pdf = t * t / (cos_light * row3.w);
    }
    return pdf;
}
