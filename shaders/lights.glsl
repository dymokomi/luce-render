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
#define LIGHT_DOME 4u
#define K_DOME 16           // x the light whose image this is (-1: none), y its texels' first vec4, z width, w height
#define K_DOME_CDF 17       // x its cumulative weights' first float, y the weights' sum

vec4 light_at(uint light, uint row) { return lights[light * LIGHT_STRIDE + row]; }
uint light_type(uint light) { return uint(light_at(light, 0u).w); }
bool light_visible(uint light) { return (uint(light_at(light, 5u).w) & 1u) != 0u; }
// The chance a light sample picks distant or dome light `light` (others: the tree's).
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

// A distant or dome light chosen in proportion to its power; `pdf` its share
// among them.
uint pick_distant(float u, out float pdf) {
    uint count = light_count();
    uint chosen = NONE;
    for (uint light = 0u; light < count; light++) {
        if (light_type(light) < LIGHT_DISTANT) continue;
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
    float cos_light;  // at the light, toward the shaded point (1 for distant lights)
    bool delta;
};

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

// -- Dome lights (src/render/environment.lucb): a lat-long image around the
// scene, its center down the light's -Z, its top +Y; one with no image (or not
// the one image K_DOME holds) shines its color evenly.

bool dome_textured(uint light) { return float(light) == constants[K_DOME].x; }

vec3 dome_local(uint light, vec3 d) {
    return vec3(dot(d, light_at(light, 1u).xyz), dot(d, light_at(light, 2u).xyz), dot(d, light_at(light, 3u).xyz));
}

// The texel a world direction falls in, and sin θ there.
uint dome_texel(uint light, vec3 d, out float sin_theta) {
    vec3 l = dome_local(light, d);
    uint width = uint(constants[K_DOME].z);
    uint height = uint(constants[K_DOME].w);
    float u = 0.5 + atan(l.x, -l.z) * (0.5 / PI);
    float v = acos(clamp(l.y, -1.0, 1.0)) / PI;
    sin_theta = sqrt(max(0.0, 1.0 - l.y * l.y));
    uint x = min(uint(u * float(width)), width - 1u);
    uint y = min(uint(v * float(height)), height - 1u);
    return y * width + x;
}

Spec dome_radiance(uint light, vec3 d, vec4 lambda) {
    Spec radiance = light_radiance(light, lambda);
    if (!dome_textured(light)) return radiance;
    float sin_theta;
    uint texel = dome_texel(light, d, sin_theta);
    return radiance * spec_of_texel(lights[uint(constants[K_DOME].y) + texel].xyz, lambda);
}

// The solid-angle pdf of sampling direction d (the selection's chance not
// included): a texel's weight over the sum, spread over its solid angle.
float dome_pdf(uint light, vec3 d) {
    if (!dome_textured(light)) return 1.0 / (4.0 * PI);
    float sin_theta;
    uint texel = dome_texel(light, d, sin_theta);
    if (sin_theta <= 0.0) return 0.0;
    float weight = lights[uint(constants[K_DOME].y) + texel].w;
    return weight / constants[K_DOME_CDF].y * constants[K_DOME].z * constants[K_DOME].w / (2.0 * PI * PI * sin_theta);
}

float dome_cdf(uint at) {
    uint i = uint(constants[K_DOME_CDF].x) + at;
    return lights[i >> 2u][i & 3u];
}

// The first of `count` cumulative values from `first` that passes u, and u's
// place within it (0..1).
uint dome_search(uint first, uint count, float u, out float within) {
    uint low = 0u;
    uint high = count - 1u;
    while (low < high) {
        uint middle = (low + high) / 2u;
        if (dome_cdf(first + middle) > u) high = middle;
        else low = middle + 1u;
    }
    float below = low > 0u ? dome_cdf(first + low - 1u) : 0.0;
    float above = dome_cdf(first + low);
    within = above > below ? clamp((u - below) / (above - below), 0.0, 0.99999994) : 0.5;
    return low;
}

bool sample_dome(uint light, vec2 u, out LightSample s) {
    s.distance = INFINITY;
    s.delta = false;
    s.cos_light = 1.0;
    vec3 l;
    if (!dome_textured(light)) {
        float z = 1.0 - 2.0 * u.x;
        float r = sqrt(max(0.0, 1.0 - z * z));
        l = vec3(r * cos(2.0 * PI * u.y), z, r * sin(2.0 * PI * u.y));
    } else {
        uint width = uint(constants[K_DOME].z);
        uint height = uint(constants[K_DOME].w);
        float fy, fx;
        uint y = dome_search(width * height, height, u.y, fy);
        uint x = dome_search(y * width, width, u.x, fx);
        float theta = PI * (float(y) + fy) / float(height);
        float phi = 2.0 * PI * ((float(x) + fx) / float(width) - 0.5);
        l = vec3(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi));
    }
    s.direction = normalize(light_at(light, 1u).xyz * l.x + light_at(light, 2u).xyz * l.y + light_at(light, 3u).xyz * l.z);
    s.pdf = dome_pdf(light, s.direction);
    return s.pdf > 0.0;
}

bool sample_light(uint light, vec3 p, vec2 u, out LightSample s) {
    vec4 row0 = light_at(light, 0u);
    vec4 row1 = light_at(light, 1u);
    vec4 row2 = light_at(light, 2u);
    vec4 row3 = light_at(light, 3u);
    uint type = uint(row0.w);
    s.delta = false;
    s.cos_light = 1.0;
    if (type == LIGHT_DOME) return sample_dome(light, u, s);
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
        s.cos_light = cos_light;
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
        s.cos_light = max(0.0, -dot(s.direction, normalize(p + s.direction * s.distance - row0.xyz)));
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

// The cosine at light `light` (not distant) toward the point a ray from o
// along d (unit) left, where it hits the light at distance t.
float light_cos(uint light, vec3 o, vec3 d, float t) {
    vec4 row0 = light_at(light, 0u);
    if (uint(row0.w) == LIGHT_SPHERE) return max(0.0, -dot(d, normalize(o + d * t - row0.xyz)));
    return max(0.0, -dot(d, light_at(light, 3u).xyz));
}

// Where a ray (unit direction) meets the light's emitting side first, or INFINITY.
// Lights are seen only from the side they shine to; distant lights are never hit.
// `light_flow`: the ray follows light (a light path), so it is stopped where
// a camera ray along it reversed would be: crossing a flat light the way it
// shines. Visibility then reads the same both ways.
float intersect_light(uint light, vec3 o, vec3 d, float t_max, bool light_flow) {
    vec4 row0 = light_at(light, 0u);
    uint type = uint(row0.w);
    if (type == LIGHT_SPHERE) {
        float t = sphere_hit(row0.xyz, light_at(light, 1u).w, o, d);
        return t < t_max ? t : INFINITY;
    }
    if (type >= LIGHT_DISTANT) return INFINITY;
    vec3 u = light_at(light, 1u).xyz;
    vec3 v = light_at(light, 2u).xyz;
    vec3 n = light_at(light, 3u).xyz;
    float facing = dot(d, n);
    if (light_flow ? facing <= 0.0 : facing >= 0.0) return INFINITY;
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
    } else if (type == LIGHT_DOME) {
        pdf = dome_pdf(light, d);
    } else {
        vec4 row3 = light_at(light, 3u);
        float cos_light = max(1e-12, -dot(d, row3.xyz));
        pdf = t * t / (cos_light * row3.w);
    }
    return pdf;
}

// -- Light paths from afar (connect.glsl): domes and distant lights with a size
// start light paths on a disk facing them, centred on what the camera sees.

#define K_EMIT 21           // x the chance a light path starts at a distant or dome light, y the disk's radius
#define K_EMIT_CENTER 22    // the disk's center

bool emits_from_afar(uint light) {
    uint type = light_type(light);
    return type == LIGHT_DOME || (type == LIGHT_DISTANT && light_at(light, 2u).w < 1.0 - 1e-7);
}

// The chance a light path leaves light `light` along -w (w toward the light).
float afar_emission_pdf(uint light, vec3 w) {
    return constants[K_EMIT].x * light_at(light, 6u).y * light_hit_pdf(light, vec3(0.0), w, INFINITY);
}

// The area density of a light path's first surface p (|cos| `c` there to w)
// past the disk the path started on: zero where p lies outside its shadow.
float afar_first_density(vec3 p, vec3 w, float c) {
    float radius = constants[K_EMIT].y;
    vec3 offset = p - constants[K_EMIT_CENTER].xyz;
    vec3 across = offset - w * dot(offset, w);
    if (dot(across, across) > radius * radius) return 0.0;
    return c / (PI * radius * radius);
}
