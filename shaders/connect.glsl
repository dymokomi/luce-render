// Light tracing and its MIS (docs/research/FIREFLIES-STUDY.md §3.2). Light
// paths start at emitters chosen by power, scatter like camera paths, and at
// each surface they reach connect to the camera with a shadow ray; what gets
// through lands, atomically, on the pixel it projects to (FILM_SPLAT). Camera
// paths keep their two techniques, a BSDF ray hitting a light and a light
// sample, so every path of light has three ways to be made:
//   s0  the camera path's BSDF ray hits the light
//   s1  the camera path's light sample reaches it
//   t1  a light path reaches the vertex the camera sees, and connects
// Each technique weighs what it finds by the power heuristic over all three.
// Their densities are products over the path's vertices (area measure); both
// sides carry the ratio of the other side's density to their own along the
// path (SV_MIS), and each vertex's factor is settled once the direction
// beyond it is known. A density is always the sampler's own, so both sides
// compute the same numbers: a camera vertex's BSDF pdf comes from the surface
// prepared for the direction the camera arrived from; a light vertex's from
// light_lobes, whose lobe choice holds for every direction (light_sampler_pdf).
//
// Light paths ignore distant lights and the background (their s0/s1 paths have
// no t1 to weigh against), orthographic cameras take no light paths, and light
// paths count only the total bounce limit's reach.
// Needs common.glsl, lights.glsl, openpbr.glsl and lighttree.glsl before it.

// The pixel the camera sees point p through from the lens point `lens`, and
// the unit direction and distance from there to p; false when p is behind the
// camera or off the image. A thin lens images the point where the ray meets
// the focus plane through the lens's center.
bool camera_pixel(vec3 p, vec3 lens, out uint pixel, out vec3 d, out float distance_to) {
    vec3 offset = p - lens;
    distance_to = length(offset);
    d = offset / max(distance_to, 1e-20);
    float c = dot(d, constants[K_VIEW].xyz);
    if (c <= 1e-6 || distance_to <= 0.0) return false;
    // On the image plane at unit distance, which the corner lies on.
    float focus = constants[K_CAMERA].w;
    vec3 film = (constants[K_CAMERA].z > 0.0 ? (lens + d * (focus / c) - params.position.xyz) / focus : d / c) - params.corner.xyz;
    float x = dot(film, params.du.xyz) / dot(params.du.xyz, params.du.xyz);
    float y = dot(film, params.dv.xyz) / dot(params.dv.xyz, params.dv.xyz);
    if (!(x >= 0.0 && y >= 0.0 && x < params.du.w && y < params.dv.w)) return false;
    pixel = uint(y) * uint(params.du.w) + uint(x);
    return true;
}

// The pdf the light sampler at this point gives `to` when the light path
// arrived from `from` (world, unit, both leaving the point): the surface (`s`,
// read) seen from `from`'s side, choosing lobes by light_lobes. Zero where the
// sampler would reject `to` (a shading normal on the wrong side of the true
// surface).
float light_sampler_pdf(Surface s, vec3 from, vec3 to) {
    bool same_side = dot(from, s.geometric) >= 0.0;
    Surface v = same_side ? s : turned(s);
    vec3 wo = to_local(v, from);
    vec3 wi = to_local(v, to);
    if ((dot(to, v.geometric) > 0.0) != (wi.z > 0.0)) return 0.0;
    if (!same_side) read_material(v);
    light_lobes(v);
    return surface_pdf(v, wi, wo);
}

// The power heuristic's weight for a technique, from the other two's
// densities over its own; safe for infinite ratios.
float mis_weight(float a, float b) {
    a = min(a, 1e18);
    b = min(b, 1e18);
    return 1.0 / (1.0 + a * a + b * b);
}

// Light emission: an emitter's chance of starting a light path (its power over
// the tree's), from its trail, as tree_sample walks it by importance.
float emitter_power_pdf(uint emitter) {
    uint record = uint(constants[K_EMITTERS].x) + emitter * 4u;
    uint trail = emitters[record + 1u];
    uint depth = emitters[record + 2u];
    uint base = uint(constants[K_TREE].y);
    uint node = 0u;
    float pdf = 1.0;
    for (uint level = 0u; level < depth; level++) {
        uint link = emitters[node * 2u];
        float left = max(lights[base + link * 3u].w, 0.0);
        float right = max(lights[base + (link + 1u) * 3u].w, 0.0);
        if (left + right <= 0.0) return 0.0;
        uint bit = (trail >> level) & 1u;
        pdf *= (bit == 0u ? left : right) / (left + right);
        node = link + bit;
    }
    return pdf;
}

// An emitter chosen by power, and its chance.
bool emitter_by_power(float u, out uint emitter, out float pdf) {
    pdf = 1.0;
    emitter = 0u;
    if (constants[K_TREE].x < 1.0) return false;
    uint base = uint(constants[K_TREE].y);
    uint node = 0u;
    for (uint depth = 0u; depth < 64u; depth++) {
        uint link = emitters[node * 2u];
        if ((emitters[node * 2u + 1u] & LEAF_NODE) != 0u) {
            emitter = link;
            return pdf > 0.0;
        }
        float left = max(lights[base + link * 3u].w, 0.0);
        float right = max(lights[base + (link + 1u) * 3u].w, 0.0);
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

// What light emission samples at an emitter: the area density of its point
// (the emitter's chance included) and the solid-angle density of a direction
// leaving it at |cos| `c` to its normal (cosine-weighted; both sides of an
// emissive triangle).
float emitter_area_pdf(uint emitter) {
    uint code = emitter_code(emitter);
    float area;
    if ((code & EMITTER_TRIANGLE) != 0u) {
        uint triangle = code & ~EMITTER_TRIANGLE;
        vec3 a = tree_point(indices[triangle * 3u]);
        area = 0.5 * length(cross(tree_point(indices[triangle * 3u + 1u]) - a, tree_point(indices[triangle * 3u + 2u]) - a));
    } else if (light_type(code) == LIGHT_SPHERE) {
        float r = light_at(code, 1u).w;
        area = 4.0 * PI * r * r;
    } else {
        area = light_at(code, 3u).w;
    }
    return area > 0.0 ? emitter_power_pdf(emitter) / area : 0.0;
}

float emission_direction_pdf(uint emitter, float c) {
    bool two_sided = (emitter_code(emitter) & EMITTER_TRIANGLE) != 0u;
    return abs(c) / (two_sided ? 2.0 * PI : PI);
}

// Camera side, at a light or emitter the BSDF ray from the last vertex hit:
// the light tracer's density over the camera's for the path so far (the
// running product `mis.x`) times the last two vertices' factors. `cos_light`
// is the cosine at the light toward the vertex, `distance_to` from the
// vertex, `bsdf_pdf` the ray's (solid angle).
float camera_hit_ratio(vec4 mis, uint emitter, float cos_light, float distance_to, float bsdf_pdf) {
    if (emitter == NONE || mis.y <= 0.0 || bsdf_pdf <= 0.0) return 0.0;
    float d2 = distance_to * distance_to;
    float at_vertex = emission_direction_pdf(emitter, cos_light) * mis.z / d2 / mis.y;
    float at_light = emitter_area_pdf(emitter) / (bsdf_pdf * abs(cos_light) / d2);
    return mis.x * at_vertex * at_light;
}
