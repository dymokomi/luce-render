// Materials' node graphs (src/render/shader_graph.lucb has the compiler and the
// opcodes): a program runs once a hit, before the surface is read, and leaves
// the OpenPBR parameters it drives in openpbr.glsl's program_values. Operands
// are constants or stack slots (NaN-tagged); colors and vectors take three.
// Programs read nothing that depends on direction, so camera and light paths
// see one surface. Needs openpbr.glsl, scene.glsl and texture tables.

#define K_PROGRAMS 24       // xy the programs' address
#define PROGRAM_STACK 64u
#define SLOT_TAG 0x7FC10000u

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ProgramWords { uint words[]; };

float program_stack[PROGRAM_STACK];

float program_operand(uint word) {
    return (word & 0xFFFF0000u) == SLOT_TAG ? program_stack[word & 0xFFFFu] : uintBitsToFloat(word);
}

// -- Procedurals.

// One of the cube's twelve edge directions, by a hash of the lattice point
// (Perlin's improved noise).
vec3 gradient_at(ivec3 c) {
    uint h = hash_u32(uint(c.x) * 73856093u ^ hash_u32(uint(c.y) * 19349663u ^ hash_u32(uint(c.z) * 83492791u))) % 12u;
    vec2 signs = vec2((h & 1u) != 0u ? -1.0 : 1.0, (h & 2u) != 0u ? -1.0 : 1.0);
    if (h < 4u) return vec3(signs, 0.0);
    if (h < 8u) return vec3(signs.x, 0.0, signs.y);
    return vec3(0.0, signs);
}

// Perlin gradient noise in about -1..1.
float gradient_noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    ivec3 c = ivec3(i);
    float n000 = dot(gradient_at(c), f);
    float n100 = dot(gradient_at(c + ivec3(1, 0, 0)), f - vec3(1, 0, 0));
    float n010 = dot(gradient_at(c + ivec3(0, 1, 0)), f - vec3(0, 1, 0));
    float n110 = dot(gradient_at(c + ivec3(1, 1, 0)), f - vec3(1, 1, 0));
    float n001 = dot(gradient_at(c + ivec3(0, 0, 1)), f - vec3(0, 0, 1));
    float n101 = dot(gradient_at(c + ivec3(1, 0, 1)), f - vec3(1, 0, 1));
    float n011 = dot(gradient_at(c + ivec3(0, 1, 1)), f - vec3(0, 1, 1));
    float n111 = dot(gradient_at(c + ivec3(1, 1, 1)), f - vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

// Fractal noise in 0..1: `detail` octaves (fractional ones blend), each
// `roughness` as strong as the last.
float fractal_noise(vec3 p, float detail, float roughness) {
    float octaves = clamp(detail, 0.0, 15.0);
    int whole = int(floor(octaves));
    float part = octaves - float(whole);
    float sum = 0.0;
    float total = 0.0;
    float amplitude = 1.0;
    for (int octave = 0; octave <= whole; octave++) {
        sum += gradient_noise(p) * amplitude;
        total += amplitude;
        amplitude *= clamp(roughness, 0.0, 1.0);
        p *= 2.0;
    }
    if (part > 0.0) {
        sum += gradient_noise(p) * amplitude * part;
        total += amplitude * part;
    }
    return clamp(0.5 + 0.5 * sum / total, 0.0, 1.0);
}

float math_of(uint operation, float a, float b) {
    switch (operation) {
    case 0u: return a + b;
    case 1u: return a - b;
    case 2u: return a * b;
    case 3u: return b != 0.0 ? a / b : 0.0;
    case 4u: return a >= 0.0 ? pow(a, b) : (b == floor(b) ? pow(-a, b) * (mod(b, 2.0) == 1.0 ? -1.0 : 1.0) : 0.0);
    case 5u: return min(a, b);
    case 6u: return max(a, b);
    case 7u: return abs(a);
    case 8u: return sin(a);
    case 9u: return cos(a);
    case 10u: return floor(a);
    default: return fract(a);
    }
}

// A tangent-space normal map's `color` on the hit's triangle: the frame its
// texture coordinates give about the shading normal n.
vec3 normal_from_map(uint triangle, vec3 n, vec3 color, float strength) {
    uint i0 = indices[triangle * 3u], i1 = indices[triangle * 3u + 1u], i2 = indices[triangle * 3u + 2u];
    vec3 p0 = vec3(positions[i0 * 3u], positions[i0 * 3u + 1u], positions[i0 * 3u + 2u]);
    vec3 e1 = vec3(positions[i1 * 3u], positions[i1 * 3u + 1u], positions[i1 * 3u + 2u]) - p0;
    vec3 e2 = vec3(positions[i2 * 3u], positions[i2 * 3u + 1u], positions[i2 * 3u + 2u]) - p0;
    vec2 uv0 = triangle_uv(triangle, vec2(0.0));
    vec2 d1 = triangle_uv(triangle, vec2(1.0, 0.0)) - uv0;
    vec2 d2 = triangle_uv(triangle, vec2(0.0, 1.0)) - uv0;
    float det = d1.x * d2.y - d2.x * d1.y;
    if (abs(det) < 1e-12) return n;
    vec3 t = (e1 * d2.y - e2 * d1.y) / det;
    vec3 b = (e2 * d1.x - e1 * d2.x) / det;
    t -= n * dot(n, t);
    if (dot(t, t) < 1e-20) return n;
    t = normalize(t);
    vec3 bt = cross(n, t);
    if (dot(bt, b) < 0.0) bt = -bt;
    vec3 c = color * 2.0 - 1.0;
    vec3 bent = t * c.x + bt * c.y + n * c.z;
    if (dot(bent, bent) < 1e-12) return n;
    return normalize(mix(n, normalize(bent), clamp(strength, 0.0, 1.0)));
}

// -- Bump: heights at the point and a footprint away along two tangents.

// The footprint: the camera pixel's width at p (`pixels` of them). It depends
// on the point alone, so camera and light paths see one bumped surface; a
// floor keeps the offsets above f32's resolution of p.
float bump_width(vec3 p, float pixels) {
    float pixel = length(params.du.xyz) * (params.position.w > 0.5 ? 1.0 : distance(p, params.position.xyz));
    vec3 a = abs(p);
    return max(pixel * max(pixels, 0.0), 4e-6 * (1.0 + max(a.x, max(a.y, a.z))));
}

// Two unit tangents about n (Duff et al.'s branchless frame).
void bump_tangents(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = vec3(c, s + n.y * n.y * a, -n.y);
}

// Where offset `axis` (0 none, 1 and 2 the tangents) moves the point.
vec3 bump_step(uint axis, float pixels, vec3 p, vec3 n) {
    if (axis == 0u) return vec3(0.0);
    vec3 t, b;
    bump_tangents(n, t, b);
    return (axis == 1u ? t : b) * bump_width(p, pixels);
}

// How the texture coordinates change along a step in the triangle's plane
// (zero for a degenerate triangle).
vec2 uv_step(uint triangle, vec3 step) {
    uint i0 = indices[triangle * 3u], i1 = indices[triangle * 3u + 1u], i2 = indices[triangle * 3u + 2u];
    vec3 p0 = vec3(positions[i0 * 3u], positions[i0 * 3u + 1u], positions[i0 * 3u + 2u]);
    vec3 e1 = vec3(positions[i1 * 3u], positions[i1 * 3u + 1u], positions[i1 * 3u + 2u]) - p0;
    vec3 e2 = vec3(positions[i2 * 3u], positions[i2 * 3u + 1u], positions[i2 * 3u + 2u]) - p0;
    float g11 = dot(e1, e1), g12 = dot(e1, e2), g22 = dot(e2, e2);
    float det = g11 * g22 - g12 * g12;
    if (abs(det) < 1e-24) return vec2(0.0);
    // The step's barycentric coordinates by the dual basis, then the uvs'.
    float a = (g22 * dot(e1, step) - g12 * dot(e2, step)) / det;
    float b = (g11 * dot(e2, step) - g12 * dot(e1, step)) / det;
    vec2 uv0 = triangle_uv(triangle, vec2(0.0));
    return (triangle_uv(triangle, vec2(1.0, 0.0)) - uv0) * a + (triangle_uv(triangle, vec2(0.0, 1.0)) - uv0) * b;
}

// Cycles' bump (svm_node_set_bump): the surface gradient of the heights
// about `normal` (zero: n), `distance` long, blended in by `strength`.
vec3 bumped(vec3 p, vec3 n, vec3 normal, float strength, float distance_scale, float pixels, float hc, float hx, float hy) {
    vec3 base = dot(normal, normal) > 1e-12 ? normalize(normal) : n;
    vec3 t, b;
    bump_tangents(n, t, b);
    float w = bump_width(p, pixels);
    vec3 dx = t * w, dy = b * w;
    vec3 rx = cross(dy, base), ry = cross(base, dx);
    float det = dot(dx, rx);
    vec3 gradient = (hx - hc) * rx + (hy - hc) * ry;
    vec3 out_normal = abs(det) * base - distance_scale * sign(det) * gradient;
    if (dot(out_normal, out_normal) < 1e-30) return base;
    return normalize(mix(base, normalize(out_normal), clamp(strength, 0.0, 1.0)));
}

// Run material `m`'s program at the hit on `triangle`: point p, shading
// normal n, texture coordinates surface_uv. Sets program_mask and values.
void run_program(uint m, uint triangle, vec3 p, vec3 n) {
    program_mask = 0u;
    if (!HAS_PROGRAMS) return;
    float entry = material_at(m, 13u).x;
    if (entry < 0.0) return;
    ProgramWords code = ProgramWords(address_of(constants[K_PROGRAMS].xy));
    uint pc = uint(entry);
    for (uint step = 0u; step < 4096u; step++) {
        uint op = code.words[pc++];
        if (op == 0u) return;
        if (op == 1u || op == 2u) {
            // A bump's offset copies read the point a step away.
            uint axis = code.words[pc++];
            vec3 step = bump_step(axis, program_operand(code.words[pc++]), p, n);
            vec3 v = op == 1u ? vec3(surface_uv + (axis != 0u ? uv_step(triangle, step) : vec2(0.0)), 0.0) : p + step;
            uint out0 = code.words[pc++]; uint out1 = code.words[pc++]; uint out2 = code.words[pc++];
            program_stack[out0] = v.x; program_stack[out1] = v.y; program_stack[out2] = v.z;
        } else if (op == 3u) {
            float v = program_operand(code.words[pc++]);
            program_stack[code.words[pc++]] = v;
        } else if (op == 4u) {
            vec3 v = vec3(program_operand(code.words[pc]), program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]));
            pc += 3u;
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = v[k];
        } else if (op == 5u) {
            float slot = float(code.words[pc++]);
            bool color = code.words[pc++] != 0u;
            vec3 at = vec3(program_operand(code.words[pc]), program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]));
            pc += 3u;
            vec4 texel = texture_at(slot, at.xy);
            vec3 rgb = color ? rec709_to_acescg * texel.rgb : texel.rgb;
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = rgb[k];
            program_stack[code.words[pc++]] = texel.a;
        } else if (op == 6u) {
            vec3 at = vec3(program_operand(code.words[pc]), program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]));
            vec3 a = vec3(program_operand(code.words[pc + 3u]), program_operand(code.words[pc + 4u]), program_operand(code.words[pc + 5u]));
            vec3 b = vec3(program_operand(code.words[pc + 6u]), program_operand(code.words[pc + 7u]), program_operand(code.words[pc + 8u]));
            float scale = program_operand(code.words[pc + 9u]);
            pc += 10u;
            // A small offset keeps faces at whole coordinates off the seams.
            ivec3 cell = ivec3(floor(at * scale + 1e-5));
            float fac = float((cell.x + cell.y + cell.z) & 1);
            vec3 rgb = fac > 0.5 ? a : b;
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = rgb[k];
            program_stack[code.words[pc++]] = fac;
        } else if (op == 7u) {
            vec3 at = vec3(program_operand(code.words[pc]), program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]));
            float scale = program_operand(code.words[pc + 3u]);
            float detail = program_operand(code.words[pc + 4u]);
            float roughness = program_operand(code.words[pc + 5u]);
            pc += 6u;
            vec3 q = at * scale;
            float fac = fractal_noise(q, detail, roughness);
            vec3 rgb = vec3(fac, fractal_noise(q + vec3(19.1, 33.4, 47.2), detail, roughness), fractal_noise(q + vec3(73.7, 11.3, 29.9), detail, roughness));
            program_stack[code.words[pc++]] = fac;
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = rgb[k];
        } else if (op == 8u) {
            uint operation = code.words[pc++];
            float a = program_operand(code.words[pc++]);
            float b = program_operand(code.words[pc++]);
            program_stack[code.words[pc++]] = math_of(operation, a, b);
        } else if (op == 9u) {
            float fac = clamp(program_operand(code.words[pc]), 0.0, 1.0);
            vec3 a = vec3(program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]), program_operand(code.words[pc + 3u]));
            vec3 b = vec3(program_operand(code.words[pc + 4u]), program_operand(code.words[pc + 5u]), program_operand(code.words[pc + 6u]));
            pc += 7u;
            vec3 rgb = mix(a, b, fac);
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = rgb[k];
        } else if (op == 10u) {
            float v = program_operand(code.words[pc]);
            float from_min = program_operand(code.words[pc + 1u]);
            float from_max = program_operand(code.words[pc + 2u]);
            float to_min = program_operand(code.words[pc + 3u]);
            float to_max = program_operand(code.words[pc + 4u]);
            pc += 5u;
            float t = from_max != from_min ? (v - from_min) / (from_max - from_min) : 0.0;
            program_stack[code.words[pc++]] = to_min + t * (to_max - to_min);
        } else if (op == 11u) {
            float v = program_operand(code.words[pc]);
            float low = program_operand(code.words[pc + 1u]);
            float high = program_operand(code.words[pc + 2u]);
            pc += 3u;
            program_stack[code.words[pc++]] = clamp(v, min(low, high), max(low, high));
        } else if (op == 12u) {
            vec3 v = vec3(program_operand(code.words[pc]), program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]));
            pc += 3u;
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = v[k];
        } else if (op == 13u) {
            vec3 v = vec3(program_operand(code.words[pc]), program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]));
            pc += 3u;
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = v[k];
        } else if (op == 14u) {
            float fac = program_operand(code.words[pc]);
            vec3 a = vec3(program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]), program_operand(code.words[pc + 3u]));
            vec3 b = vec3(program_operand(code.words[pc + 4u]), program_operand(code.words[pc + 5u]), program_operand(code.words[pc + 6u]));
            float first = program_operand(code.words[pc + 7u]);
            float last = program_operand(code.words[pc + 8u]);
            pc += 9u;
            float t = last != first ? clamp((fac - first) / (last - first), 0.0, 1.0) : (fac < first ? 0.0 : 1.0);
            vec3 rgb = mix(a, b, t);
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = rgb[k];
        } else if (op == 15u) {
            vec3 color = vec3(program_operand(code.words[pc]), program_operand(code.words[pc + 1u]), program_operand(code.words[pc + 2u]));
            float strength = program_operand(code.words[pc + 3u]);
            pc += 4u;
            vec3 bent = normal_from_map(triangle, n, color, strength);
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = bent[k];
        } else if (op == 17u) {
            float strength = program_operand(code.words[pc]);
            float distance_scale = program_operand(code.words[pc + 1u]);
            float pixels = program_operand(code.words[pc + 2u]);
            float hc = program_operand(code.words[pc + 3u]);
            float hx = program_operand(code.words[pc + 4u]);
            float hy = program_operand(code.words[pc + 5u]);
            vec3 normal = vec3(program_operand(code.words[pc + 6u]), program_operand(code.words[pc + 7u]), program_operand(code.words[pc + 8u]));
            pc += 9u;
            vec3 bent = bumped(p, n, normal, strength, distance_scale, pixels, hc, hx, hy);
            for (uint k = 0u; k < 3u; k++) program_stack[code.words[pc++]] = bent[k];
        } else if (op == 16u) {
            uint first = code.words[pc++];
            uint width = code.words[pc++];
            uint bit = code.words[pc++];
            for (uint k = 0u; k < width; k++) program_values[first + k] = program_operand(code.words[pc++]);
            program_mask |= 1u << bit;
        } else {
            return;
        }
    }
}
