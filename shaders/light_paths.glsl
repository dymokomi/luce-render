// Light paths' hits (connect.glsl), shaded in shade_surface's dispatch after
// the camera paths', so the two kinds' latencies overlap. A light path that
// reaches a light ends. At a surface it may pass through (cut-out opacity);
// otherwise it connects to the camera — a shadow ray carrying what the camera
// would see of it, MIS-weighted, to the pixel it projects to — and scatters
// on: a direction from the BSDF as seen from where the light came, weighed by
// the BSDF as the camera would evaluate it (radiance transport), with the
// shading-normal correction light paths need (Veach §5.3). Russian roulette
// and the total bounce limit end it. Needs what shade_surface includes.

// The light sample a camera path at this point would take toward the light
// path's first point x0 (emitter `emitter`), over its BSDF pdf `bsdf_pdf` toward
// it; the surface `s` faces the camera path's side.
float light_sample_ratio(uint emitter, Surface s, vec3 p, vec3 x0, float bsdf_pdf) {
    if (!(bsdf_pdf > 0.0)) return 0.0;
    vec3 from = offset_origin(p, s.geometric);
    vec3 offset = x0 - from;
    float reach = length(offset);
    vec3 d = offset / reach;
    float chance = (1.0 - constants[K_TREE].z) * tree_pdf(emitter, from, s.normal);
    uint code = emitter_code(emitter);
    float pdf = (code & EMITTER_TRIANGLE) != 0u ? triangle_hit_pdf(code & ~EMITTER_TRIANGLE, d, reach) : light_hit_pdf(code, from, d, reach);
    return chance * pdf / bsdf_pdf;
}

void shade_light(uint path) {
    uint what = get_u(SU_HIT, path);
    if ((what & LIGHT_HIT) != 0u) return;
    uint emitter = get_u(SU_PIXEL, path);
    uint bounces = get_u(SU_BOUNCE, path);
    uint total = bounce_total(bounces);
    vec4 hit = get_v(SV_HIT, path);
    vec3 origin = get_v(SV_ORIGIN, path).xyz;
    vec4 ray = get_v(SV_DIRECTION, path);
    Spec throughput = get_v(SV_THROUGHPUT, path);
    vec4 lambda = get_v(SV_LAMBDA, path);
    vec4 vertex_row = get_v(SV_VERTEX, path);
    vec4 mis = get_v(SV_MIS, path);

    uint triangle = what;
    vec3 p0 = point_of(indices[triangle * 3u]);
    vec3 geometric = normalize(cross(point_of(indices[triangle * 3u + 1u]) - p0, point_of(indices[triangle * 3u + 2u]) - p0));
    bool back = dot(geometric, ray.xyz) > 0.0;
    if (back) geometric = -geometric;
    uvec4 attributes = shading[triangle];
    vec3 p = origin + hit.x * ray.xyz;
    vec4 u_opacity = sample4(LIGHT_PATH_ID(path), params.sample_index, GROUP_OPACITY(total));

    // Cut-out opacity, as camera paths take it.
    float opacity = clamp(material_at(attributes.w, 12u).x, 0.0, 1.0);
    if (CUTOUTS && opacity < 1.0 && fract(u_opacity.x + float(hash_u32(floatBitsToUint(hit.x))) * (1.0 / 4294967296.0)) >= opacity) {
        set_v(SV_ORIGIN, path, vec4(offset_origin(p, -geometric), 0.0));
        queue_push(Q_CLOSEST + 1u - params.current, path);
        return;
    }
    // Paths of more vertices than camera paths reach are not counted.
    if (total >= uint(constants[K_INFO].y)) return;

    vec3 normal = normalize((1.0 - hit.z - hit.w) * octahedral_decode(attributes.x) + hit.z * octahedral_decode(attributes.y) + hit.w * octahedral_decode(attributes.z));
    if (dot(normal, geometric) < 0.0) normal = -normal;
    // The surface on the side light arrives from, read once: seen from the
    // camera or the next vertex on that side, it is only oriented.
    Surface lit = surface_at(normal, geometric, attributes.w, back);
    vec3 toward_light = -ray.xyz;
    read_material(lit);
    read_colors(lit, lambda);
    // The light tracer's density of this vertex, and of the last one as the
    // camera's sampler here would make it (back_reach × its pdf).
    vec3 x_last = vertex_row.xyz;
    float reach = distance(x_last, p);
    float cos_in = abs(dot(ray.xyz, geometric));
    float density = ray.w * cos_in / (reach * reach);
    float back_reach = mis.z / (reach * reach);
    // Light arriving on a shading normal's far side cannot be shaded.
    float cos_shading_in = dot(toward_light, normal);
    if (cos_shading_in <= 0.0 || cos_in < 1e-6) return;
    float shading_in = cos_shading_in / cos_in;

    // Connect to the camera.
    uint pixel;
    vec3 to_point;
    float camera_distance;
    if (camera_pixel(p, pixel, to_point, camera_distance)) {
        vec3 to_camera = -to_point;
        float side = dot(to_camera, geometric);
        Surface seen = side > 0.0 ? lit : turned(lit);
        vec3 wo = to_local(seen, to_camera);
        vec3 wi = to_local(seen, toward_light);
        if ((dot(toward_light, seen.geometric) > 0.0) == (wi.z > 0.0) && wo.z > 0.0) {
            if (side > 0.0) orient_surface(seen, wo);
            else prepare_surface(seen, wo, lambda);
            float camera_bsdf_pdf;
            Spec f = evaluate_surface(seen, wi, wo, camera_bsdf_pdf);
            float camera_density = camera_pdf(to_point) * abs(side) / (camera_distance * camera_distance);
            Spec contribution = throughput * f * (shading_in * camera_density / light_paths());
            if (spec_max(contribution) > 0.0) {
                // The camera techniques' densities over this one's.
                float last = mis.y > 0.0 ? camera_bsdf_pdf * back_reach / mis.y : 0.0;
                float ratio = mis.x * last * camera_density / density;
                float sample_ratio = total == 0u ? light_sample_ratio(emitter, seen, p, x_last, camera_bsdf_pdf) : max(mis.w, 0.0);
                float weight = mis_weight(ratio / light_paths(), ratio * sample_ratio / light_paths());
                float margin = 4e-4 * max(1.0, max(abs(p.x), max(abs(p.y), abs(p.z))));
                set_v(SV_SHADOW_ORIGIN, path, vec4(offset_origin(p, seen.geometric), max(0.0, camera_distance - margin)));
                set_v(SV_SHADOW_DIRECTION, path, vec4(to_camera, 0.0));
                // In the film's terms already: intersection kernels read no spectra.
                set_v(SV_SHADOW_RADIANCE, path, vec4(spec_to_film(contribution * weight, lambda), 0.0));
                set_u(SU_SPLAT, path, pixel);
                queue_push(Q_SHADOW, path);
            }
        }
    }

    // Scatter on, unless the next vertex would be past the bounce limit.
    if (total + 1u >= uint(constants[K_INFO].y)) return;
    vec4 u = sample4(LIGHT_PATH_ID(path), params.sample_index, GROUP_BSDF(total));
    Surface choosing = lit;
    light_lobes(choosing);
    vec3 wo_light = to_local(lit, toward_light);
    vec3 wi_light;
    float light_pdf;
    uint lobe;
    if (!sample_direction(choosing, wo_light, u.xyw, wi_light, light_pdf, lobe)) return;
    vec3 direction = to_world(lit, wi_light);
    float side = dot(direction, geometric);
    if ((side > 0.0) != (wi_light.z > 0.0)) return;
    // The BSDF as a camera arriving along `direction` evaluates it.
    Surface seen = side > 0.0 ? lit : turned(lit);
    vec3 wo = to_local(seen, direction);
    vec3 wi = to_local(seen, toward_light);
    if ((dot(toward_light, seen.geometric) > 0.0) != (wi.z > 0.0)) return;
    if (side > 0.0) orient_surface(seen, wo);
    else prepare_surface(seen, wo, lambda);
    float camera_bsdf_pdf;
    Spec f = evaluate_surface(seen, wi, wo, camera_bsdf_pdf);
    bounces = next_bounce(bounces, lobe);
    if (bounces == NONE) return;
    throughput *= f * (shading_in * abs(side) / light_pdf);
    float survival_scale = vertex_row.w;
    if (total + 1u >= 3u) {
        float survive = min(0.95, spec_max(throughput) * survival_scale);
        if (u.z >= survive) return;
        throughput /= survive;
    }
    if (!(spec_max(throughput) > 0.0)) return;
    // Settle the last vertex's factor; the light sample's ratio at the first.
    float last = mis.y > 0.0 ? camera_bsdf_pdf * back_reach / mis.y : 0.0;
    float sample_ratio = total == 0u ? light_sample_ratio(emitter, seen, p, x_last, camera_bsdf_pdf) : mis.w;
    set_u(SU_BOUNCE, path, bounces);
    set_v(SV_ORIGIN, path, vec4(offset_origin(p, side > 0.0 ? geometric : -geometric), 0.0));
    set_v(SV_VERTEX, path, vec4(p, survival_scale));
    set_v(SV_DIRECTION, path, vec4(direction, light_pdf));
    set_v(SV_THROUGHPUT, path, throughput);
    set_v(SV_MIS, path, vec4(mis.x * last, density, abs(side), sample_ratio));
    queue_push(Q_CLOSEST + 1u - params.current, path);
}
