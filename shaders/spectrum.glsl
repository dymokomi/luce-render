// The kernels' color science (the host's is src/render/spectrum.lucb). In the
// spectral build a Spec holds radiance at four wavelengths; in the RGB build it
// holds ACEScg in xyz.

#define LAMBDA_MIN 360.0
#define LAMBDA_MAX 830.0

// pbrt-v4's visible-wavelength distribution: four hero-rotated wavelengths
// from one uniform number.
vec4 sample_wavelengths(float u) {
    vec4 lambda;
    for (int i = 0; i < 4; i++) {
        float v = fract(u + float(i) * 0.25);
        lambda[i] = 538.0 - 138.888889 * atanh(0.85691062 - 1.82750197 * v);
    }
    return lambda;
}

float wavelength_pdf(float lambda) {
    if (lambda < LAMBDA_MIN || lambda > LAMBDA_MAX) return 0.0;
    float c = cosh(0.0072 * (lambda - 538.0));
    return 0.0039398042 / (c * c);
}

float lobe(float x, float mean, float below, float above) {
    float t = (x - mean) / (x < mean ? below : above);
    return exp(-0.5 * t * t);
}

// Wyman, Sloan and Shirley's fit of the CIE 1931 2° observer.
vec3 cmf(float lambda) {
    return vec3(1.056 * lobe(lambda, 599.8, 37.9, 31.0) + 0.362 * lobe(lambda, 442.0, 16.0, 26.7) - 0.065 * lobe(lambda, 501.1, 20.4, 26.2),
                0.821 * lobe(lambda, 568.8, 46.9, 40.5) + 0.286 * lobe(lambda, 530.9, 16.3, 31.1),
                1.217 * lobe(lambda, 437.0, 11.8, 36.0) + 0.681 * lobe(lambda, 459.0, 26.0, 13.8));
}

float sigmoid(float z) { return 0.5 + z / (2.0 * sqrt(1.0 + z * z)); }

// A color as a Spec: its fitted spectrum (scale × S(c0 x² + c1 x + c2)) at the
// path's wavelengths, or its RGB.
Spec spec_of(vec4 fit, vec3 rgb, vec4 lambda) {
#if SPECTRAL
    vec4 x = (lambda - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN);
    vec4 z = (fit.x * x + fit.y) * x + fit.z;
    return fit.w * (0.5 + z / (2.0 * sqrt(1.0 + z * z)));
#else
    return vec4(rgb, 0.0);
#endif
}

// Planck's law divided by its value at its peak, written relative to the peak
// so f32 never sees Planck's constants: with x = c2 / (λT) (c2 = hc/k), the
// peak is at x = 4.965114, so B(λ)/B(peak) = (λpeak/λ)^5 (e^4.965114 - 1) / (e^x - 1).
float blackbody(float lambda, float kelvin) {
    float peak = 2.8977721e6 / kelvin;
    float x = 1.4387769e7 / (lambda * kelvin);
    float r = peak / lambda;
    return r * r * r * r * r * (142.32492 / (exp(x) - 1.0));
}

// An environment texel's ACEScg as a Spec: three smooth spectra that sum to
// the flat one (blue, green, red), weighted by K_BASIS's matrix of the color
// (src/render/environment.lucb), or the RGB as it is.
#define K_BASIS 18          // 3 rows: ACEScg to the basis spectra's weights
float logistic(float x) { return 1.0 / (1.0 + exp(-x)); }

Spec spec_of_texel(vec3 rgb, vec4 lambda) {
#if SPECTRAL
    vec3 w = vec3(dot(constants[K_BASIS].xyz, rgb), dot(constants[K_BASIS + 1].xyz, rgb), dot(constants[K_BASIS + 2].xyz, rgb));
    Spec s;
    for (int i = 0; i < 4; i++) {
        float blue = 1.0 - logistic((lambda[i] - 490.0) / 12.0);
        float red = logistic((lambda[i] - 595.0) / 12.0);
        s[i] = w.x * blue + w.y * (1.0 - blue - red) + w.z * red;
    }
    return s;
#else
    return vec4(rgb, 0.0);
#endif
}

// A Spec's luminance as the film sees it (Y), for path guiding.
float spec_luminance(Spec s, vec4 lambda) {
#if SPECTRAL
    float sum = 0.0;
    for (int i = 0; i < 4; i++) {
        float pdf = wavelength_pdf(lambda[i]);
        if (pdf > 0.0) sum += cmf(lambda[i]).y * (s[i] / pdf);
    }
    return sum * 0.25 / constants[K_CMF].y;
#else
    return dot(s.xyz, vec3(0.2722287168, 0.6740817658, 0.0536895174));
#endif
}

// The largest lane: what Russian roulette weighs.
float spec_max(Spec s) {
#if SPECTRAL
    return max(max(s.x, s.y), max(s.z, s.w));
#else
    return max(s.x, max(s.y, s.z));
#endif
}

// A path's radiance as the film stores it: XYZ under E (each channel by its
// integral) averaged over the four wavelengths, or ACEScg as is.
vec3 spec_to_film(Spec radiance, vec4 lambda) {
#if SPECTRAL
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        float pdf = wavelength_pdf(lambda[i]);
        if (pdf > 0.0) sum += cmf(lambda[i]) * (radiance[i] / pdf);
    }
    return sum * 0.25 / constants[K_CMF].xyz;
#else
    return radiance.xyz;
#endif
}
