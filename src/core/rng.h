#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <math.h>

typedef struct { uint32_t state; } RNG;

static inline void rng_seed(RNG *r, uint32_t s) { r->state = s ? s : 1; }

static inline uint32_t rng_u32(RNG *r) {
    uint32_t x = r->state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return r->state = x;
}

static inline float rng_f32(RNG *r) { 
    return (float)rng_u32(r) / (float)UINT32_MAX; 
}

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Marsaglia and Tsang method for Gamma distribution
static inline float rng_gamma(RNG *r, float alpha) {
    if (alpha < 1.0f) return rng_gamma(r, 1.0f + alpha) * powf(rng_f32(r), 1.0f / alpha);
    float d = alpha - 1.0f/3.0f, c = 1.0f / sqrtf(9.0f * d);
    while (1) {
        float x, v;
        do { 
            // Box-Muller for normal sample
            float u1 = rng_f32(r);
            float u2 = rng_f32(r);
            if (u1 < 1e-9f) u1 = 1e-9f;
            x = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2); 
            v = 1.0f + c * x; 
        } while (v <= 0);
        v = v * v * v;
        float u = rng_f32(r);
        if (u < 1.0f - 0.0331f * x*x*x*x || logf(u) < 0.5f * x*x + d * (1.0f - v + logf(v))) 
            return d * v;
    }
}

#endif // RNG_H
