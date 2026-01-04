#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <math.h>
#include <time.h>

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

// =============================================================================
// GLOBAL RNG (Thread-Local for Multi-threaded Safety)
// =============================================================================
// Use rng_global() to access a thread-local RNG instance. Call rng_global_init()
// once at program startup to seed the global RNG.
//
// THREAD SAFETY: Uses pthread_once for one-time initialization to prevent
// race conditions when multiple threads call rng_global() simultaneously.

#ifdef _OPENMP
#include <omp.h>
#define RNG_MAX_THREADS 64
#else
#define RNG_MAX_THREADS 1
#endif

#include <pthread.h>

// Global RNG storage
static RNG g_rng_pool[RNG_MAX_THREADS];
static uint32_t g_rng_base_seed = 0;  // Set before first rng_global() call
static pthread_once_t g_rng_once = PTHREAD_ONCE_INIT;

// Internal: One-time initialization (called by pthread_once)
static void rng_do_init(void) {
    uint32_t seed = g_rng_base_seed ? g_rng_base_seed : (uint32_t)time(NULL);
    for (int i = 0; i < RNG_MAX_THREADS; i++) {
        rng_seed(&g_rng_pool[i], seed ^ (i * 2654435761u));  // Golden ratio hash
    }
}

/**
 * Initialize the global RNG pool with a specific seed.
 * Thread-safe: Can be called from any thread, will only initialize once.
 * If called multiple times, only the first seed value is used.
 */
static inline void rng_global_init(uint32_t base_seed) {
    if (base_seed != 0) g_rng_base_seed = base_seed;  // Set before init
    pthread_once(&g_rng_once, rng_do_init);
}

/**
 * Get the thread-local global RNG. Thread-safe for OpenMP.
 * Auto-initializes with time-based seed if not already initialized.
 */
static inline RNG* rng_global(void) {
    pthread_once(&g_rng_once, rng_do_init);
#ifdef _OPENMP
    int tid = omp_get_thread_num();
    return &g_rng_pool[tid < RNG_MAX_THREADS ? tid : 0];
#else
    return &g_rng_pool[0];
#endif
}

#endif // RNG_H

