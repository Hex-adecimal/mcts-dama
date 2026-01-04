/**
 * bench_framework.h - Benchmark Framework
 * 
 * Provides timing utilities and benchmark registration.
 */

#ifndef BENCH_FRAMEWORK_H
#define BENCH_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// =============================================================================
// TIMING UTILITIES
// =============================================================================

static inline double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static inline double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// =============================================================================
// BENCHMARK MACROS
// =============================================================================

#define BENCH_WARMUP_ITERATIONS 3
#define BENCH_MIN_ITERATIONS    10
#define BENCH_TARGET_TIME_MS    1000.0  // Run benchmark for at least 1 second

typedef struct {
    const char *name;
    double total_time_ms;
    int iterations;
    double ops_per_sec;
    double avg_time_us;
    double min_time_us;
    double max_time_us;
} BenchResult;

static inline void print_bench_header(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                        BENCHMARK RESULTS                               ║\n");
    printf("╠════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ %-35s │ %10s │ %12s │ %8s ║\n", "Benchmark", "Ops/sec", "Avg (μs)", "Iters");
    printf("╠════════════════════════════════════════════════════════════════════════╣\n");
}

static inline void print_bench_result(BenchResult *r) {
    printf("║ %-35s │ %10.0f │ %12.2f │ %8d ║\n", 
           r->name, r->ops_per_sec, r->avg_time_us, r->iterations);
}

static inline void print_bench_separator(const char *section) {
    printf("╠────────────────────────────────────────────────────────────────────────╣\n");
    printf("║ %-70s ║\n", section);
    printf("╠════════════════════════════════════════════════════════════════════════╣\n");
}

static inline void print_bench_footer(void) {
    printf("╚════════════════════════════════════════════════════════════════════════╝\n\n");
}

// Run a benchmark function and collect stats
#define RUN_BENCH(name, setup_code, bench_code, teardown_code) do { \
    BenchResult _result = {0}; \
    _result.name = name; \
    _result.min_time_us = 1e9; \
    _result.max_time_us = 0; \
    \
    /* Warmup */ \
    for (int _w = 0; _w < BENCH_WARMUP_ITERATIONS; _w++) { \
        setup_code; \
        bench_code; \
        teardown_code; \
    } \
    \
    /* Actual benchmark */ \
    double _start = get_time_ms(); \
    while (_result.total_time_ms < BENCH_TARGET_TIME_MS || \
           _result.iterations < BENCH_MIN_ITERATIONS) { \
        setup_code; \
        double _iter_start = get_time_us(); \
        bench_code; \
        double _iter_time = get_time_us() - _iter_start; \
        teardown_code; \
        \
        if (_iter_time < _result.min_time_us) _result.min_time_us = _iter_time; \
        if (_iter_time > _result.max_time_us) _result.max_time_us = _iter_time; \
        _result.iterations++; \
        _result.total_time_ms = get_time_ms() - _start; \
    } \
    \
    _result.avg_time_us = (_result.total_time_ms * 1000.0) / _result.iterations; \
    _result.ops_per_sec = _result.iterations / (_result.total_time_ms / 1000.0); \
    print_bench_result(&_result); \
} while(0)

// Simpler version for benchmarks that don't need setup/teardown
#define BENCH(name, code) RUN_BENCH(name, {}, code, {})

#endif // BENCH_FRAMEWORK_H
