/**
 * test_framework.h - Minimal Unit Testing Framework
 * 
 * Usage:
 *   TEST(test_name) { ... assertions ... }
 *   ASSERT_TRUE(condition)
 *   ASSERT_FALSE(condition)
 *   ASSERT_EQ(expected, actual)
 *   ASSERT_NE(expected, actual)
 *   ASSERT_FLOAT_EQ(expected, actual, epsilon)
 */

#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// --- Test Registry ---
typedef void (*TestFunc)(void);

typedef struct {
    const char *name;
    TestFunc func;
} TestCase;

#define MAX_TESTS 256
static TestCase g_tests[MAX_TESTS];
static int g_test_count = 0;
static int g_current_test_passed = 1;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static const char *g_current_test_name = NULL;

// --- Registration Function ---
static inline void register_test(const char *name, TestFunc func) {
    if (g_test_count < MAX_TESTS) {
        g_tests[g_test_count].name = name;
        g_tests[g_test_count].func = func;
        g_test_count++;
    }
}

// --- Test Definition Macro (uses explicit registration) ---
#define TEST(name) \
    static void test_##name(void); \
    static void test_##name(void)

// --- Assertions ---
#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("    FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_current_test_passed = 0; \
        return; \
    } \
} while(0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(expected, actual) do { \
    long long _e = (long long)(expected); \
    long long _a = (long long)(actual); \
    if (_e != _a) { \
        printf("    FAIL: %s:%d: expected %lld, got %lld\n", __FILE__, __LINE__, _e, _a); \
        g_current_test_passed = 0; \
        return; \
    } \
} while(0)

#define ASSERT_NE(expected, actual) do { \
    long long _e = (long long)(expected); \
    long long _a = (long long)(actual); \
    if (_e == _a) { \
        printf("    FAIL: %s:%d: expected != %lld\n", __FILE__, __LINE__, _e); \
        g_current_test_passed = 0; \
        return; \
    } \
} while(0)

#define ASSERT_FLOAT_EQ(expected, actual, epsilon) do { \
    double _e = (double)(expected); \
    double _a = (double)(actual); \
    if (fabs(_e - _a) > (epsilon)) { \
        printf("    FAIL: %s:%d: expected %.6f, got %.6f (eps=%.6f)\n", __FILE__, __LINE__, _e, _a, (double)(epsilon)); \
        g_current_test_passed = 0; \
        return; \
    } \
} while(0)

#define ASSERT_STR_EQ(expected, actual) do { \
    if (strcmp((expected), (actual)) != 0) { \
        printf("    FAIL: %s:%d: expected \"%s\", got \"%s\"\n", __FILE__, __LINE__, (expected), (actual)); \
        g_current_test_passed = 0; \
        return; \
    } \
} while(0)

#define ASSERT_NOT_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        printf("    FAIL: %s:%d: expected non-NULL\n", __FILE__, __LINE__); \
        g_current_test_passed = 0; \
        return; \
    } \
} while(0)

#define ASSERT_GT(a, b) ASSERT_TRUE((a) > (b))
#define ASSERT_GE(a, b) ASSERT_TRUE((a) >= (b))
#define ASSERT_LT(a, b) ASSERT_TRUE((a) < (b))
#define ASSERT_LE(a, b) ASSERT_TRUE((a) <= (b))

// --- Test Runner ---
static inline void run_tests(const char *filter) {
    printf("\n=== Running Tests ===\n\n");
    
    for (int i = 0; i < g_test_count; i++) {
        // Filter by module name if provided
        if (filter && strstr(g_tests[i].name, filter) == NULL) {
            continue;
        }
        
        g_current_test_passed = 1;
        g_current_test_name = g_tests[i].name;
        
        g_tests[i].func();
        
        if (g_current_test_passed) {
            printf("  ✓ %s\n", g_tests[i].name);
            g_tests_passed++;
        } else {
            printf("  ✗ %s\n", g_tests[i].name);
            g_tests_failed++;
        }
    }
    
    printf("\n=== Results ===\n");
    printf("Passed: %d | Failed: %d | Total: %d\n\n", 
           g_tests_passed, g_tests_failed, g_tests_passed + g_tests_failed);
}

// --- Registration Macro ---
#define REGISTER_TEST(name) register_test(#name, test_##name)

#endif // TEST_FRAMEWORK_H
