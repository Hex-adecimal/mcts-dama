/**
 * debug.h - Debug Assertions
 * 
 * Runtime invariant checks for development.
 * Disabled in release builds via NDEBUG.
 * 
 * Uses log_error + return instead of abort() for graceful handling.
 * 
 * Macro prefix: DBG_ (to avoid conflict with test_framework.h)
 */

#ifndef DEBUG_H
#define DEBUG_H

#include "dama/common/logging.h"
#include "dama/common/error_codes.h"

// =============================================================================
// DEBUG ASSERT MACROS
// =============================================================================

#ifdef NDEBUG
    // Release: No overhead
    #define DBG_ASSERT(cond, msg) ((void)0)
    #define DBG_ASSERT_RETURN(cond, msg, ret) ((void)0)
#else
    // Debug: Log error (no abort - graceful handling)
    #define DBG_ASSERT(cond, msg) \
        do { \
            if (!(cond)) { \
                log_error("[ASSERT] %s", (msg)); \
            } \
        } while(0)
    
    // Assert with return on failure
    #define DBG_ASSERT_RETURN(cond, msg, ret) \
        do { \
            if (!(cond)) { \
                log_error("[ASSERT] %s", (msg)); \
                return (ret); \
            } \
        } while(0)
#endif

// =============================================================================
// CONVENIENCE MACROS
// =============================================================================

// NULL pointer checks
#define DBG_NOT_NULL(ptr) \
    DBG_ASSERT((ptr) != NULL, #ptr " is NULL")

#define DBG_NOT_NULL_RETURN(ptr, ret) \
    DBG_ASSERT_RETURN((ptr) != NULL, #ptr " is NULL", ret)

// Valid square index (0-63)
#define DBG_VALID_SQ(sq) \
    DBG_ASSERT((sq) >= 0 && (sq) < 64, "invalid square index")

// Valid color (0 or 1)
#define DBG_VALID_COLOR(c) \
    DBG_ASSERT((c) == 0 || (c) == 1, "invalid color")

// Positive value
#define DBG_POSITIVE(val) \
    DBG_ASSERT((val) > 0, #val " must be positive")

#endif // DEBUG_H
