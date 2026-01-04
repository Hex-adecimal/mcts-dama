/**
 * logging.h - Structured Logging Module with Severity Levels
 * 
 * Features:
 *   - 5 log levels: ERROR, WARN, INFO, DEBUG, VERBOSE
 *   - Color output for terminal (ANSI codes)
 *   - Dual output: stdout/stderr + optional file
 *   - Level filtering (set minimum level to display)
 * 
 * Usage:
 *   log_init("path/to/file.log");   // Optional file logging
 *   log_set_level(LOG_DEBUG);       // Set minimum level
 *   log_error("Failed: %s", msg);   // Always shown (red)
 *   log_warn("Warning: %d", x);     // Yellow
 *   log_info("Info: %s", msg);      // Default (white)
 *   log_debug("Debug data: %d", n); // Only if level >= DEBUG
 *   log_verbose("Trace: %p", ptr);  // Only if level >= VERBOSE
 *   log_close();                    // Cleanup
 */

#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>
#include <stdarg.h>

// =============================================================================
// LOG LEVELS
// =============================================================================

typedef enum {
    LOG_ERROR   = 0,    // Critical errors (always shown)
    LOG_WARN    = 1,    // Warnings
    LOG_INFO    = 2,    // Normal output (default)
    LOG_DEBUG   = 3,    // Debug information
    LOG_VERBOSE = 4     // Maximum verbosity
} LogLevel;

// =============================================================================
// ANSI COLOR CODES
// =============================================================================

#define LOG_COLOR_RESET   "\033[0m"
#define LOG_COLOR_RED     "\033[1;31m"
#define LOG_COLOR_YELLOW  "\033[1;33m"
#define LOG_COLOR_GREEN   "\033[0;32m"
#define LOG_COLOR_CYAN    "\033[0;36m"
#define LOG_COLOR_GRAY    "\033[0;90m"

// =============================================================================
// GLOBAL STATE
// =============================================================================

static FILE *g_log_file = NULL;
static LogLevel g_log_level = LOG_INFO;
static int g_log_use_color = 1;

// =============================================================================
// CONFIGURATION
// =============================================================================

static inline void log_init(const char *path) {
    if (path && path[0]) {
        g_log_file = fopen(path, "w");
    }
}

static inline void log_close(void) {
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

static inline void log_set_level(LogLevel level) {
    g_log_level = level;
}

static inline void log_set_color(int enabled) {
    g_log_use_color = enabled;
}

// =============================================================================
// CORE LOGGING FUNCTION
// =============================================================================

static inline void log_message(LogLevel level, const char *color, 
                               FILE *stream, const char *fmt, va_list args) {
    if (level > g_log_level) return;
    
    // Terminal output with optional color
    if (g_log_use_color && color) {
        fprintf(stream, "%s", color);
    }
    vfprintf(stream, fmt, args);
    if (g_log_use_color && color) {
        fprintf(stream, "%s", LOG_COLOR_RESET);
    }
    fprintf(stream, "\n");
    fflush(stream);
    
    // File output (no color)
    if (g_log_file) {
        va_list args_copy;
        va_copy(args_copy, args);
        vfprintf(g_log_file, fmt, args_copy);
        fprintf(g_log_file, "\n");
        fflush(g_log_file);
        va_end(args_copy);
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

static inline void log_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_ERROR, LOG_COLOR_RED, stderr, fmt, args);
    va_end(args);
}

static inline void log_warn(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_WARN, LOG_COLOR_YELLOW, stderr, fmt, args);
    va_end(args);
}

static inline void log_info(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_INFO, NULL, stdout, fmt, args);
    va_end(args);
}

static inline void log_debug(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_DEBUG, LOG_COLOR_CYAN, stdout, fmt, args);
    va_end(args);
}

static inline void log_verbose(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message(LOG_VERBOSE, LOG_COLOR_GRAY, stdout, fmt, args);
    va_end(args);
}

// =============================================================================
// BACKWARD COMPATIBILITY
// =============================================================================

// log_printf now works like before but uses log_info internally
// Note: We keep the old signature without automatic newline for compatibility
static inline void log_printf(const char *fmt, ...) {
    if (LOG_INFO > g_log_level) return;
    
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    if (g_log_file) {
        va_start(args, fmt);
        vfprintf(g_log_file, fmt, args);
        fflush(g_log_file);
        va_end(args);
    }
}

#endif // LOGGING_H
