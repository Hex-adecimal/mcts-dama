/**
 * logging.h - Header-only logging module
 * 
 * Provides dual output: stdout + optional log file.
 * Usage:
 *   log_init("path/to/file.log");  // Optional, enables file logging
 *   log_printf("message %d\n", x); // Writes to stdout + file
 *   log_close();                   // Cleanup
 */

#ifndef LOGGING_H
#define LOGGING_H

#include <stdio.h>
#include <stdarg.h>

static FILE *g_log_file = NULL;

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

static inline void log_printf(const char *fmt, ...) {
    va_list args;
    
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    if (g_log_file) {
        va_start(args, fmt);
        vfprintf(g_log_file, fmt, args);
        va_end(args);
        fflush(g_log_file);
    }
}

#endif // LOGGING_H
