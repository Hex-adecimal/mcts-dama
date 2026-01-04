/**
 * error_codes.h - Unified Error Codes
 * 
 * Standard return codes for functions that can fail.
 * Use these instead of raw -1, -2, etc.
 */

#ifndef ERROR_CODES_H
#define ERROR_CODES_H

typedef enum {
    ERR_OK           =  0,   // Success
    ERR_NULL_PTR     = -1,   // NULL pointer argument
    ERR_MEMORY       = -2,   // Memory allocation failed
    ERR_FILE_OPEN    = -3,   // Cannot open file
    ERR_FILE_FORMAT  = -4,   // Invalid file format
    ERR_VERSION      = -5,   // Version mismatch
    ERR_INVALID_ARG  = -6,   // Invalid argument value
    ERR_NOT_IMPL     = -7    // Not implemented
} ErrorCode;

// Helper macro for early return on error
#define RETURN_IF_NULL(ptr, ret) do { if (!(ptr)) return (ret); } while(0)

#endif // ERROR_CODES_H
