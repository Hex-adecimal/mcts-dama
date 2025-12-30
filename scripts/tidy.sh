#!/bin/bash
# scripts/tidy.sh - Project Organization Tool
# Moves old/non-standard files to archive and enforces naming conventions.

ARCHIVE_DIR="out/archive"
DATA_ARCHIVE="$ARCHIVE_DIR/data"
LOGS_ARCHIVE="$ARCHIVE_DIR/logs"
MODELS_ARCHIVE="$ARCHIVE_DIR/models"

mkdir -p "$DATA_ARCHIVE" "$LOGS_ARCHIVE" "$MODELS_ARCHIVE"

# ==============================================================================
# 1. LOGS CLEANUP
# ==============================================================================
echo "--- Cleaning Logs ---"
# Move logs older than 24 hours to archive
find out/logs -name "*.log" -mtime +1 -exec mv {} "$LOGS_ARCHIVE" \;
echo "Moved old logs to $LOGS_ARCHIVE"

# ==============================================================================
# 2. DATA CLEANUP
# ==============================================================================
echo "--- Organizing Data ---"

# If we have the big legacy file, standardize it
if [ -f "out/data/run_3h_master.dat" ] && [ ! -f "out/data/master.dat" ]; then
    echo "Renaming legacy master dataset: run_3h_master.dat -> master.dat"
    mv "out/data/run_3h_master.dat" "out/data/master.dat"
fi

# Move any random .dat files that are NOT standard to archive
# Standard files: master.dat, active.dat, valid.dat
for f in out/data/*.dat; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    if [[ "$fname" != "master.dat" && "$fname" != "active.dat" && "$fname" != "valid.dat" ]]; then
        echo "Archiving $fname..."
        mv "$f" "$DATA_ARCHIVE/"
    fi
done

# Move random .bin data files (like selfplay_deduped.bin)
for f in out/data/*.bin; do
    [ -f "$f" ] || continue
    echo "Archiving $fname..."
    mv "$f" "$DATA_ARCHIVE/"
done


# ==============================================================================
# 3. MODELS CLEANUP
# ==============================================================================
echo "--- Organizing Models ---"

# Move versioned models to archive (keep cnn_weights.bin and cnn_weights_backup.bin)
for f in out/models/*.bin; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    if [[ "$fname" != "cnn_weights.bin" && "$fname" != "cnn_weights_backup.bin" ]]; then
        # Check if it's a versioned file we want to keep or archive
        # For now, archive everything else to keep folder clean
        echo "Archiving model $fname..."
        mv "$f" "$MODELS_ARCHIVE/"
    fi
done

echo ""
echo "Organization Complete."
echo "Master Data: out/data/master.dat"
echo "Active Data: out/data/active.dat"
echo "Active Model: out/models/cnn_weights.bin"
echo ""
