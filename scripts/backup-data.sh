#!/bin/bash
# Backup gamified-hs data folder hourly
# Run via cron: 0 * * * * /home/ssugar/claude/gamified-hs/scripts/backup-data.sh

SOURCE="/home/ssugar/claude/gamified-hs/data"
BACKUP_DIR="/home/ssugar/backups/gamified-hs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/data_$TIMESTAMP"

# Create backup directory if needed
mkdir -p "$BACKUP_DIR"

# Create timestamped backup
cp -r "$SOURCE" "$BACKUP_PATH"

# Keep only last 48 backups (2 days worth of hourly backups)
cd "$BACKUP_DIR" && ls -1t | tail -n +49 | xargs -r rm -rf

echo "Backup completed: $BACKUP_PATH"
