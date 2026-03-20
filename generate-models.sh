#!/bin/bash
#
# Build script for llmmllab
# Generates code from schemas and proto files
#

set -e

SCHEMAS_DIR="./llmmllab-schemas/"
MODELS_DIR="./models"

# Create log file
LOG_FILE="build.log"
echo "Starting build at $(date)" > "$LOG_FILE"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Generate proto files from schemas
generate_py_from_schemas() {
    log_info "Generating Python files from schemas..."
    
    for py_file in "$MODELS_DIR"/*.py; do
        py_filename=$(basename "$py_file")
        schema_name="${py_filename%.py}"
        schema_file="$SCHEMAS_DIR/${schema_name}.yaml"
        
        if [ -f "$schema_file" ]; then
            log_info "Generating $schema_name.py from $schema_name.yaml"
            schema2code "$schema_file" -l python -o "$py_file"
        else
            log_warn "Schema file not found: $schema_file"
        fi
    done
}

generate_py_from_schemas