#!/usr/bin/env bash
set -euo pipefail

PI="mathias@192.168.1.229"

# fiducial-server
rsync -avz \
    --exclude=".venv/" \
    --exclude="__pycache__/" \
    --exclude=".git/" \
    --exclude=".devenv/" \
    --exclude=".github/" \
    ./ $PI:~/fiducial-server

# projectorium
rsync -avz ~/Repositories/projectorium/build/ "$PI":~/projectorium
