#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/Repositories/projectorium"
GODOT_BIN="$HOME/.var/app/io.github.MakovWait.Godots/data/godot/app_userdata/Godots/versions/Godot_v4_7_2-stable_mono_linux_x86_64/Godot_v4.7.2-stable_mono_linux_x86_64/Godot_v4.7.2-stable_mono_linux.x86_64"
XDG_DATA_HOME="$HOME/.var/app/io.github.MakovWait.Godots/data"
PRESET_NAME="Raspberry Pi"

mkdir -p "$PROJECT_DIR/build"
"$GODOT_BIN" --headless --path "$PROJECT_DIR" --export-release "$PRESET_NAME" "$PROJECT_DIR/build/projectorium.arm64"
