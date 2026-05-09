#!/usr/bin/env bash
# Extract every ```mermaid ... ``` block from docs/pipeline_diagrams.md and
# render it to PNG (300 dpi) + SVG (vector) for use in posters / papers.
#
# Outputs:
#   plots/pipeline/<NAME>.png
#   plots/pipeline/<NAME>.svg
#
# Where <NAME> comes from a `%% NAME: foo` comment on the FIRST line of the
# Mermaid block. If absent, falls back to pipeline_<n>.
#
# Usage:
#   ./scripts/render_diagrams.sh
#
# Dependencies:
#   - npx (ships with Node.js / npm); no global install needed.
#   - The first run downloads @mermaid-js/mermaid-cli on demand (~30 s).
#
# To regenerate just one diagram, edit the SOURCE file (docs/pipeline_diagrams.md)
# and re-run this script. Mermaid blocks are picked up in document order.

set -euo pipefail
cd "$(dirname "$0")/.."

SRC="docs/pipeline_diagrams.md"
OUT_DIR="plots/pipeline"
mkdir -p "$OUT_DIR"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: $SRC not found" >&2
    exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
    echo "ERROR: npx not found. Install Node.js (https://nodejs.org)." >&2
    exit 1
fi

# Split out each fenced ```mermaid block into its own .mmd file
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

awk -v outdir="$TMP_DIR" '
    /^```mermaid$/ { in_block = 1; idx++; next }
    /^```$/ && in_block { in_block = 0; next }
    in_block { print > sprintf("%s/diagram_%02d.mmd", outdir, idx) }
' "$SRC"

count=$(ls "$TMP_DIR"/*.mmd 2>/dev/null | wc -l | tr -d " ")
if [[ "$count" -eq 0 ]]; then
    echo "ERROR: no Mermaid blocks found in $SRC" >&2
    exit 1
fi
echo "Found $count Mermaid block(s) in $SRC"
echo

# Render each block. Use a meaningful base name from `%% NAME:` comment if present.
i=0
for mmd in "$TMP_DIR"/diagram_*.mmd; do
    i=$((i + 1))
    # Look for `%% NAME: foo` directive on any line of the block
    # (POSIX-compatible classes — BSD sed on macOS does not support \s)
    name=$(grep -E '^%%[[:space:]]*NAME:[[:space:]]*' "$mmd" | head -1 \
           | sed -E 's/^%%[[:space:]]*NAME:[[:space:]]*//; s/[[:space:]]+$//')
    base="${name:-pipeline_$i}"
    echo "[$i/$count] rendering $(basename "$mmd")  →  $base.{png,svg}"
    npx -y @mermaid-js/mermaid-cli \
        -i "$mmd" \
        -o "$OUT_DIR/$base.png" \
        --backgroundColor white \
        --scale 3
    npx -y @mermaid-js/mermaid-cli \
        -i "$mmd" \
        -o "$OUT_DIR/$base.svg" \
        --backgroundColor white
done

echo
echo "Done. Files written to $OUT_DIR/:"
ls -la "$OUT_DIR"/*.{png,svg} 2>/dev/null
