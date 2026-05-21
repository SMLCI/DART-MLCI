#!/usr/bin/env bash
# Generate the README's demo media (hero GIF + per-chamber thumbnails) from the
# MP4s produced by scripts/generate_sak_videos.py.
#
# Workflow:
#   python scripts/generate_sak_videos.py
#   bash scripts/make_demo_assets.sh        # writes docs/assets/ locally
#
# The output lands under docs/assets/, which is .gitignore'd on main so the
# binaries do not bloat the project's git history. Publish the media via the
# orphan `media` branch instead (the README points at
# https://raw.githubusercontent.com/SMLCI/DART-MLCI/media/docs/assets/...):
#
#   # One-time setup
#   git checkout --orphan media
#   git rm -rf .
#   mkdir -p docs/assets
#   cp -r /path/to/previous/docs/assets/* docs/assets/   # or rerun this script
#   git add docs/assets && git commit -m "media v0.1.0"
#   git push -u origin media
#   git checkout develop
#
#   # Subsequent updates (using a worktree so you keep your dev branch checked
#   # out on the main working tree):
#   git worktree add /tmp/media-wt media
#   cp -r docs/assets/* /tmp/media-wt/docs/assets/
#   ( cd /tmp/media-wt && git add docs/assets && git commit -m "refresh media" && git push )
#   git worktree remove /tmp/media-wt
#
# Requires: ffmpeg on PATH, or imageio-ffmpeg available in the active env.

set -euo pipefail

VIDEOS_DIR="${VIDEOS_DIR:-scripts/output/sak_videos}"
ASSETS_DIR="${ASSETS_DIR:-docs/assets}"
THUMBS_DIR="$ASSETS_DIR/thumbnails"
TEASER_SOURCE="${TEASER_SOURCE:-$VIDEOS_DIR/OpenBox-collector-inner.mp4}"
TEASER_OUT="${TEASER_OUT:-$ASSETS_DIR/pipeline_teaser.gif}"
# Set TEASER_ONLY=1 to skip the per-chamber thumbnail loop (e.g. when only
# refreshing the hero GIF from a custom source).
TEASER_ONLY="${TEASER_ONLY:-0}"

FFMPEG="${FFMPEG:-}"
if [ -z "$FFMPEG" ]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        FFMPEG="ffmpeg"
    elif command -v python >/dev/null 2>&1 \
            && python -c "import imageio_ffmpeg" >/dev/null 2>&1; then
        FFMPEG=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
        echo "Using bundled ffmpeg: $FFMPEG"
    else
        echo "ERROR: ffmpeg not found on PATH and imageio_ffmpeg is not installed." >&2
        echo "Install either:  conda install -n <env> ffmpeg" >&2
        echo "             or: pip install imageio-ffmpeg" >&2
        exit 1
    fi
fi

# $VIDEOS_DIR is only required for the per-chamber thumbnail loop. In
# TEASER_ONLY mode the user supplies an explicit TEASER_SOURCE outside the
# loop's directory, so don't enforce it.
if [ "$TEASER_ONLY" != "1" ] && [ ! -d "$VIDEOS_DIR" ]; then
    echo "ERROR: $VIDEOS_DIR does not exist. Run generate_sak_videos.py first." >&2
    exit 1
fi

mkdir -p "$ASSETS_DIR"
if [ "$TEASER_ONLY" != "1" ]; then
    mkdir -p "$THUMBS_DIR"
fi

# --- Hero teaser GIF (~2-3 MB, 720px wide, 12 fps, palette-optimized) -------
if [ ! -f "$TEASER_SOURCE" ]; then
    echo "ERROR: $TEASER_SOURCE missing — cannot build hero GIF." >&2
    exit 1
fi

echo "Building hero GIF -> $TEASER_OUT"
# Two-pass: prime palettegen with synthetic red and blue patches so the marker
# colors survive quantization (matplotlib's 'x' marker is thin antialiased
# lines and gets dropped by palettegen otherwise). The primer frames are only
# fed to palettegen, not into the final GIF.
PALETTE_TMP=$(mktemp --suffix=.png)
trap 'rm -f "$PALETTE_TMP"' EXIT
"$FFMPEG" -y -loglevel error \
    -i "$TEASER_SOURCE" \
    -f lavfi -i "color=c=red:s=560x630:d=30:r=6" \
    -f lavfi -i "color=c=blue:s=560x630:d=30:r=6" \
    -filter_complex "[0:v]fps=6,scale=560:-1:flags=lanczos[v];[v][1:v][2:v]concat=n=3:v=1:a=0,palettegen=stats_mode=full:max_colors=128" \
    -frames:v 1 "$PALETTE_TMP"
"$FFMPEG" -y -loglevel error \
    -i "$TEASER_SOURCE" \
    -i "$PALETTE_TMP" \
    -filter_complex "[0:v]fps=6,scale=560:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
    "$TEASER_OUT"
echo "  size: $(du -h "$TEASER_OUT" | cut -f1)"

if [ "$TEASER_ONLY" = "1" ]; then
    echo
    echo "TEASER_ONLY=1 — skipping per-chamber thumbnails."
    echo "Done. Generated: $TEASER_OUT"
    exit 0
fi

# --- Per-chamber thumbnails (frame from the masking step at ~t=10.5s) -------
# This timestamp falls inside "Masking: ROI Mask Applied" where the full
# rotated frame is shown with the hatched gray overlay on the excluded
# regions — best for communicating microfluidic-structure removal at a glance.
THUMB_TS=10.5
for mp4 in "$VIDEOS_DIR"/*.mp4; do
    [ -e "$mp4" ] || continue
    name=$(basename "$mp4" .mp4)
    out="$THUMBS_DIR/$name.png"
    echo "Thumbnail $name"
    "$FFMPEG" -y -loglevel error -ss "$THUMB_TS" -i "$mp4" -frames:v 1 \
        -vf "scale=iw/2:ih/2:flags=lanczos" "$out"
done

echo
echo "Done. Generated:"
echo "  - $TEASER_OUT"
ls -lh "$THUMBS_DIR"
