#!/bin/bash
# Comprehensive smoke-test for the DART API running in Docker.
#
# Builds the image, starts a container on port 8001, and exercises every
# endpoint documented in docs/API_QUICK_START.md:
#   /health, /available-chips, /chamber-types,
#   /process-image, /process-image-preview, /calibrate
#
# Usage:
#   ./test_docker_api.sh                  # build + test
#   ./test_docker_api.sh --skip-build     # reuse existing dart-mlci:test image
#   ./test_docker_api.sh --image dart:dev # test a different image tag

set -euo pipefail

IMAGE="dart-mlci:test"
CONTAINER="dart-mlci-test"
PORT=8001
SKIP_BUILD=false
TEST_FIXTURE="tests/fixtures/calibration_image_0000.tif"
PASSED=0
FAILED=0
SKIPPED=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "Cleaning up..."
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true
}

pass() { echo "  PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL: $1"; FAILED=$((FAILED + 1)); }
skip() { echo "  SKIP: $1"; SKIPPED=$((SKIPPED + 1)); }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=true; shift ;;
        --image) IMAGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

trap cleanup EXIT

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [ "$SKIP_BUILD" = false ]; then
    echo "Building Docker image ($IMAGE)..."
    docker build -t "$IMAGE" . || { echo "Docker build failed"; exit 1; }
    echo ""
fi

# ---------------------------------------------------------------------------
# Start container
# ---------------------------------------------------------------------------
echo "Starting container on port $PORT..."
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d \
    --name "$CONTAINER" \
    -p "$PORT":8000 \
    "$IMAGE"

# Wait for the API to become ready (up to 60 s)
echo "Waiting for API to be ready..."
BASE="http://localhost:$PORT"
for i in $(seq 1 60); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then
        echo "API ready after ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "API did not become ready in 60 s. Container logs:"
        docker logs "$CONTAINER"
        exit 1
    fi
    sleep 1
done

echo ""
echo "========================================"
echo "  Testing DART API endpoints"
echo "========================================"

# ---------------------------------------------------------------------------
# 1. /health
# ---------------------------------------------------------------------------
echo ""
echo "--- /health (GET) ---"
HEALTH=$(curl -sf "$BASE/health")
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='healthy' and d['model_loaded']" 2>/dev/null; then
    pass "/health returns healthy with model loaded"
    echo "$HEALTH" | python3 -m json.tool
else
    fail "/health did not return healthy or model not loaded"
    echo "$HEALTH"
fi

# ---------------------------------------------------------------------------
# 2. /available-chips
# ---------------------------------------------------------------------------
echo ""
echo "--- /available-chips (GET) ---"
CHIPS=$(curl -sf "$BASE/available-chips")
if echo "$CHIPS" | python3 -c "import sys,json; d=json.load(sys.stdin); assert isinstance(d, list) and len(d)>0" 2>/dev/null; then
    pass "/available-chips returns non-empty list"
    echo "  Chips: $CHIPS"
else
    fail "/available-chips returned unexpected data"
    echo "$CHIPS"
fi

# ---------------------------------------------------------------------------
# 3. /chamber-types
# ---------------------------------------------------------------------------
echo ""
echo "--- /chamber-types (GET) ---"
TYPES=$(curl -sf "$BASE/chamber-types")
N_TYPES=$(echo "$TYPES" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
if [ "$N_TYPES" -gt 0 ]; then
    pass "/chamber-types returns $N_TYPES types"
else
    fail "/chamber-types returned no types"
    echo "$TYPES"
fi

# ---------------------------------------------------------------------------
# 4. /process-image  (requires test fixture)
# ---------------------------------------------------------------------------
echo ""
echo "--- /process-image (POST) ---"
if [ -f "$TEST_FIXTURE" ]; then
    IMAGE_B64=$(python3 -c "
import base64, sys
with open('$TEST_FIXTURE', 'rb') as f:
    sys.stdout.write(base64.b64encode(f.read()).decode())
")
    PROCESS_RESP=$(curl -sf -X POST "$BASE/process-image" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json, sys
print(json.dumps({
    'image': '$IMAGE_B64'[:80] + '...',  # placeholder, we'll pipe full below
    'roi_id': '0000',
    'pixel_size': 0.065789
}))
" 2>/dev/null || true)" 2>/dev/null || true)

    # The above won't work for large base64 — use python to send the request
    PROCESS_RESP=$(python3 -c "
import base64, json, urllib.request

with open('$TEST_FIXTURE', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

body = json.dumps({
    'image': image_b64,
    'roi_id': '0000',
    'pixel_size': 0.065789
}).encode()

req = urllib.request.Request(
    '$BASE/process-image',
    data=body,
    headers={'Content-Type': 'application/json'},
    method='POST'
)
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read().decode())
")

    PROCESS_STATUS=$(echo "$PROCESS_RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'success' in d, 'missing success field'
if d['success']:
    assert len(d.get('cropped_image', '')) > 0
    assert len(d.get('mask', '')) > 0
    ct = d.get('chamber_type', 'unknown')
    ang = d.get('rotation_angle', 0)
    print(f'SUCCESS chamber={ct} angle={ang:.2f}')
else:
    # A well-formed error is acceptable — the endpoint works,
    # the test image just may not have detectable markers.
    msg = d.get('error_message', 'unknown')
    print(f'HANDLED_ERROR msg={msg}')
" 2>/dev/null || echo "MALFORMED")

    if echo "$PROCESS_STATUS" | grep -q "^SUCCESS"; then
        pass "/process-image succeeded ($PROCESS_STATUS)"
    elif echo "$PROCESS_STATUS" | grep -q "^HANDLED_ERROR"; then
        pass "/process-image returned well-formed error ($PROCESS_STATUS)"
    else
        fail "/process-image returned malformed response"
        echo "$PROCESS_RESP" | python3 -m json.tool 2>/dev/null || echo "$PROCESS_RESP"
    fi
else
    skip "/process-image — test fixture not found: $TEST_FIXTURE"
fi

# ---------------------------------------------------------------------------
# 5. /process-image-preview  (requires test fixture)
# ---------------------------------------------------------------------------
echo ""
echo "--- /process-image-preview (POST) ---"
if [ -f "$TEST_FIXTURE" ]; then
    PREVIEW_RESP=$(python3 -c "
import base64, json, urllib.request, urllib.error

with open('$TEST_FIXTURE', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

body = json.dumps({
    'image': image_b64,
    'roi_id': '0000',
    'pixel_size': 0.065789
}).encode()

req = urllib.request.Request(
    '$BASE/process-image-preview',
    data=body,
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = resp.read().decode()
        status = resp.status
except urllib.error.HTTPError as e:
    content = e.read().decode()
    status = e.code

assert '<!DOCTYPE html>' in content, 'Missing DOCTYPE'
if status == 200:
    assert 'Cropped Image' in content, 'Missing Cropped Image section'
    assert 'data:image/png;base64,' in content, 'Missing embedded images'
    print('OK_SUCCESS')
elif status == 422:
    # Error HTML page — endpoint works, image processing just failed
    assert 'Error' in content, 'Missing error message in HTML'
    print('OK_ERROR_HTML')
else:
    print(f'UNEXPECTED_STATUS={status}')
")
    if echo "$PREVIEW_RESP" | grep -q "^OK"; then
        pass "/process-image-preview returns valid HTML ($PREVIEW_RESP)"
    else
        fail "/process-image-preview returned unexpected content"
        echo "$PREVIEW_RESP"
    fi
else
    skip "/process-image-preview — test fixture not found: $TEST_FIXTURE"
fi

# ---------------------------------------------------------------------------
# 6. /calibrate  (requires 3 calibration images — use same image 3x as smoke test)
# ---------------------------------------------------------------------------
echo ""
echo "--- /calibrate (POST) ---"
if [ -f "$TEST_FIXTURE" ]; then
    CAL_RESP=$(python3 -c "
import base64, json, urllib.request

with open('$TEST_FIXTURE', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Use the same image for all 3 calibration points (smoke test only —
# calibration quality doesn't matter, we just verify the endpoint works)
body = json.dumps({
    'chip_name': 'SAK',
    'calibration_images': [
        {'image': image_b64, 'roi_id': '0000',
         'stage_position': {'x': 0.0, 'y': 0.0}},
        {'image': image_b64, 'roi_id': '0000',
         'stage_position': {'x': 100.0, 'y': 0.0}},
        {'image': image_b64, 'roi_id': '0000',
         'stage_position': {'x': 0.0, 'y': 100.0}},
    ],
    'pixel_size': 0.065789,
    'blueprint_map_path': 'artifacts/sak_blueprint_map.csv'
}).encode()

req = urllib.request.Request(
    '$BASE/calibrate',
    data=body,
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
        # The calibration may or may not succeed with identical images,
        # but the endpoint should return a well-formed response
        assert 'success' in data, 'Missing success field'
        if data['success']:
            assert 'calibrated_map' in data
            assert 'statistics' in data
            stats = data['statistics']
            print(f\"OK success=True rmse={stats['rmse']:.4f} n_points={stats['n_points']}\")
        else:
            # Even failure is acceptable for a smoke test with dummy data
            print(f\"OK success=False msg={data.get('error_message', 'unknown')}\")
except Exception as e:
    print(f'ERROR: {e}')
")
    if echo "$CAL_RESP" | grep -q "^OK"; then
        pass "/calibrate endpoint responds correctly ($CAL_RESP)"
    else
        fail "/calibrate returned unexpected response"
        echo "$CAL_RESP"
    fi
else
    skip "/calibrate — test fixture not found: $TEST_FIXTURE"
fi

# ---------------------------------------------------------------------------
# 7. /docs  (Swagger UI)
# ---------------------------------------------------------------------------
echo ""
echo "--- /docs (GET) ---"
DOCS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/docs")
if [ "$DOCS_STATUS" = "200" ]; then
    pass "/docs returns 200 (Swagger UI available)"
else
    fail "/docs returned status $DOCS_STATUS"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Results: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "========================================"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "Container logs:"
    docker logs --tail 30 "$CONTAINER"
    exit 1
fi

echo ""
echo "All tests passed!"
echo ""
echo "To run the API in Docker:"
echo "  docker-compose up -d"
echo ""
echo "Or manually:"
echo "  docker run -d -p 8000:8000 --name dart-api $IMAGE"
