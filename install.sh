#!/bin/bash

set -e

echo "============================================"
echo "  Api Dzeck Ai - Auto Install Script"
echo "============================================"
echo ""

REQUIRED_PACKAGES=(
    "flask[async]:flask:Flask"
    "g4f:g4f:g4f (GPT4Free)"
    "werkzeug>=3.1.4:werkzeug:Werkzeug"
    "aiohttp:aiohttp:aiohttp"
    "aiohttp_socks:aiohttp_socks:aiohttp_socks"
    "curl_cffi:curl_cffi:curl_cffi"
    "python-multipart:multipart:python-multipart"
    "platformdirs:platformdirs:platformdirs"
    "requests:requests:requests"
    "flask-cors:flask_cors:flask-cors"
    "psycopg2-binary:psycopg2:psycopg2-binary"
    "gunicorn:gunicorn:gunicorn"
)

echo "[Step 1/3] Checking for missing dependencies..."
echo ""

MISSING=()
ALL_OK=true

for entry in "${REQUIRED_PACKAGES[@]}"; do
    IFS=':' read -r pip_name import_name display_name <<< "$entry"
    if python3 -c "import $import_name" 2>/dev/null; then
        echo "  [OK] $display_name"
    else
        echo "  [MISSING] $display_name"
        MISSING+=("$pip_name")
        ALL_OK=false
    fi
done

echo ""

if $ALL_OK; then
    echo "[Step 2/3] All dependencies already installed, skipping..."
else
    echo "[Step 2/3] Installing ${#MISSING[@]} missing dependencies..."
    echo ""
    pip install --quiet "${MISSING[@]}" 2>&1 | tail -5

    RETRY_FAIL=()
    for entry in "${REQUIRED_PACKAGES[@]}"; do
        IFS=':' read -r pip_name import_name display_name <<< "$entry"
        if ! python3 -c "import $import_name" 2>/dev/null; then
            RETRY_FAIL+=("$pip_name")
        fi
    done

    if [ ${#RETRY_FAIL[@]} -gt 0 ]; then
        echo ""
        echo "  Retrying failed packages individually..."
        for pkg in "${RETRY_FAIL[@]}"; do
            echo "  Installing $pkg..."
            pip install --quiet "$pkg" 2>&1 | tail -2
        done
    fi
fi

echo ""
echo "[Step 3/3] Final verification..."
echo ""

VERIFY_PASS=0
VERIFY_FAIL=0

for entry in "${REQUIRED_PACKAGES[@]}"; do
    IFS=':' read -r pip_name import_name display_name <<< "$entry"
    if python3 -c "import $import_name" 2>/dev/null; then
        echo "  [OK] $display_name"
        VERIFY_PASS=$((VERIFY_PASS + 1))
    else
        echo "  [FAIL] $display_name"
        VERIFY_FAIL=$((VERIFY_FAIL + 1))
    fi
done

echo ""
echo "============================================"
echo "  Installation Summary"
echo "============================================"
echo "  Verified: $VERIFY_PASS passed, $VERIFY_FAIL failed"
echo ""

if [ $VERIFY_FAIL -eq 0 ]; then
    echo "  All dependencies installed successfully!"
    echo ""
    echo "  To start the server, run:"
    echo "    python3 src/FreeGPT4_Server.py --port 5000 --password YOUR_PASSWORD"
    echo ""
else
    echo "  Some dependencies failed to install."
    echo "  Try running: pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "============================================"
