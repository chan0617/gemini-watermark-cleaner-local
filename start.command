#!/bin/bash
# Double-clickable launcher. Sets up a local venv on first run, then processes input/.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[setup] 가상환경을 생성합니다..."
  # simple-lama-inpainting requires Pillow<10, which has no Python 3.13 wheels,
  # so prefer 3.12 if available and fall back to whatever python3 resolves to.
  PY_BIN="python3"
  if command -v python3.12 >/dev/null 2>&1; then
    PY_BIN="python3.12"
  fi
  "$PY_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import simple_lama_inpainting" >/dev/null 2>&1; then
  echo "[setup] 의존성을 설치합니다 (최초 1회, 몇 분 걸릴 수 있습니다)..."
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
fi

mkdir -p input output failed models

# python.org's Python build doesn't use the macOS system CA store, which
# breaks the one-time model-weight download over HTTPS. Point it at
# certifi's bundle instead of requiring the user to run
# "Install Certificates.command" by hand.
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

python main.py "$@"
