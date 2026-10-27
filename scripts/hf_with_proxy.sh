#!/usr/bin/env bash
# Run one command through an explicit proxy; preserve TLS verification.
set -euo pipefail
if [[ $# -eq 0 || "$1" == "--help" ]]; then
  printf '%s\n' \
    'Usage: bash scripts/hf_with_proxy.sh PROXY_URL [--check-only | COMMAND ...]' \
    'Example: bash scripts/hf_with_proxy.sh socks5h://127.0.0.1:17890 hf auth login' \
    'No global proxy/DNS settings are modified. Keep the SSH tunnel open.'
  exit 0
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  export PATH="$PROJECT_DIR/.venv/bin:$PATH"
fi
PROXY_URL="$1"
shift
export HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" ALL_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" all_proxy="$PROXY_URL"
export NO_PROXY= no_proxy=
export HF_ENDPOINT=https://huggingface.co HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
python - <<'PY'
import importlib.util
import os
import sys
from urllib.parse import urlsplit
import requests
try:
    proxy = urlsplit(os.environ['HTTPS_PROXY'])
    if proxy.scheme not in {'http', 'https', 'socks5h'} or not proxy.hostname or not proxy.port:
        raise ValueError('Use http://, https:// or socks5h:// with an explicit port')
    if proxy.username or proxy.password:
        raise ValueError('Use a proxy URL without embedded credentials')
    if proxy.scheme == 'socks5h' and importlib.util.find_spec('socks') is None:
        raise RuntimeError('Install PySocks in the project environment: python -m pip install PySocks==1.7.1')
    response = requests.get('https://huggingface.co/api/whoami-v2', timeout=(5, 10), allow_redirects=False)
    if response.status_code not in (200, 401):
        raise RuntimeError(f'Unexpected anonymous HF status: HTTP {response.status_code}')
    print(f'HF HTTPS reachable with TLS verification; anonymous status={response.status_code}.', flush=True)
except Exception as exc:
    print(f'HF network check failed ({type(exc).__name__}): {exc}', file=sys.stderr)
    sys.exit(1)
PY
if [[ "${1:-}" == "--check-only" ]]; then
  exit 0
fi
if [[ $# -eq 0 ]]; then
  set -- hf auth login
fi
exec "$@"
