#!/usr/bin/env bash
set -euo pipefail

# Install vLLM from source for Linux x86_64 hosts that support AVX (AVX1) but
# should avoid AVX512/BF16/VNNI build paths.
#
# Usage:
#   bash scripts/install_vllm_avx1_linux.sh
#
# Optional environment variables:
#   PYTHON_BIN=python3.12
#   VENV_DIR=$HOME/.venvs/vllm-avx1
#   VLLM_SRC_DIR=/path/to/vllm            # default: current repo if detected
#   VLLM_REF=main                         # ignored when using current repo
#   INSTALL_MODE=editable                 # editable | wheel
#   INSTALL_SYSTEM_DEPS=1                 # try apt install (best effort)

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/vllm-avx1}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-}"
VLLM_REF="${VLLM_REF:-main}"
INSTALL_MODE="${INSTALL_MODE:-editable}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
USE_CURRENT_REPO=0

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "required command not found: $1"
  fi
}

validate_inputs() {
  case "$INSTALL_MODE" in
    editable|wheel) ;;
    *) fail "INSTALL_MODE must be 'editable' or 'wheel' (got: $INSTALL_MODE)" ;;
  esac

  case "$INSTALL_SYSTEM_DEPS" in
    0|1) ;;
    *) fail "INSTALL_SYSTEM_DEPS must be 0 or 1 (got: $INSTALL_SYSTEM_DEPS)" ;;
  esac
}

maybe_install_system_deps() {
  if [[ "$INSTALL_SYSTEM_DEPS" != "1" ]]; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found; skip system package installation"
    return
  fi

  local SUDO=""
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      SUDO="sudo"
    else
      log "Not root and sudo is unavailable; skip system package installation"
      return
    fi
  fi

  log "Installing system dependencies via apt"
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends \
    git build-essential gcc-12 g++-12 libnuma-dev libtcmalloc-minimal4 curl ca-certificates
}

check_cpu() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    fail "this script supports Linux only"
  fi
  if [[ "$(uname -m)" != "x86_64" ]]; then
    fail "this script supports x86_64 only"
  fi

  if ! grep -qiE '(^|[[:space:]])avx([[:space:]]|$)' /proc/cpuinfo; then
    fail "CPU does not advertise AVX. vLLM x86 CPU backend requires AVX"
  fi

  if command -v lscpu >/dev/null 2>&1; then
    local flags
    flags="$(lscpu | sed -n 's/^Flags:\s*//p' | tr '[:upper:]' '[:lower:]')"
    if [[ "$flags" == *"avx512f"* ]]; then
      log "Notice: AVX512 is present; the script still forces AVX-only build flags."
    fi
  fi
}

prepare_venv() {
  require_cmd "$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10) or (major, minor) >= (3, 14):
    raise SystemExit(f"Python {major}.{minor} is unsupported for vLLM; need >=3.10,<3.14")
PY
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
}

detect_or_prepare_source() {
  require_cmd git

  if [[ -z "$VLLM_SRC_DIR" && -d "$REPO_ROOT/.git" ]]; then
    VLLM_SRC_DIR="$REPO_ROOT"
    USE_CURRENT_REPO=1
    log "Detected current vLLM repo at $VLLM_SRC_DIR"
    return
  fi

  if [[ -z "$VLLM_SRC_DIR" ]]; then
    VLLM_SRC_DIR="$HOME/src/vllm"
  fi

  mkdir -p "$(dirname "$VLLM_SRC_DIR")"
  if [[ -d "$VLLM_SRC_DIR/.git" ]]; then
    log "Using existing repository at $VLLM_SRC_DIR"
    git -C "$VLLM_SRC_DIR" fetch --tags origin
  else
    log "Cloning vLLM into $VLLM_SRC_DIR"
    git clone https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
  fi

  if [[ "$USE_CURRENT_REPO" == "0" ]]; then
    git -C "$VLLM_SRC_DIR" checkout "$VLLM_REF"
  fi
}

install_python_deps() {
  python -m pip install -v -r "$VLLM_SRC_DIR/requirements/cpu-build.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
  python -m pip install -v -r "$VLLM_SRC_DIR/requirements/cpu.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
}

install_vllm() {
  export VLLM_TARGET_DEVICE=cpu
  export CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON

  # Force AVX baseline path and disable AVX512-family toggles.
  export VLLM_CPU_AVX2=1
  export VLLM_CPU_AVX512=0
  export VLLM_CPU_AVX512BF16=0
  export VLLM_CPU_AVX512VNNI=0
  export VLLM_CPU_AMXBF16=0

  pushd "$VLLM_SRC_DIR" >/dev/null
  if [[ "$INSTALL_MODE" == "wheel" ]]; then
    python -m build --wheel --no-isolation
    python -m pip install dist/*.whl
  else
    python -m pip install -e . --no-build-isolation
  fi
  popd >/dev/null
}

configure_ld_preload_hint() {
  local tc_path iomp_path
  tc_path="$(find /usr/lib /usr/lib64 /lib /lib64 -name 'libtcmalloc_minimal.so*' 2>/dev/null | head -n 1 || true)"
  iomp_path="$(find "$VENV_DIR" /usr/lib /usr/lib64 /lib /lib64 -name 'libiomp5.so*' 2>/dev/null | head -n 1 || true)"

  if [[ -n "$tc_path" && -n "$iomp_path" ]]; then
    log "Recommended runtime setting:"
    echo "export LD_PRELOAD=\"$tc_path:$iomp_path:\$LD_PRELOAD\""
  else
    log "Could not auto-detect both libtcmalloc and libiomp5; configure LD_PRELOAD manually if needed."
  fi
}

main() {
  validate_inputs
  check_cpu
  maybe_install_system_deps
  prepare_venv
  detect_or_prepare_source
  install_python_deps
  install_vllm
  configure_ld_preload_hint

  log "Done. Activate with: source $VENV_DIR/bin/activate"
  log "Quick check: python -c 'import vllm; print(vllm.__version__)'"
}

main "$@"
