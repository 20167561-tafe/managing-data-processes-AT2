#!/usr/bin/env bash
#
# run_distributed.sh - fetch train.py from a (private) GitHub repo and run it,
# with GPU detection + CPU fallback, validated pass-through arguments,
# timestamped logs under runs/, and troubleshooting guidance for common errors.
#
#   ./run_distributed.sh --help          full documentation
#   ./run_distributed.sh --dry-run       do every check, print the command, don't train
#
set -Eeuo pipefail

readonly PROG="run_distributed.sh"
readonly VERSION="1.0.0"
readonly TOTAL_STEPS=5

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2) )); then
  echo "[error] bash >= 4.2 is required (found ${BASH_VERSION})." >&2
  exit 65
fi

# Exit codes (train.py's own exit code is propagated when training itself fails)
readonly E_USAGE=64 E_PREREQ=65 E_FETCH=66 E_GPU=67 E_SCRIPT=68 E_INTERRUPT=130

# -----------------------------------------------------------------------------
# Wrapper settings (all overridable with flags)
# -----------------------------------------------------------------------------
readonly DEFAULT_REPO_URL="https://github.com/20167561-tafe/managing-data-processes-AT2.git"
REPO_URL="$DEFAULT_REPO_URL"
REF="HEAD"                      # HEAD = the repo's default branch; a branch, tag or commit SHA also works
SCRIPT_REL="train.py"           # path of the script inside the repo
DEVICE="auto"                   # auto | cuda | cpu
GPUS=""                         # e.g. 0,1 (default: all visible GPUs)
LAUNCHER="auto"                 # auto | python | torchrun
RUNS_DIR="runs"
RUN_NAME=""
DRY_RUN=0
REPO_CACHE="${REPO_CACHE:-.repo_cache}"

# Training options: validated here, forwarded to train.py only when given.
TRAIN_ARGS=()
EXTRA_ARGS=()
declare -A GIVEN=()
declare -A DEFAULTS=(
  [seed]=42 [epochs]=4 [batch-size]=4 [lr]=0.001 [momentum]=0.9 [num-workers]=2
  [log-interval]=2000 [data-dir]=/kaggle/working [save-path]=./cifar_net.pth
  [wandb-mode]=online [wandb-project]=cifar10-cnn [deterministic]=on
)

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
usage() {
cat <<EOF
$PROG $VERSION - fetch train.py from GitHub and run it with GPU detection and logging

USAGE
  ./$PROG [wrapper options] [training options] [-- extra args for train.py]

WRAPPER OPTIONS
  --repo-url URL       Git repo to pull from.
                       default: $DEFAULT_REPO_URL
  --ref REF            Branch, tag or commit SHA to run.            default: HEAD (default branch)
  --script PATH        Script path inside the repo (relative).      default: train.py
  --device MODE        auto | cuda | cpu.                           default: auto
                         auto : use the GPU if PyTorch can really use it, otherwise fall back to CPU
                         cuda : require a GPU; stop with an explanation if none is usable
                         cpu  : never use a GPU
  --gpus LIST          Comma-separated GPU indices, e.g. 0 or 0,1.  default: all visible GPUs
  --launcher MODE      auto | python | torchrun.                    default: auto
                         auto uses torchrun only if >1 GPU is selected AND train.py contains
                         distributed code (init_process_group / DistributedDataParallel / LOCAL_RANK);
                         otherwise a single 'python' process is used.
  --runs-dir DIR       Where logs are written.                      default: runs
  --run-name NAME      Label added to the run id / log file name (A-Z a-z 0-9 . _ -)
  --dry-run            Run every check and print the command, but do not start training
  -h, --help           Show this help           --version   Show version

TRAINING OPTIONS (validated, then passed through to train.py when given)
  Flags you omit are not sent, so train.py's own defaults apply (shown below).
  --seed N               integer 0..4294967295                      default: ${DEFAULTS[seed]}
  --deterministic        deterministic algorithms (slower)          default: on
  --no-deterministic     turn deterministic mode off
  --epochs N             integer >= 1                               default: ${DEFAULTS[epochs]}
  --batch-size N         integer >= 1                               default: ${DEFAULTS[batch-size]}
  --lr X                 learning rate, number > 0                  default: ${DEFAULTS[lr]}
  --momentum X           number between 0 and 1                     default: ${DEFAULTS[momentum]}
  --num-workers N        DataLoader workers, integer >= 0           default: ${DEFAULTS[num-workers]}
  --log-interval N       print/log every N mini-batches, >= 1       default: ${DEFAULTS[log-interval]}
  --data-dir DIR         dataset download/read location             default: ${DEFAULTS[data-dir]}
                         (outside Kaggle, where that path doesn't exist, ./data is used instead)
  --save-path FILE       where the trained weights are saved        default: ${DEFAULTS[save-path]}
  --wandb-mode MODE      online | offline | disabled                default: ${DEFAULTS[wandb-mode]}
  --wandb-project NAME   Weights & Biases project                   default: ${DEFAULTS[wandb-project]}
  --wandb-run-name NAME  W&B run name                               default: the run id of this wrapper

PASS-THROUGH
  Anything after a lone '--' is forwarded to train.py untouched and NOT validated:
      ./$PROG --epochs 10 -- --some-new-flag 1

ENVIRONMENT
  GITHUB_TOKEN (or GH_TOKEN)  Token with read access to the private repo (fine-grained token with
                              "Contents: Read-only" on that one repo is enough). Never put it on the
                              command line or in the notebook source. It is used only for the fetch,
                              never written to disk or the log, and removed before train.py starts.
  WANDB_API_KEY               Weights & Biases key (train.py can also read a Kaggle secret).
  PYTHON                      Python interpreter to use (default: python3, then python).
  REPO_CACHE                  Where the repo checkout is kept (default: .repo_cache).

OUTPUT
  runs/<YYYYmmdd-HHMMSS>[_name].log    full timestamped log (also shown live); runs/latest.log -> newest

EXIT CODES
  0 success | $E_USAGE bad arguments | $E_PREREQ missing tool/package | $E_FETCH could not fetch the repo
  $E_GPU GPU required but unusable | $E_SCRIPT train.py missing/invalid | 130 interrupted
  otherwise: train.py's own exit code

EXAMPLES
  ./$PROG --epochs 10 --seed 7 --batch-size 64 --run-name bs64
  ./$PROG --device cpu --wandb-mode disabled
  ./$PROG --gpus 0,1 --launcher torchrun --epochs 20
  ./$PROG --ref 3f2a9c1 --dry-run

KAGGLE NOTEBOOK (pass the secret via the environment, never typed into a cell)
  import os
  from kaggle_secrets import UserSecretsClient
  os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
  !bash run_distributed.sh --epochs 10 --wandb-mode offline
EOF
}

# -----------------------------------------------------------------------------
# Argument parsing + validation (runs before the log exists; errors go to stderr)
# -----------------------------------------------------------------------------
usage_error() {
  printf '[error] %s\n' "$1" >&2
  printf '        Run ./%s --help for the option list.\n' "$PROG" >&2
  exit "$E_USAGE"
}

check_int() {    # opt value min max
  local o=$1 v=$2 min=$3 max=$4
  if [[ ! $v =~ ^[0-9]{1,18}$ ]] || (( 10#$v < min || 10#$v > max )); then
    usage_error "invalid value for $o: '$v' (expected an integer from $min to $max${DEFAULTS[${o#--}]:+; default ${DEFAULTS[${o#--}]}})"
  fi
}
FLOAT_RE='^([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$'
check_float_pos() {   # opt value
  if [[ ! $2 =~ $FLOAT_RE ]] || ! awk -v v="$2" 'BEGIN{exit !(v+0 > 0)}'; then
    usage_error "invalid value for $1: '$2' (expected a number > 0; default ${DEFAULTS[${1#--}]})"
  fi
  if awk -v v="$2" 'BEGIN{exit !(v+0 > 1)}'; then
    printf '[warn]  %s %s is unusually large; training will probably diverge.\n' "$1" "$2" >&2
  fi
}
check_float_01() {    # opt value
  if [[ ! $2 =~ $FLOAT_RE ]] || ! awk -v v="$2" 'BEGIN{exit !(v+0 >= 0 && v+0 <= 1)}'; then
    usage_error "invalid value for $1: '$2' (expected a number from 0 to 1; default ${DEFAULTS[${1#--}]})"
  fi
}
check_choice() {      # opt value choice...
  local o=$1 v=$2; shift 2
  local c
  for c in "$@"; do [[ $v == "$c" ]] && return 0; done
  usage_error "invalid value for $o: '$v' (choose one of: $*)"
}
check_nonempty() { [[ -n $2 ]] || usage_error "$1 needs a non-empty value"; }

add_train() {         # flag [value]
  local flag=$1 key=${1#--}
  [[ $key == no-deterministic ]] && key=deterministic
  if [[ -n ${GIVEN[$key]+x} ]]; then
    usage_error "$flag was given more than once, or conflicts with another flag for the same setting; give it once"
  fi
  TRAIN_ARGS+=("$flag")
  if (($# > 1)); then TRAIN_ARGS+=("$2"); GIVEN[$key]=$2
  elif [[ $flag == --no-deterministic ]]; then GIVEN[$key]=0
  else GIVEN[$key]=1; fi
}

VALUE_OPTS=" --repo-url --ref --script --device --gpus --launcher --runs-dir --run-name --seed --epochs --batch-size --lr --momentum --num-workers --log-interval --data-dir --save-path --wandb-mode --wandb-project --wandb-run-name "

parse_args() {
  local opt val inline
  while (($#)); do
    opt=$1; shift; val=""; inline=0
    if [[ $opt == --*=* ]]; then val=${opt#*=}; opt=${opt%%=*}; inline=1; fi
    if [[ $VALUE_OPTS == *" $opt "* ]]; then
      if ((! inline)); then
        (($#)) || usage_error "option $opt requires a value"
        val=$1; shift
      fi
    elif ((inline)); then
      usage_error "option $opt does not take a value"
    fi
    case $opt in
      -h|--help)  usage; exit 0 ;;
      --version)  echo "$PROG $VERSION"; exit 0 ;;
      --dry-run)  DRY_RUN=1 ;;
      --repo-url) check_nonempty "$opt" "$val"; REPO_URL=$val ;;
      --ref)      check_nonempty "$opt" "$val"; REF=$val ;;
      --script)   check_nonempty "$opt" "$val"; SCRIPT_REL=$val ;;
      --device)   check_choice "$opt" "$val" auto cuda cpu; DEVICE=$val ;;
      --gpus)     [[ $val =~ ^[0-9]+(,[0-9]+)*$ ]] || usage_error "invalid value for --gpus: '$val' (expected e.g. 0 or 0,1)"
                  GPUS=$val ;;
      --launcher) check_choice "$opt" "$val" auto python torchrun; LAUNCHER=$val ;;
      --runs-dir) check_nonempty "$opt" "$val"; RUNS_DIR=$val ;;
      --run-name) [[ $val =~ ^[A-Za-z0-9._-]{1,64}$ ]] || usage_error "invalid --run-name '$val' (1-64 chars: letters, digits, . _ -)"
                  RUN_NAME=$val ;;
      --seed)          check_int "$opt" "$val" 0 4294967295; add_train "$opt" "$val" ;;
      --epochs)        check_int "$opt" "$val" 1 100000;     add_train "$opt" "$val" ;;
      --batch-size)    check_int "$opt" "$val" 1 10000000;   add_train "$opt" "$val" ;;
      --num-workers)   check_int "$opt" "$val" 0 256;        add_train "$opt" "$val" ;;
      --log-interval)  check_int "$opt" "$val" 1 1000000000; add_train "$opt" "$val" ;;
      --lr)            check_float_pos "$opt" "$val";        add_train "$opt" "$val" ;;
      --momentum)      check_float_01 "$opt" "$val";         add_train "$opt" "$val" ;;
      --data-dir|--save-path|--wandb-project|--wandb-run-name)
                       check_nonempty "$opt" "$val";         add_train "$opt" "$val" ;;
      --wandb-mode)    check_choice "$opt" "$val" online offline disabled; add_train "$opt" "$val" ;;
      --deterministic|--no-deterministic) add_train "$opt" ;;
      --)  EXTRA_ARGS=("$@"); break ;;
      -*)  usage_error "unknown option '$opt'. If train.py supports it, pass it after '--' (e.g. -- $opt VALUE)." ;;
      *)   usage_error "unexpected argument '$opt'. Options start with '--'; extra train.py arguments go after '--'." ;;
    esac
  done

  # cross-checks
  if [[ $SCRIPT_REL == /* || $SCRIPT_REL == *..* ]]; then
    usage_error "--script must be a relative path inside the repo (no leading '/' or '..'): '$SCRIPT_REL'"
  fi
  if [[ $SCRIPT_REL == *.ipynb ]]; then
    usage_error "--script points at a notebook ('$SCRIPT_REL'); python cannot run .ipynb files. Export it first: jupyter nbconvert --to script"
  fi
  if [[ -n $GPUS && $DEVICE == cpu ]]; then
    printf '[warn]  --gpus %s ignored because --device cpu was given.\n' "$GPUS" >&2
  fi
}

parse_args "$@"

# -----------------------------------------------------------------------------
# Run identity, log file, shared state (all created before the logging pipeline)
# -----------------------------------------------------------------------------
RUN_TS=$(date +%Y%m%d-%H%M%S)
RUN_ID="${RUN_TS}${RUN_NAME:+_$RUN_NAME}"
if ! mkdir -p "$RUNS_DIR" 2>/dev/null || [[ ! -w $RUNS_DIR ]]; then
  printf '[error] cannot write to the runs directory "%s".\n        Choose another with --runs-dir, or fix permissions.\n' "$RUNS_DIR" >&2
  exit "$E_PREREQ"
fi
LOG="$RUNS_DIR/$RUN_ID.log"
: > "$LOG"
ln -sfn "$RUN_ID.log" "$RUNS_DIR/latest.log" 2>/dev/null || true
STATE_DIR=$(mktemp -d)
PROGRESS_FILE="$STATE_DIR/progress"
RC_FILE="$STATE_DIR/train_rc"
START_MARK="$STATE_DIR/start_mark"
touch "$START_MARK"
trap 'rm -rf "$STATE_DIR"' EXIT

# -----------------------------------------------------------------------------
# Output helpers. Everything the pipeline prints is timestamped by stream_filter.
# Lines starting with \001 (raw) are printed without a timestamp (headers/banners).
# -----------------------------------------------------------------------------
say()  { printf '%s\n' "$*"; }
raw()  { printf '\001%s\n' "$*"; }
info() { say "[info]  $*"; }
ok()   { say "[ ok ]  $*"; }
warn() { say "[warn]  $*"; }
err()  { say "[error] $*"; }
hint() { say "        -> $*"; }
step() { raw ""; raw "[$1/$TOTAL_STEPS] $2"; }

die() {   # die CODE "message" "hint"...
  local code=$1 msg=$2; shift 2
  err "$msg"
  local h; for h in "$@"; do if [[ -n $h ]]; then hint "$h"; fi; done
  exit "$code"
}

fmt_dur() { local s=$1; printf '%02d:%02d:%02d' $((s/3600)) $((s%3600/60)) $((s%60)); }

stream_filter() {
  local line total="" t0="" epoch pct now el eta
  while IFS= read -r line || [[ -n $line ]]; do
    if [[ $line == $'\001'* ]]; then printf '%s\n' "${line#?}"; continue; fi
    printf '%(%F %T)T %s\n' -1 "$line"
    # progress indicator: train.py prints "--- Epoch N: ..." after each epoch
    if [[ $line =~ ---[[:space:]]Epoch[[:space:]]([0-9]+): ]]; then
      epoch=${BASH_REMATCH[1]}
      total=""; t0=""
      if [[ -r $PROGRESS_FILE ]]; then { read -r total; read -r t0; } < "$PROGRESS_FILE" || true; fi
      now=$(printf '%(%s)T' -1)
      el=$(( now - ${t0:-$now} ))
      if [[ $total =~ ^[0-9]+$ && $total -gt 0 ]]; then
        pct=$(( 100 * epoch / total ))
        eta=$(( epoch > 0 ? el * (total - epoch) / epoch : 0 ))
        printf '%(%F %T)T [prog]  epoch %s/%s done (%s%%) | elapsed %s | ETA ~%s\n' -1 \
          "$epoch" "$total" "$pct" "$(fmt_dur "$el")" "$(fmt_dur "$eta")"
      else
        printf '%(%F %T)T [prog]  epoch %s done | elapsed %s\n' -1 "$epoch" "$(fmt_dur "$el")"
      fi
    fi
  done
}

# -----------------------------------------------------------------------------
# Stage helpers
# -----------------------------------------------------------------------------
find_python() {
  local c
  for c in "${PYTHON:-}" python3 python; do
    [[ -n $c ]] && command -v "$c" >/dev/null 2>&1 && { PYTHON=$(command -v "$c"); return 0; }
  done
  return 1
}

redact() {   # hide the token if it ever shows up in tool output
  local t="${GITHUB_TOKEN:-${GH_TOKEN:-}}" s=$1
  [[ -n $t ]] && s=${s//"$t"/***}
  printf '%s' "$s"
}

classify_fetch_error() {
  local out=$1
  if   grep -Eqi 'Authentication failed|could not read (Username|Password)|Invalid username|Repository not found|error: 40[134]|Permission denied|access denied|HTTP Basic' <<<"$out"; then echo auth
  elif grep -Eqi "couldn't find remote ref|not our ref|Remote branch .* not found|invalid refspec" <<<"$out"; then echo ref
  elif grep -Eqi 'Could not resolve host|Temporary failure in name resolution|Failed to connect|timed out|Network is unreachable|Could not connect|Connection (refused|reset)|SSL' <<<"$out"; then echo network
  else echo other; fi
}

fetch_repo() {
  local token="${GITHUB_TOKEN:-${GH_TOKEN:-}}" out rc=0 askpass="$STATE_DIR/askpass.sh"
  mkdir -p "$REPO_CACHE"
  [[ -d $REPO_CACHE/.git ]] || git init -q "$REPO_CACHE"
  if git -C "$REPO_CACHE" remote get-url origin >/dev/null 2>&1; then
    git -C "$REPO_CACHE" remote set-url origin "$REPO_URL"
  else
    git -C "$REPO_CACHE" remote add origin "$REPO_URL"
  fi

  local -a gitenv=(GIT_TERMINAL_PROMPT=0) gitopts=(-c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30)
  if [[ -n $token ]]; then
    # Token is handed to git through an askpass helper + environment, so it never appears in a
    # command line, in .git/config, in the remote URL, or in this log.
    printf '#!/bin/sh\ncase "$1" in\n  Username*) printf "%%s\\n" "x-access-token" ;;\n  *) printf "%%s\\n" "$_RD_GIT_TOKEN" ;;\nesac\n' > "$askpass"
    chmod 700 "$askpass"
    gitenv+=(GIT_ASKPASS="$askpass" _RD_GIT_TOKEN="$token")
    gitopts+=(-c credential.helper=)
    info "Using GitHub token from the environment (value hidden)."
  else
    warn "No GITHUB_TOKEN/GH_TOKEN set; trying without credentials (works for public repos or a configured git credential helper)."
  fi

  out=$(env "${gitenv[@]}" git "${gitopts[@]}" -C "$REPO_CACHE" fetch --depth 1 --quiet origin "$REF" 2>&1) || rc=$?
  out=$(redact "$out")

  if ((rc != 0)); then
    local kind; kind=$(classify_fetch_error "$out")
    if [[ $kind == network && -f $REPO_CACHE/$SCRIPT_REL ]] && git -C "$REPO_CACHE" rev-parse HEAD >/dev/null 2>&1; then
      warn "Could not reach GitHub; using the CACHED copy at commit $(git -C "$REPO_CACHE" rev-parse --short HEAD). It may be out of date."
      hint "Reason: $(head -n1 <<<"$out")"
      return 0
    fi
    case $kind in
      network) die "$E_FETCH" "Cannot reach GitHub (network/DNS failure)." \
                 "Kaggle: Session options -> Internet -> On, then restart the session." \
                 "Elsewhere: check your connection/proxy: 'curl -I https://github.com'" \
                 "Details: $(head -n2 <<<"$out" | tr '\n' ' ')" ;;
      auth)    die "$E_FETCH" "GitHub refused access to $REPO_URL." \
                 "The repo is private: set GITHUB_TOKEN to a token that can read it." \
                 "Create one at github.com -> Settings -> Developer settings -> Fine-grained tokens," \
                 "  repository access: this repo only, permission 'Contents: Read-only'." \
                 "Kaggle: add the token under Add-ons -> Secrets, attach it to the notebook, then" \
                 "  os.environ['GITHUB_TOKEN'] = UserSecretsClient().get_secret('GITHUB_TOKEN') before running this script." \
                 "A 404/'Repository not found' also means the token cannot see the repo (wrong owner, or missing permission)." ;;
      ref)     die "$E_FETCH" "Ref '$REF' was not found in the repository." \
                 "Check the branch/tag/commit name; omit --ref to use the default branch." ;;
      *)       die "$E_FETCH" "git fetch failed." "Output: $(head -n5 <<<"$out" | tr '\n' ' ')" \
                 "Check --repo-url ($REPO_URL) and --ref ($REF)." ;;
    esac
  fi
  git -C "$REPO_CACHE" -c advice.detachedHead=false checkout -q --force FETCH_HEAD
}

# Probe PyTorch in a child process. Sets PROBE_* from key=value lines (never eval'd).
PROBE_TORCH="" PROBE_TV="" PROBE_CUDA_VER="" PROBE_AVAIL=0 PROBE_COUNT=0 PROBE_NAMES="" PROBE_SMOKE="skipped" PROBE_ERR="" PROBE_MISSING=""
probe_torch() {
  local out k v
  out=$(PROBE_SMOKE_REQUEST="$1" "$PYTHON" - <<'PY' 2>&1 || true
import os, importlib.util
r = {"torch": "", "tv": "", "cuda_ver": "", "avail": "0", "count": "0", "names": "", "smoke": "skipped", "err": "", "missing": ""}
mods = ["numpy", "pandas", "pyarrow", "PIL", "huggingface_hub", "matplotlib"]
if os.environ.get("PROBE_NEED_WANDB") == "1":
    mods.append("wandb")
r["missing"] = ",".join(m for m in mods if importlib.util.find_spec(m) is None)
try:
    import torch
except Exception as e:
    r["err"] = "import torch failed: %s: %s" % (type(e).__name__, e)
else:
    r["torch"] = torch.__version__
    r["cuda_ver"] = str(torch.version.cuda or "")
    try:
        import torchvision
        r["tv"] = torchvision.__version__
    except Exception as e:
        r["err"] = "import torchvision failed: %s: %s" % (type(e).__name__, e)
    try:
        r["avail"] = "1" if torch.cuda.is_available() else "0"
        r["count"] = str(torch.cuda.device_count())
        if r["avail"] == "1":
            names = []
            for i in range(int(r["count"])):
                p = torch.cuda.get_device_properties(i)
                cap = torch.cuda.get_device_capability(i)
                names.append("%s (%d MiB, sm_%d%d)" % (p.name, p.total_memory // 2**20, cap[0], cap[1]))
            r["names"] = "|".join(names)
            if os.environ.get("PROBE_SMOKE_REQUEST") == "1":
                try:
                    x = torch.ones(16, device="cuda")
                    float((x * 2).sum().item())
                    r["smoke"] = "ok"
                except Exception as e:
                    r["smoke"] = "failed"
                    r["err"] = "%s: %s" % (type(e).__name__, e)
    except Exception as e:
        r["err"] = "%s: %s" % (type(e).__name__, e)
for k, v in r.items():
    print("PROBE_%s=%s" % (k, str(v).replace("\n", " ")[:300]))
PY
)
  while IFS='=' read -r k v; do
    case $k in
      PROBE_torch) PROBE_TORCH=$v ;;       PROBE_tv) PROBE_TV=$v ;;
      PROBE_cuda_ver) PROBE_CUDA_VER=$v ;; PROBE_avail) PROBE_AVAIL=$v ;;
      PROBE_count) PROBE_COUNT=$v ;;       PROBE_names) PROBE_NAMES=$v ;;
      PROBE_smoke) PROBE_SMOKE=$v ;;       PROBE_err) PROBE_ERR=$v ;;
      PROBE_missing) PROBE_MISSING=$v ;;
    esac
  done <<<"$out"
  if [[ -z $PROBE_TORCH && -z $PROBE_ERR ]]; then
    PROBE_ERR="probe produced no output: $(head -n3 <<<"$out" | tr '\n' ' ')"
  fi
}

NV_COUNT=0 NV_SUMMARY=()
detect_nvidia_smi() {
  local out rc=0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    info "nvidia-smi not found (no NVIDIA driver in this environment)."
    return 0
  fi
  out=$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1) || rc=$?
  if ((rc != 0)); then
    warn "nvidia-smi is installed but failed: $(head -n1 <<<"$out")"
    hint "The NVIDIA driver is not loaded or does not match the CUDA libraries. Restart the machine/session."
    return 0
  fi
  mapfile -t NV_SUMMARY <<<"$out"
  NV_COUNT=${#NV_SUMMARY[@]}
  ok "nvidia-smi sees $NV_COUNT GPU(s):"
  local l; for l in "${NV_SUMMARY[@]}"; do say "          $l"; done
}

explain_gpu_problem() {   # why PyTorch can't use a GPU (an error only if --device cuda was demanded)
  local lvl=warn; [[ $DEVICE == cuda ]] && lvl=err
  if [[ $PROBE_SMOKE == failed ]]; then
    "$lvl" "PyTorch found a GPU but a test computation on it failed: $PROBE_ERR"
    if grep -Eqi 'no kernel image|not compatible|sm_[0-9]+' <<<"$PROBE_ERR"; then
      hint "This PyTorch build does not include kernels for your GPU's architecture (typical: Kaggle P100 = sm_60"
      hint "  with a recent PyTorch). Switch the accelerator to a T4, or install a PyTorch build that supports it."
    else
      hint "Driver/CUDA mismatch is the usual cause. Compare 'nvidia-smi' (driver) with torch.version.cuda=$PROBE_CUDA_VER."
    fi
  elif ((NV_COUNT > 0)) && [[ -z $PROBE_CUDA_VER ]]; then
    "$lvl" "A GPU is present but the installed PyTorch ($PROBE_TORCH) is a CPU-only build."
    hint "Install a CUDA build, e.g.: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    hint "(pick the CUDA version matching your driver at pytorch.org/get-started), then restart the session."
  elif ((NV_COUNT > 0)); then
    "$lvl" "nvidia-smi sees a GPU but torch.cuda.is_available() is False (torch $PROBE_TORCH, CUDA $PROBE_CUDA_VER)."
    hint "Driver too old for this CUDA build, or CUDA_VISIBLE_DEVICES hides the GPU (currently: '${CUDA_VISIBLE_DEVICES-<unset>}')."
  else
    "$lvl" "No usable NVIDIA GPU was found."
    hint "Kaggle: Session options -> Accelerator -> choose a GPU (T4/P100), then restart the session."
    hint "Local machine: install the NVIDIA driver and verify with 'nvidia-smi'."
  fi
}

# Quick, explicit check that train.py will not choke on notebook-only syntax.
check_script_syntax() {
  local out rc=0
  out=$("$PYTHON" - "$1" <<'PY' 2>&1
import sys
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
try:
    compile(src, path, "exec")
except SyntaxError as e:
    line = (e.text or "").strip()
    print("line %s: %s | %s" % (e.lineno, e.msg, line))
    sys.exit(1)
PY
) || rc=$?
  if ((rc != 0)); then
    err "$SCRIPT_REL has a syntax error: $out"
    if grep -Eq '\| *[!%]' <<<"$out"; then
      hint "That line is notebook-only syntax ('!pip ...' or a % magic). The repo file looks like an exported/"
      hint "  Jupyter notebook, not a plain script. Remove those lines (install packages beforehand) or export with:"
      hint "  jupyter nbconvert --to script <notebook>.ipynb   (then delete any get_ipython() lines)"
    fi
    exit "$E_SCRIPT"
  fi
}

# -----------------------------------------------------------------------------
# Main (runs inside the logging pipeline)
# -----------------------------------------------------------------------------
main() {
  set -Eeuo pipefail
  trap 'err "Unexpected failure at line $LINENO: $BASH_COMMAND"' ERR

  raw "=============================================================="
  raw " $PROG $VERSION   run: $RUN_ID"
  raw " started $(date '+%F %T %Z') on $(hostname 2>/dev/null || echo unknown)"
  raw " log: $LOG"
  raw "=============================================================="

  # ---- 1. prerequisites --------------------------------------------------
  step 1 "Checking prerequisites"
  command -v git >/dev/null 2>&1 || die "$E_PREREQ" "git is not installed." \
      "Install it (Debian/Ubuntu: apt-get install git) and re-run."
  ok "git $(git --version | awk '{print $3}')"
  find_python || die "$E_PREREQ" "No python interpreter found (tried \$PYTHON, python3, python)." \
      "Install Python 3.9+ or set PYTHON=/path/to/python."
  local pyver; pyver=$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || die "$E_PREREQ" \
      "Python $pyver is too old (3.9+ required; train.py uses argparse.BooleanOptionalAction)." \
      "Use a newer interpreter: PYTHON=/path/to/python3.11 ./$PROG ..."
  ok "python $pyver ($PYTHON)"

  # data dir default: train.py defaults to /kaggle/working, which only exists on Kaggle
  local data_dir=${GIVEN[data-dir]:-${DEFAULTS[data-dir]}}
  if [[ -z ${GIVEN[data-dir]+x} && ! -d /kaggle/working ]]; then
    data_dir="$PWD/data"
    DEFAULT_DATA_DIR_ARG=("--data-dir" "$data_dir")
    info "/kaggle/working not found: using $data_dir for the dataset (override with --data-dir)."
  fi
  mkdir -p "$data_dir" 2>/dev/null && [[ -w $data_dir ]] || die "$E_PREREQ" \
      "Cannot write to the data directory '$data_dir'." "Pick a writable one with --data-dir."
  local save_path=${GIVEN[save-path]:-${DEFAULTS[save-path]}}
  local save_parent; save_parent=$(dirname "$save_path")
  mkdir -p "$save_parent" 2>/dev/null && [[ -w $save_parent ]] || die "$E_PREREQ" \
      "Cannot write the model to '$save_path'." "Pick a writable location with --save-path."
  ok "writable: data dir ($data_dir), model path ($save_path), runs dir ($RUNS_DIR)"
  local free_kb; free_kb=$(df -Pk "$data_dir" 2>/dev/null | awk 'NR==2{print $4}')
  if [[ $free_kb =~ ^[0-9]+$ ]] && ((free_kb < 2*1024*1024)); then
    warn "Only $((free_kb/1024)) MiB free on the data volume; the CIFAR-10 download needs roughly 0.5 GiB."
  fi

  # ---- 2. fetch ----------------------------------------------------------
  step 2 "Fetching $SCRIPT_REL from $(sed -E 's#(https?://)[^/@]*@#\1#' <<<"$REPO_URL") (ref: $REF)"
  fetch_repo
  local commit subject
  commit=$(git -C "$REPO_CACHE" rev-parse --short HEAD)
  subject=$(git -C "$REPO_CACHE" log -1 --format=%s 2>/dev/null || true)
  ok "checked out $commit${subject:+ - $subject}"
  unset GITHUB_TOKEN GH_TOKEN     # not needed any more; keep it away from train.py and its subprocesses

  local script_path="$REPO_CACHE/$SCRIPT_REL"
  if [[ ! -f $script_path ]]; then
    local listing; listing=$(git -C "$REPO_CACHE" ls-tree -r --name-only HEAD | head -n 15 | tr '\n' ' ' || true)
    local nb; nb=$(git -C "$REPO_CACHE" ls-tree -r --name-only HEAD | grep -E '\.ipynb$' | head -n3 | tr '\n' ' ' || true)
    die "$E_SCRIPT" "'$SCRIPT_REL' does not exist in the repo at commit $commit." \
        "Files in the repo: ${listing:-<empty>}" \
        "${nb:+The repo contains notebook(s) ($nb) but python cannot run .ipynb files: commit a plain train.py,}" \
        "${nb:+  or export one with 'jupyter nbconvert --to script' and use --script to point at it.}" \
        "Use --script PATH if train.py lives in a subfolder."
  fi
  ok "found $SCRIPT_REL ($(wc -l < "$script_path") lines)"

  # ---- 3. hardware / PyTorch --------------------------------------------
  step 3 "Detecting hardware and PyTorch"
  local cores; cores=$(nproc 2>/dev/null || echo 1)
  info "CPU cores: $cores"
  if [[ -n ${GIVEN[num-workers]+x} ]] && ((GIVEN[num-workers] > cores)); then
    warn "--num-workers ${GIVEN[num-workers]} is more than the $cores CPU core(s); expect no speed-up (try $cores or fewer)."
  fi
  detect_nvidia_smi
  local want_smoke=1; [[ $DEVICE == cpu ]] && want_smoke=0
  PROBE_NEED_WANDB=$([[ ${GIVEN[wandb-mode]:-online} == disabled ]] && echo 0 || echo 1) probe_torch "$want_smoke"
  if [[ -z $PROBE_TORCH || ( -n $PROBE_ERR && $PROBE_ERR == import\ * ) ]]; then
    die "$E_PREREQ" "PyTorch could not be imported: ${PROBE_ERR:-unknown error}" \
        "Not installed: pip install torch torchvision" \
        "\"module 'torch' has no attribute '_utils'\" or similar: restart the session, make sure no file/folder named torch.py/torch/ is in $PWD," \
        "  and reinstall matching versions together: pip install --upgrade --force-reinstall torch torchvision"
  fi
  ok "torch $PROBE_TORCH, torchvision ${PROBE_TV:-?}, built for CUDA ${PROBE_CUDA_VER:-none}"
  if [[ -n $PROBE_MISSING ]]; then
    warn "Python packages not installed: ${PROBE_MISSING//,/ }. train.py will fail when it imports them."
    hint "Install: pip install ${PROBE_MISSING//,/ }"
  fi

  USE_DEVICE=cpu; NPROC=1; SELECTED=""
  if [[ $DEVICE == cpu ]]; then
    info "--device cpu: GPUs will not be used."
  elif [[ $PROBE_AVAIL == 1 && $PROBE_SMOKE == ok ]]; then
    ok "PyTorch can use $PROBE_COUNT GPU(s) (test computation passed):"
    local -a n; IFS='|' read -ra n <<<"$PROBE_NAMES"
    local i; for i in "${!n[@]}"; do say "          cuda:$i  ${n[$i]}"; done
    if [[ -n $GPUS ]]; then
      local -a g; IFS=',' read -ra g <<<"$GPUS"
      local seen=" " idx
      for idx in "${g[@]}"; do
        if ((10#$idx >= PROBE_COUNT)); then
          die "$E_GPU" "--gpus $GPUS asks for GPU $idx but only indices 0..$((PROBE_COUNT-1)) are available." \
              "Indices refer to the GPUs visible to PyTorch (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-<unset>}')."
        fi
        [[ $seen == *" $((10#$idx)) "* ]] && die "$E_GPU" "--gpus $GPUS lists GPU $idx twice."
        seen+="$((10#$idx)) "
      done
      SELECTED=$GPUS
    else
      SELECTED=$(seq -s, 0 $((PROBE_COUNT-1)))
    fi
    USE_DEVICE=cuda
    NPROC=$(awk -F, '{print NF}' <<<"$SELECTED")
  else
    explain_gpu_problem
    if [[ $DEVICE == cuda ]]; then
      die "$E_GPU" "--device cuda was requested but no usable GPU is available." \
          "Fix the problem above, or use --device auto to fall back to CPU, or --device cpu."
    fi
    warn "Falling back to CPU. Training will be much slower."
  fi
  info "Device: $USE_DEVICE${SELECTED:+ (GPUs $SELECTED)}"

  # ---- 4. validate + assemble the command -------------------------------
  step 4 "Validating $SCRIPT_REL and arguments"
  check_script_syntax "$script_path"
  ok "syntax OK"

  local k flag
  for k in "${!GIVEN[@]}"; do
    flag="--$k"
    grep -qF -- "$flag" "$script_path" || warn "$SCRIPT_REL does not mention $flag - it may reject it. (Check spelling, or pass it after '--'.)"
  done

  local -a final_args=("${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}")
  if [[ -n ${DEFAULT_DATA_DIR_ARG+x} ]] && grep -qF -- '--data-dir' "$script_path"; then
    final_args+=("${DEFAULT_DATA_DIR_ARG[@]}")
  fi
  if [[ -z ${GIVEN[wandb-run-name]+x} ]] && grep -qF -- '--wandb-run-name' "$script_path"; then
    final_args+=(--wandb-run-name "$RUN_ID")
  fi
  final_args+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

  # W&B hint (train.py handles Kaggle secrets / offline fallback itself)
  if [[ ${GIVEN[wandb-mode]:-online} == online && -z ${WANDB_API_KEY:-} ]]; then
    info "WANDB_API_KEY is not set in the environment; train.py will look for a Kaggle secret, else W&B falls back to offline."
  fi

  # does train.py actually use a GPU / support multiple processes? (static heuristics)
  local has_dist=0
  grep -Eq 'init_process_group|DistributedDataParallel|LOCAL_RANK' "$script_path" && has_dist=1
  if [[ $USE_DEVICE == cuda ]] && ! grep -Eq '\.to\(|\.cuda\(' "$script_path"; then
    warn "$SCRIPT_REL never moves the model/data to a device (.to(...)/.cuda()), so it will run on CPU despite the GPU."
    hint "The wrapper exports RUN_DEVICE=cuda and CUDA_VISIBLE_DEVICES for scripts that want to use them."
  fi

  local launcher=python
  case $LAUNCHER in
    python) launcher=python ;;
    torchrun)
      launcher=torchrun
      ((has_dist)) || warn "--launcher torchrun but $SCRIPT_REL has no distributed code: $NPROC identical copies would each train on the full data."
      ;;
    auto)
      if ((NPROC > 1)); then
        if ((has_dist)); then launcher=torchrun
        else
          warn "$NPROC GPUs selected but $SCRIPT_REL has no distributed code (DistributedDataParallel / LOCAL_RANK);"
          hint "running ONE process. It can use at most one GPU unless the script itself handles several."
          NPROC=1
        fi
      fi ;;
  esac
  if [[ $launcher == torchrun && $USE_DEVICE != cuda ]]; then
    warn "torchrun without GPUs: falling back to a single python process."
    launcher=python; NPROC=1
  fi

  local -a cmd
  if [[ $launcher == torchrun ]]; then
    cmd=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$script_path")
  else
    cmd=("$PYTHON" -u "$script_path")
  fi
  cmd+=("${final_args[@]+"${final_args[@]}"}")

  # environment for train.py
  export PYTHONUNBUFFERED=1 MPLBACKEND=Agg HF_HUB_DISABLE_PROGRESS_BARS=1 RUN_ID RUN_DEVICE=$USE_DEVICE
  if [[ $USE_DEVICE == cuda ]]; then
    export CUDA_VISIBLE_DEVICES=$SELECTED
    [[ ${GIVEN[deterministic]:-1} == 0 ]] || export CUBLAS_WORKSPACE_CONFIG=:4096:8
  else
    export CUDA_VISIBLE_DEVICES=""
  fi
  [[ -n ${GIVEN[seed]+x} ]] && export PYTHONHASHSEED=${GIVEN[seed]}

  # total epochs for the progress indicator
  local epochs_total=${GIVEN[epochs]:-}
  if [[ -z $epochs_total ]]; then
    epochs_total=$(grep -E '"--epochs"' "$script_path" | grep -Eo 'default *= *[0-9]+' | grep -Eo '[0-9]+' | head -n1 || true)
  fi

  # dataset reachability (train.py downloads it from Hugging Face if it is not already on disk)
  if [[ -n $(find "$data_dir" -name 'train-*.parquet' -print -quit 2>/dev/null) ]]; then
    ok "dataset already present under $data_dir"
  elif ! timeout 15 "$PYTHON" -c 'import socket; socket.gethostbyname("huggingface.co")' 2>/dev/null; then
    warn "huggingface.co is not reachable and no dataset was found under $data_dir: the download will fail."
    hint "Kaggle: Session options -> Internet -> On, or attach a CIFAR-10 dataset and set --data-dir."
  fi

  raw ""
  raw "---------------------------- RUN HEADER ----------------------------"
  raw " repo     : $(sed -E 's#(https?://)[^/@]*@#\1#' <<<"$REPO_URL") @ $commit (ref $REF)"
  raw " script   : $SCRIPT_REL"
  raw " device   : $USE_DEVICE${SELECTED:+ [GPUs $SELECTED]}   launcher: $launcher${NPROC:+ x$NPROC}"
  raw " software : python $pyver, torch $PROBE_TORCH (CUDA ${PROBE_CUDA_VER:-none})"
  raw " args     : ${final_args[*]+$(printf '%q ' "${final_args[@]}")}"
  raw "              (flags not shown use train.py defaults)"
  raw " log      : $LOG"
  raw "--------------------------------------------------------------------"

  if ((DRY_RUN)); then
    raw ""
    info "Dry run: would execute:"
    say "        $(printf '%q ' "${cmd[@]}")"
    ok "All checks passed. Nothing was started (--dry-run)."
    exit 0
  fi

  # ---- 5. train ----------------------------------------------------------
  step 5 "Training (live output below; '[prog]' lines track epochs)"
  printf '%s\n%s\n' "${epochs_total:-?}" "$(date +%s)" > "$PROGRESS_FILE"
  local t_start=$SECONDS rc=0
  "${cmd[@]}" &
  TRAIN_PID=$!
  on_signal() {
    err "Interrupted - stopping training (pid $TRAIN_PID)..."
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
    echo 130 > "$RC_FILE"
    exit "$E_INTERRUPT"
  }
  trap on_signal INT TERM
  wait "$TRAIN_PID" || rc=$?
  trap - INT TERM
  echo "$rc" > "$RC_FILE"
  if ((rc == 0)); then
    ok "train.py finished in $(fmt_dur $((SECONDS - t_start)))"
  else
    err "train.py exited with code $rc after $(fmt_dur $((SECONDS - t_start)))"
  fi
  exit "$rc"
}

# -----------------------------------------------------------------------------
# After the run: explain failures and print the closing report
# -----------------------------------------------------------------------------
diagnose_failure() {   # log rc
  local tail_txt rc=$2 found=0 m
  tail_txt=$(tail -n 400 "$1")
  has() { grep -Eqi -- "$1" <<<"$tail_txt"; }
  say "[diag]  Looking for known problems in the log..."
  if has 'CUDA out of memory|OutOfMemoryError'; then found=1
    hint "GPU out of memory: lower --batch-size, free the GPU (check 'nvidia-smi' for other processes), or use --device cpu."; fi
  if has 'no kernel image is available|not compatible with the current PyTorch'; then found=1
    hint "PyTorch has no kernels for this GPU architecture (e.g. P100/sm_60 with a new build): pick a T4 / install a matching build / --device cpu."; fi
  if has 'CUDA driver version is insufficient|NVIDIA driver on your system is too old|Torch not compiled with CUDA|Found no NVIDIA driver'; then found=1
    hint "Driver/PyTorch CUDA mismatch or a CPU-only PyTorch build: update the driver or reinstall a matching CUDA build of torch."; fi
  if has "No module named"; then found=1
    m=$(grep -Eoi "No module named '?[A-Za-z0-9_.]+" <<<"$tail_txt" | tail -n1 | sed -E "s/.*named '?//; s/\..*//")
    hint "Missing package '${m:-?}': pip install ${m:-<package>}  (import name and pip name can differ, e.g. PIL -> pillow)."; fi
  if has "has no attribute '_utils'|partially initialized module|cannot import name"; then found=1
    hint "Broken/mismatched install or a shadowing file: restart the session, delete any torch.py in the working dir, then pip install --force-reinstall torch torchvision."; fi
  if has 'LocalEntryNotFoundError|Temporary failure in name resolution|Name or service not known|ConnectError|Max retries exceeded|Connection (reset|refused)'; then found=1
    hint "No internet (dataset download or W&B). Kaggle: Session options -> Internet -> On. Or attach the dataset and use --data-dir; use --wandb-mode offline."; fi
  if has 'No (train|test) parquet found|AssertionError: No'; then found=1
    hint "Dataset files not found under --data-dir: check the path and that the download finished (re-run with Internet on)."; fi
  if has 'unrecognized arguments|error: argument|invalid choice'; then found=1
    hint "train.py rejected an argument. Compare with 'python train.py --help'; unknown flags must come after '--'."; fi
  if has 'DataLoader worker|shared memory|/dev/shm|Bus error'; then found=1
    hint "DataLoader worker crashed (low shared memory/RAM): try --num-workers 0 (or 1) and a smaller --batch-size."; fi
  if ((rc == 137)) || has '^Killed$'; then found=1
    hint "Process was killed (exit 137), usually the OS out-of-memory killer: reduce --num-workers and --batch-size."; fi
  if has 'No space left on device'; then found=1
    hint "Disk is full: free space, or point --data-dir/--save-path/--runs-dir at a larger volume."; fi
  if has 'PermissionError|Permission denied'; then found=1
    hint "Permission problem: make --data-dir, --save-path and the runs directory writable."; fi
  if has 'wandb.*(Error|error)|api_key|CommError|UsageError'; then found=1
    hint "Weights & Biases problem: set WANDB_API_KEY (or a Kaggle secret), or run with --wandb-mode offline / disabled."; fi
  if has 'use_deterministic_algorithms|CUBLAS_WORKSPACE_CONFIG|does not have a deterministic implementation'; then found=1
    hint "Deterministic mode hit an unsupported operation: re-run with --no-deterministic."; fi
  if has 'Expected all tensors to be on the same device'; then found=1
    hint "The script mixes CPU and GPU tensors (model and batches must be moved to the same device)."; fi
  if has "name 'get_ipython' is not defined|SyntaxError"; then found=1
    hint "Notebook-only code (get_ipython()/!pip/%magics) in the script: use a plain .py export."; fi
  if ((! found)); then
    hint "No known pattern matched. Read the last lines of the log above; full log: $1"
  fi
}

final_report() {   # rc
  local rc=$1 secs=$2 meaning outputs=""
  case $rc in
    0) meaning="success" ;;
    "$E_USAGE") meaning="bad arguments" ;;
    "$E_PREREQ") meaning="missing prerequisite" ;;
    "$E_FETCH") meaning="could not fetch the repository" ;;
    "$E_GPU") meaning="GPU required but unusable" ;;
    "$E_SCRIPT") meaning="train.py missing or invalid" ;;
    "$E_INTERRUPT") meaning="interrupted" ;;
    *) meaning="training failed (train.py exit code)" ;;
  esac
  if [[ -f $RC_FILE ]] && ((rc != 0 && rc != E_INTERRUPT)); then
    diagnose_failure "$LOG" "$rc"
  fi
  local sp=${GIVEN[save-path]:-${DEFAULTS[save-path]}}
  [[ -f $sp && $sp -nt $START_MARK ]] && outputs="$sp"
  raw ""
  raw "=============================================================="
  if ((rc == 0)); then raw " RESULT : SUCCESS"; else raw " RESULT : FAILED - exit code $rc ($meaning)"; fi
  raw " time   : $(fmt_dur "$secs")"
  raw " log    : $LOG   (latest: $RUNS_DIR/latest.log)"
  [[ -n $outputs ]] && raw " output : $outputs"
  if ((rc != 0)); then raw " help   : ./$PROG --help   (options, exit codes, examples)"; fi
  raw "=============================================================="
}

set +e
main 2>&1 | stream_filter | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
final_report "$RC" "$SECONDS" 2>&1 | stream_filter | tee -a "$LOG"
exit "$RC"
