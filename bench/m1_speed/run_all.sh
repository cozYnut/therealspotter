#!/bin/zsh
# The M1 speed test: one heat, analysed 3 ways one after another (~25 min on an M1).
#   bench/m1_speed/run_all.sh "<the Race 1 Season 2027 event folder>"
# A run that fills the memory (critical pressure, or swap grows > 1 GB) is stopped, and so is the test.
# Results: bench/m1_speed/results/<computer>-<date>/ (results.jsonl, system.txt, a log per run).
# The analyses' own files go to /tmp/m1_speed_out (not kept).
#   ONLY="base4": just those runs;  TAG=_fpvt: added to the names (e.g. a round with FPVTrackside open)
set -u
EVENT="$1"
ONLY=${ONLY:-}
TAG=${TAG:-}
HEAT=${HEAT:-b9661a9b}                    # Race 1 Season 2027, heat 53: 4 pilots, 93 s race, ~42 passes each
HERE=${0:A:h}
APP=${HERE:h:h}
PY="$APP/venv/bin/python"
RES="$HERE/results/$(scutil --get LocalHostName 2>/dev/null || hostname -s)-$(date +%Y%m%d-%H%M)$TAG"
OUT=/tmp/m1_speed_out
[[ -n "$(ls "$EVENT" 2>/dev/null | grep "^$HEAT")" ]] || { echo "heat $HEAT not in $EVENT"; exit 1; }
mkdir -p "$RES" "$OUT"

{
  echo "date: $(date)"
  echo "git: $(git -C "$APP" rev-parse --short HEAD) ($(git -C "$APP" branch --show-current))"
  echo "chip: $(sysctl -n machdep.cpu.brand_string)"
  echo "memory_gb: $(( $(sysctl -n hw.memsize) / 1073741824 ))"
  echo "cores: $(sysctl -n hw.ncpu) (performance $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null), efficiency $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null))"
  echo "macos: $(sw_vers -productVersion)"
  echo "python: $("$PY" -V 2>&1)"
  echo "swap at start: $(sysctl -n vm.swapusage)"
  echo "memory pressure at start: $(sysctl -n kern.memorystatus_vm_pressure_level) (1 normal, 2 warning, 4 critical)"
  echo "running apps using CPU (top 8):"
  ps -Ao pcpu,rss,comm -r | head -9
  echo "event: $EVENT"
  echo "heat: $HEAT"
} > "$RES/system.txt"

run() {   # name pilots variant
  local name=$1$TAG p=$2 v=$3
  [[ -n "$ONLY" && " $ONLY " != *" $1 "* ]] && return 0
  echo "$(date +%H:%M:%S) $name …"
  rm -rf "$OUT/$name"
  "$PY" "$HERE/speed_test.py" "$EVENT" "$HEAT" "$v" "$p" "$OUT/$name" > "$OUT/$name.json" 2> "$RES/$name.log"
  local code=$?
  echo "{\"name\": \"$name\", \"exit\": $code, \"r\": $(tail -1 "$OUT/$name.json" 2>/dev/null || echo null)}" >> "$RES/results.jsonl"
  tail -1 "$RES/results.jsonl" | cut -c1-220
  tail -c 20000 "$RES/$name.log" > "$RES/$name.log.tmp" && mv "$RES/$name.log.tmp" "$RES/$name.log"   # keep it small
  if [[ $code == 3 ]]; then
    echo "STOPPED: $name filled the memory — the test stops here" | tee -a "$RES/system.txt"
    return 3
  fi
  sleep 20                                # let the machine settle between runs
}

run base4    4 base   &&                  # the app as it is
run split2_4 4 split2 &&                  # the pilots in 2 processes (2 each, each its own YOLO and CLIP)
run split2_6 6 split2                     # 6 pilots in 2 processes (3 each)

echo "swap at end: $(sysctl -n vm.swapusage)" >> "$RES/system.txt"
echo "done: $RES"
