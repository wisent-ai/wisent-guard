#!/usr/bin/env bash
# Real test of the pair and activation counts through the built binary, on a
# toy checkpoint it writes under this run's directory: a run with too few
# pairs or prompts fails with the counts, and a run with enough reads exactly
# one activation per prompt and requested layer.
#
# Refusals: a BiPO export whose rows all lack an answer yields no pair and is
# refused on import; an export yielding fewer pairs than --count is refused
# naming both numbers; `vector train` on a set with no pairs and `extract` on
# a set with no prompts are refused before any weight is mapped. Success:
# `extract` over the checked-in prompt set writes one record per prompt, each
# holding every requested layer, and `vector train` over the checked-in pairs
# writes one vector per requested layer. Every command, whether it was
# accepted, and its answer go to the run's report.txt; a failed check stops
# the run.
#
# Usage: STER=target/debug/ster STER_TEST_SEED=<seed> tests/pairs/counts.sh
set -eu
cd "$(dirname "$0")/../.."
BIN=${STER:?set STER to the ster binary under test, e.g. STER=target/debug/ster}
SEED=${STER_TEST_SEED:?set STER_TEST_SEED to the seed the imports draw with; ster assumes none}
RUN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT="$PWD/target/real-tests/pairs/counts-$RUN"
REPORT="$ROOT/report.txt"
mkdir -p "$ROOT"
echo "revision: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')" >"$REPORT"
echo "binary: $BIN" >>"$REPORT"

fail() {
  echo "FAIL: $1" | tee -a "$REPORT" >/dev/stderr
  false
}
# run accepted|refused ARGS...: the command must end the way named. An accepted
# run's answer is its stdout; a refused run's answer is everything it wrote.
run() {
  expected=$1
  shift
  if [ "$expected" = accepted ]; then
    if "$BIN" "$@" >"$ROOT/output" </dev/null; then outcome=accepted; else outcome=refused; fi
  else
    if "$BIN" "$@" &>"$ROOT/output" </dev/null; then outcome=accepted; else outcome=refused; fi
  fi
  out=$(cat "$ROOT/output")
  printf '$ ster %s\noutcome: %s\noutput: %s\n\n' "$*" "$outcome" "$out" >>"$REPORT"
  [ "$outcome" = "$expected" ] || fail "$* was $outcome, expected $expected: $out"
}
check() {
  [ "$2" = "$3" ] || fail "$1: got '$2', expected '$3'"
  echo "ok: $1 = $2" >>"$REPORT"
}
refused() {
  case "$out" in
    *"$1"*) echo "ok: refused with: $1" >>"$REPORT" ;;
    *) fail "expected a refusal containing '$1', got: $out" ;;
  esac
}

TOY="$ROOT/toy-model"
PAIRS=docs/examples/pairs.json
PROMPTS=docs/examples/prompts.json
echo "seed: $SEED" >>"$REPORT"
run accepted toy-model "$TOY"
LAYERS=$(jq .num_hidden_layers "$TOY/config.json")
ALL=$(jq -r '[range(.num_hidden_layers)] | map(tostring) | join(",")' "$TOY/config.json")

# Imports that yield too few pairs.
UNANSWERED="$ROOT/unanswered.csv"
printf 'question,matching,not_matching\nIs it calm?,,\nIs it quiet?,,\n' >"$UNANSWERED"
run refused pairs import --benchmark bipo --source "$UNANSWERED" --seed "$SEED" --output "$ROOT/none.pairs.json"
refused "contains no pairs"
check "a refused import writes nothing" "$(test -e "$ROOT/none.pairs.json" && echo written || echo absent)" "absent"

ONE="$ROOT/one.csv"
printf 'question,matching,not_matching\nIs it calm?,Yes,No\nIs it quiet?,,\n' >"$ONE"
YIELD=$(grep -c ',Yes,No$' "$ONE")
# One pair asked for each question the export holds, answered or not.
ASKED=$(grep -c '^Is it' "$ONE")
run refused pairs import --benchmark bipo --source "$ONE" --count "$ASKED" --seed "$SEED" --output "$ROOT/few.pairs.json"
refused "yields $YIELD pairs, fewer than --count $ASKED"

# Runs over sets with nothing in them.
EMPTY_PAIRS="$ROOT/empty.pairs.json"
jq '.pairs = []' "$PAIRS" >"$EMPTY_PAIRS"
run refused vector train --model "$TOY" --pairs "$EMPTY_PAIRS" --output "$ROOT/empty.ster.json" --layers all
refused "contains no pairs"
EMPTY_PROMPTS="$ROOT/empty.prompts.json"
jq '.prompts = []' "$PROMPTS" >"$EMPTY_PROMPTS"
run refused extract --model "$TOY" --input "$EMPTY_PROMPTS" --output "$ROOT/empty.activations.json" --layers all
refused "prompt set contains no prompts"

# Enough to read: one activation per prompt and requested layer.
run accepted extract --model "$TOY" --input "$PROMPTS" --output "$ROOT/activations.json" --layers all
check "one record per prompt" \
  "$(jq '.records | length' "$ROOT/activations.json")" "$(jq '.prompts | length' "$PROMPTS")"
check "every record holds every requested layer" \
  "$(jq -r '[.records[] | (.layers | keys | map(tonumber) | sort | map(tostring) | join(","))] | unique | join(";")' "$ROOT/activations.json")" "$ALL"
check "every activation vector is as wide as the model" \
  "$(jq --argjson width "$(jq .hidden_size "$TOY/config.json")" '[.records[].layers[] | length == $width] | all' "$ROOT/activations.json")" "true"
run accepted vector train --model "$TOY" --pairs "$PAIRS" --output "$ROOT/calm.ster.json" --layers all
check "one vector per requested layer" "$(jq '.vectors | length' "$ROOT/calm.ster.json")" "$LAYERS"

touch "$ROOT/passed"
echo "PASS" >>"$REPORT"
echo "PASS: $REPORT"
