#!/usr/bin/env bash
# Single source of truth for TEST_TYPE: the tier aliases AND the valid set.
# Shared by the Makefile TEST_TYPE resolution and _test_matrix.yaml's resolve
# and gate steps, so every entry point applies the same mapping and the same
# validation. To add a new test type, edit ONLY this file (VALID_TYPES, plus
# an alias arm below if it has a user-facing name); the callers do not change.
#
# Usage: resolve_test_type.sh TEST_TYPE
# Prints the resolved canonical type to stdout. If the resolved value is not in
# VALID_TYPES, prints a GitHub Actions ::error:: line to stderr and exits 1, so
# callers do not re-validate. Empty/no arg resolves to "full" (this repo's own
# default label; callers that want the user-facing "regression" default pass it
# explicitly).

set -euo pipefail

# The canonical test types. This is the ONLY list to edit when adding a type.
VALID_TYPES="smoke core full trunk perf"
# The user-facing tier aliases, for the error message. Keep in sync with the
# alias arms of the case below and the choice options callers expose.
ALIASES="unit | integration | regression"

RAW_TEST_TYPE="${1:-}"

case "$RAW_TEST_TYPE" in
    unit)        resolved=core ;;
    integration) resolved=smoke ;;
    regression)  resolved=full ;;
    '')          resolved=full ;;
    *)           resolved="$RAW_TEST_TYPE" ;;
esac

case " $VALID_TYPES " in
    *" $resolved "*) ;;
    *)
        valid_display="${VALID_TYPES// / | }"
        echo "::error::Invalid test_type '${RAW_TEST_TYPE}'. Valid: ${valid_display} | ${ALIASES}" >&2
        exit 1
        ;;
esac

echo "$resolved"
