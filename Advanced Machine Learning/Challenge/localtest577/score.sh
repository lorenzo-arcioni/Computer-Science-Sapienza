#!/bin/bash
# Portable localtest577 wrapper.
#   ./score.sh <submission.csv> <exp_id> [note]
# Scores one frozen prediction file against labels/test_labels.json.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -lt 2 ]; then
  echo "usage: $(basename "$0") <submission.csv> <exp_id> [note]" >&2
  exit 2
fi
exec "$here/score.py" "$@"
