#!/bin/bash
# Interactive shell on a Clariden GPU node, charged to the hackathon project g238.
#
#   bash slurm/interactive.sh                          # 30 minutes on the debug partition
#   bash slurm/interactive.sh -p normal -t 02:00:00    # extra srun options override the defaults
#
# The shell runs inside a CSCS uenv: by default pytorch/v2.9.1:v2 (Python 3.12 with PyTorch),
# pulled once with `uenv image pull pytorch/v2.9.1:v2`. Set UENV to use another one.
# The whole node is billed until you exit, so exit as soon as you are done.
#
# --account is given on the command line, which wins over SLURM_ACCOUNT in the environment and
# over your default project.

set -euo pipefail

exec srun --account=g238 --partition=debug --time=00:30:00 --nodes=1 --ntasks=1 \
    --uenv="${UENV:-pytorch/v2.9.1:v2}" --view=default \
    "$@" --pty bash
