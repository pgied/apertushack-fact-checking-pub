#!/bin/bash
#SBATCH --job-name=apertushack
#SBATCH --account=g238
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.log
#
# Runs one command on a Clariden compute node, charged to the hackathon project g238.
# Submit from track_2a/, the project root, since the log path above is relative to it:
#
#   sbatch slurm/run.sbatch python my_script.py --some-arg
#   sbatch -p debug -t 00:20:00 slurm/run.sbatch python my_script.py    # short test run
#
# The command runs inside a CSCS uenv: by default pytorch/v2.9.1:v2 (Python 3.12 with PyTorch),
# pulled once with `uenv image pull pytorch/v2.9.1:v2`. Set UENV to use another one. For our own
# packages, set VENV to a venv built on top of the uenv with `python -m venv --system-site-packages`.
#
# Clariden bills the whole node (4 GPUs, 288 cores) for as long as the job runs, so the job asks
# for all of it.

set -euo pipefail

# SBATCH_ACCOUNT or SLURM_ACCOUNT in the environment override the --account line above.
# Stop right away rather than run on another project's budget.
if [ "${SLURM_JOB_ACCOUNT:-g238}" != "g238" ]; then
    echo "ERROR: job charged to '${SLURM_JOB_ACCOUNT}', not g238. Unset SBATCH_ACCOUNT/SLURM_ACCOUNT or pass -A g238." >&2
    exit 2
fi

if [ $# -eq 0 ]; then
    echo "usage: sbatch slurm/run.sbatch <command> [args...]" >&2
    exit 2
fi

UENV="${UENV:-pytorch/v2.9.1:v2}"
echo "[run.sbatch] job ${SLURM_JOB_ID:-?} on $(hostname), account ${SLURM_JOB_ACCOUNT:-?}, uenv ${UENV}"
echo "[run.sbatch] command: $*"

# --uenv/--view go on srun, not in the #SBATCH header: there they break the Slurm commands
# inside the job ("srun: fatal: failed to initialize cli_filter plugin").
srun --uenv="$UENV" --view=default bash -c '
    if [ -n "${VENV:-}" ]; then . "$VENV/bin/activate"; fi
    exec "$@"
' bash "$@"
