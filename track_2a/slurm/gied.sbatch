#!/bin/bash
#SBATCH --job-name=gied
#SBATCH --account=g238
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.log
#
# Decide the cases of one task with a gied_ strategy and Apertus v1.5 on the CSCS Inference API,
# then score the run, with the challenge's evidence score. Submit from track_2a/, the project root:
#
#   sbatch slurm/gied.sbatch                                       # gied_tags_prompt_adapted, advanced task, all 1,488 cases, 70B
#   LIMIT=150 sbatch -p debug -t 00:20:00 slurm/gied.sbatch        # a quick check on 150 cases
#   STRATEGY=gied_tags TASK=beginner sbatch slurm/gied.sbatch      # another strategy or task
#
# Settings, from the environment you submit from: STRATEGY, TASK (advanced or beginner), MODEL (a
# file in conf/model/), LIMIT (cases, spread evenly; all by default) and WORKERS (requests at once).
#
# Apertus runs at CSCS, not on this node: the job reads the booklets and sends the requests, so it
# spends most of its time waiting for answers. A full run of 1,488 cases takes about 10 to 20 minutes.
# It needs:
#   - the API key: `export API_KEY=...` before sbatch (Slurm passes your environment on), or API_KEY
#     in .env, see .env.example;
#   - the data in data/: `stage=prepare` runs first if it is missing. If the node cannot download
#     the booklets, copy data/raw, data/beginner and data/advanced over from your laptop instead;
#   - the Python packages: a venv on top of the uenv, made on the first run (about a minute).
#
# The run lands in data/runs/<date>_<task>_<strategy>_<model>_slurm-<job id>/, the log in slurm/logs/.

set -euo pipefail

# SBATCH_ACCOUNT or SLURM_ACCOUNT in the environment override the --account line above.
# Stop right away rather than run on another project's budget.
if [ "${SLURM_JOB_ACCOUNT:-g238}" != "g238" ]; then
    echo "ERROR: job charged to '${SLURM_JOB_ACCOUNT}', not g238. Unset SBATCH_ACCOUNT/SLURM_ACCOUNT or pass -A g238." >&2
    exit 2
fi

if [ -z "${API_KEY:-}" ] && ! grep -q "^API_KEY=." .env 2>/dev/null; then
    echo "ERROR: no API key. Run 'export API_KEY=...' before sbatch, or put API_KEY into .env." >&2
    exit 2
fi

export STRATEGY="${STRATEGY:-gied_tags_prompt_adapted}"
export TASK="${TASK:-advanced}"
export MODEL="${MODEL:-apertus-70b}"
export LIMIT="${LIMIT:-null}"
export WORKERS="${WORKERS:-8}"
export VENV="${VENV:-$SCRATCH/apertushack/venv}"
export RUN_DIR="data/runs/$(date +%Y-%m-%d_%H-%M-%S)_${TASK}_${STRATEGY}_${MODEL}_slurm-${SLURM_JOB_ID:-0}"
UENV="${UENV:-pytorch/v2.9.1:v2}"
echo "[gied.sbatch] job ${SLURM_JOB_ID:-?} on $(hostname), account ${SLURM_JOB_ACCOUNT:-?}: ${STRATEGY}, ${TASK} task," \
     "model ${MODEL}, limit ${LIMIT}, ${WORKERS} workers -> ${RUN_DIR}"

# --uenv/--view go on srun, not in the #SBATCH header: there they break the Slurm commands
# inside the job ("srun: fatal: failed to initialize cli_filter plugin").
srun --uenv="$UENV" --view=default bash -c '
    set -euo pipefail
    if [ ! -x "$VENV/bin/python" ]; then
        echo "[gied.sbatch] making the venv $VENV"
        python -m venv --system-site-packages "$VENV"
        "$VENV/bin/pip" install --quiet "hydra-core>=1.3" "openai>=1.50" "pypdfium2>=4.30" "rapidfuzz>=3.14"
    fi
    . "$VENV/bin/activate"
    if [ ! -f data/advanced/cases.jsonl ] || [ ! -f data/beginner/cases.jsonl ]; then
        python -m src stage=prepare
    fi
    python -m src task="$TASK" strategy="$STRATEGY" model="$MODEL" limit="$LIMIT" \
        backends.cscs.workers="$WORKERS" run_dir="$RUN_DIR"
    python -m src.strategies._gied.score run "$RUN_DIR"
'
