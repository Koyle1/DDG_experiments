#!/bin/bash
#SBATCH --job-name=graph_embed_gp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --hint=multithread
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=out/slurm_%j.out
#SBATCH --error=out/slurm_%j.err

set -euo pipefail

# Graph embedding GP (island model) on synthetic graph families.
# Mirrors the structure of run_psyclass_intersection_folded_mis_cluster.sh
# while targeting this experiment's pipeline.

PROJECT_DIR="${PROJECT_DIR:-$HOME/project/DDG_experiments/20-graph-embedding-gp}"
cd "$PROJECT_DIR"
mkdir -p out logs

# --- Environment setup ---
# Adjust module names/versions to your cluster defaults if needed.
module load python/3.12 || true

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/cluster_${timestamp}.txt"

python3 learn_embedding_gp.py \
  --generations 600 \
  --num-islands 8 \
  --migration-interval 25 \
  --migration-size 2 \
  --topology ring \
  --population 48 \
  --offspring 240 \
  --rng-seed 42 \
  --grammar base \
  --num-graphs 96 \
  --min-nodes 20 \
  --max-nodes 80 \
  --test-fraction 0.25 \
  --max-pairs 3000 \
  --knn-k 8 \
  --max-train-graphs 64 \
  --stress-weight 1.0 \
  --knn-weight 0.35 \
  --length-penalty \
  --epoch-resampling 25 \
  --log-interval 10 \
  --checkpoint-interval 50 \
  --log-file "$log_file" \
  "$@"

echo "Log saved to: $log_file"
