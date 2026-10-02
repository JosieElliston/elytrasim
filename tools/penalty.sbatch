#!/bin/bash
#SBATCH --job-name=penalty
#SBATCH --partition=camelot
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=logs/%x-%A_%a.out
# One array task per node, 8 single-threaded runs at a time, as tools/floor.sbatch. TREE is the
# tree rsync'd to the cluster (its target/release/floor is the binary):
#   sbatch --array=0-$((N-1)) --export=ALL,WORK=<list>,RUN=<dir>,TREE=<tree> tools/penalty.sbatch
set -u
cd "$RUN" || exit 1
mkdir -p out
export FLOOR="$TREE/target/release/floor" RAYON_NUM_THREADS=1
NSHARD=${SLURM_ARRAY_TASK_COUNT:?submit as an array}
ID=${SLURM_ARRAY_TASK_ID:?submit as an array}
MINE=$(mktemp)
awk -v n="$NSHARD" -v i="$ID" 'NR % n == i' "$WORK" > "$MINE"
echo "node=$(hostname) shard=$ID/$NSHARD lines=$(wc -l < "$MINE")"
xargs -P 8 -d '\n' -I@ "$TREE/tools/penalty_job.sh" "@" < "$MINE"
rm -f "$MINE"
