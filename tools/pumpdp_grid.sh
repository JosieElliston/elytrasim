#!/bin/bash
# Submit pumpdp's grid study to cif-cpu's camelot partition, one exclusive node per solve:
# lambda x node spacing x how V is read when the policy is extracted. The solve itself is always
# bilinear (see pumpdp's Interp: a cubic read inside the max gets exploited).
#
#   tools/pumpdp_grid.sh <remote run dir>      e.g. tools/pumpdp_grid.sh pumpdp-grid
#
# Expects ~/pumpdp-clean to hold this tree, built (rsync it, then
# `srun -p camelot -c 8 --exclusive cargo build --release --bin pumpdp`).
set -u
RUN=${1:?usage: pumpdp_grid.sh <remote run dir>}
HOST=${HOST:-cif-cpu}
ssh -o BatchMode=yes "$HOST" "bash -s" <<REMOTE
mkdir -p ~/$RUN/logs && cd ~/$RUN || exit 1
for lam in 0 4.3; do for dv in 0.01 0.005 0.0025; do for ex in linear clipped cubic; do
  name=lam\${lam}_dv\${dv}_\${ex}
  sbatch -p camelot -N 1 --exclusive -J \$name -o logs/\$name.log --wrap \
    "~/pumpdp-clean/target/release/pumpdp --mode disc --discount 0.9996666666666667 --evals 300 \
     --conv-lift-exp 0.99 --lambda \$lam --dv \$dv --vymin -4 --vymax 4 --vzmax 4.5 \
     --extract \$ex --every 20 --out \$name"
done; done; done
squeue -u \$USER -h | wc -l
REMOTE
