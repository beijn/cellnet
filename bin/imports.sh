
source bin/secrets.sh  # NOTE hide this via gitignore 
# This should define HPC_SSH_HOST


# HPC args (adapt these to your HPC(s))
export HPC_UTHPC_ARGS="--cpus-per-task=8 --mem=40G --partition=gpu --gres=gpu:a100-80g:1 $HPC_UTHPC_ARGS_SECRET"	
export HPC_UTHPC_LIVE="srun $HPC_TARTU_ARGS --time 6:00:00 --gres=gpu:tesla:1"
export HPC_UTHPC_BATCH=""
#export LSF_ARGS="span[ptile=8] rusage[mem=40000] select[mem>40000]"

export HPC_SCC_ARGS="--cpus-per-task=8 --mem=40G --partition=scc-gpu -G A100 -C inet"	
export HPC_SCC_BATCH="sbatch $HPC_SCC_ARGS"
export HPC_SCC_LIVE="srun $HPC_SCC_ARGS --partition=jupyter -G V100 --time 6:00:00"


export HPC_LIVE="$HPC_SCC_LIVE"
export HPC_BATCH="$HPC_SCC_BATCH"


if [ -z "$GITHUB_TOKEN" ]; then
  export GITHUB_TOKEN="GITHUB_TOKEN_WAS_NOT_DEFINED_IN_bin/secrets.sh"
fi



