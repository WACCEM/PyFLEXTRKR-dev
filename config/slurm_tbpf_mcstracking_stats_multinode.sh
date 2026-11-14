#!/bin/bash
#SBATCH -A @ACCOUNT@
#SBATCH -J @JOB_NAME@
#SBATCH --qos=@QOS@
#SBATCH --nodes=@NODES@
#SBATCH -C cpu
#SBATCH --time=@WALLCLOCK@
#SBATCH --exclusive
#SBATCH --output=logs/log_@JOB_NAME@_%j.log
@MAIL_LINES@
# The stats job of a chunk on several nodes (config/generate_tbpf_mcstracking_scripts.py --stats-nodes N): the links to the cloudid and track
# files of the months and the stitched track numbers first (serial, a few minutes), then trackstats ... speed over the whole chunk with
# run_parallel: 2. A Dask scheduler runs on the first node and @WORKERS_PER_NODE@ workers on every node (the scheduler file is the second argument
# of the run script). The serial steps (identifymcs, robustmcs) run on the first node while the others wait, so more nodes pay only when the
# parallel steps (trackstats, matchpf, mapfeature, speed) dominate.
date

module load python
source activate @PYTHON_ENV@
@ENV_LINES@
@PRELUDE@ulimit -n 32000
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=360s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=360s
workers_per_node=@WORKERS_PER_NODE@
scheduler_file=$SCRATCH/scheduler_${SLURM_JOB_ID}.json
rm -f $scheduler_file

dask-scheduler --scheduler-file=$scheduler_file > logs/dask_scheduler_${SLURM_JOB_ID}.log 2>&1 &
until [ -s $scheduler_file ]; do sleep 2; done
srun --kill-on-bad-exit=0 -N $SLURM_NNODES --ntasks-per-node=$workers_per_node --cpus-per-task=@CPUS_PER_WORKER@ --cpu-bind=cores \
    dask-worker --scheduler-file=$scheduler_file --nthreads 1 --memory-limit=@WORKER_MEMORY@ --local-directory=/tmp \
    --worker-class distributed.Worker > logs/dask_workers_${SLURM_JOB_ID}.log 2>&1 &
# the run starts when all the workers are there (10 min at most)
python -c "
from dask.distributed import Client
client = Client(scheduler_file='$scheduler_file', timeout=120)
client.wait_for_workers($workers_per_node * $SLURM_NNODES, timeout=600)
client.close()"
status=$?
if [ $status -ne 0 ]; then
    echo "the Dask workers did not connect"
    exit $status
fi

# the exit status of python (not of the last date) is the exit status of the job, so a failed step shows as FAILED in sacct
python @RUN_SCRIPT@ @CONFIG@ $scheduler_file
status=$?

date
exit $status
