#!/bin/bash
#SBATCH -A @ACCOUNT@
#SBATCH -J @JOB_NAME@
#SBATCH --qos=@QOS@
#SBATCH --nodes=1
#SBATCH -C cpu
#SBATCH --time=@WALLCLOCK@
#SBATCH --exclusive
#SBATCH --output=logs/log_@JOB_NAME@_%j.log
@MAIL_LINES@
# One node, one job of config/generate_tbpf_mcstracking_scripts.py: a track job (idfeature, tracksingle, gettracks of one month), the
# plain job of a chunk of one month (all steps), or a stats job on one node (--stats-nodes 1: the links and the stitched track numbers of
# its chunk first, then trackstats ... speed over the whole chunk). The run script starts its own Dask LocalCluster (run_parallel: 1 and
# nprocesses in the config). The filled copies of this template are written by the generator, which says what each job does.
date

module load python
source activate @PYTHON_ENV@
@ENV_LINES@
@PRELUDE@# the exit status of python (not of the last date) is the exit status of the job, so a failed step shows as FAILED in sacct
python @RUN_SCRIPT@ @CONFIG@
status=$?

date
exit $status
