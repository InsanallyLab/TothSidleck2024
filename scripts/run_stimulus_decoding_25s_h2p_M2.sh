#!/bin/bash
#SBATCH --job-name=stimdecoding25s                # Job name
#SBATCH --mail-type=ALL                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=tih85@pitt.edu    # Where to send mail
#SBATCH --ntasks=1                          # Run on seven CPUs
#SBATCH --cpus-per-task=10                   # Number of CPU cores per task
#SBATCH --mem=32gb                          # Job memory request
#SBATCH --time=48:00:00                     # Time limit hrs:min:sec
#SBATCH --output=results_experiment_RevLearn_stimdecoding_25s_%j.log # Standard output and error log

#module add git
#module add gcc/8.2.0
#module add python/anaconda3.7-2019.03

#export JULIA_DEPOT_PATH="~/.julia"
#export JULIA_PROJECT="~/AL_RNN_notebooks/"

python ~/EphysAnalysis/scripts/run_stimulus_decoding_25s_M2.py /bgfs/minsanally/tih85/Analysis_Cache /bgfs/minsanally/tih85/results_experiment_RevLearn_stimdecoding_resp 500 stimulus