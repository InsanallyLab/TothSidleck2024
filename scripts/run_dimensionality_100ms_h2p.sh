#!/bin/bash
#SBATCH --job-name=dimensionality100ms                # Job name
#SBATCH --mail-type=ALL                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jackmtoth@pitt.edu    # Where to send mail
#SBATCH --ntasks=1                          # Run on seven CPUs
#SBATCH --cpus-per-task=16                   # Number of CPU cores per task
#SBATCH --mem=32gb                          # Job memory request
#SBATCH --time=48:00:00                     # Time limit hrs:min:sec
#SBATCH --output=results_experiment_RevLearn_dimensionality_100ms_%j.log # Standard output and error log

#module add git
#module add gcc/8.2.0
#module add python/anaconda3.7-2019.03

#export JULIA_DEPOT_PATH="~/.julia"
#export JULIA_PROJECT="~/AL_RNN_notebooks/"

python ~/EphysAnalysis/scripts/run_dimensionality.py /bgfs/balbanna/jmt195/Analysis_Cache /bgfs/balbanna/jmt195/results_experiment_RevLearn_dimensionality_100ms 10 100 dimenstionality_100ms