#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=batch
#SBATCH --nodelist=sof1-h200-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=16G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=10-0         # Job timeout
#SBATCH --output=./joblogs/bash%N.log
#SBATCH --error=./joblogs/bash%N.error
#SBATCH --mail-type=ALL         # Send updates via Slack

cd ~
./cursor tunnel