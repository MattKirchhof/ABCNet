#!/bin/bash
#SBATCH --time=03-00:00:00 # DD-HH:MM
#SBATCH --account=def-skremer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=noATP_%A_%a.log
#SBATCH --mem=32G
#SBATCH --array=0-18
module load python/3.9.6
module load cuda/11.2.2
virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install torch --no-index
pip install numpy --no-index
pip install matplotlib --no-index
python ABCModelHarness.py $SLURM_ARRAY_TASK_ID noATP
