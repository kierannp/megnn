#!/bin/bash
#SBATCH --mail-user=kieran.d.nehil-puleo@vanderbilt.edu
#SBATCH --mail-type=end
#SBATCH --error=MEGNN_%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=megnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=day-long-tesla
#SBATCH --exclude=node7,node9,node10,node3
#SBATCH --output=MEGNN_%J.txt

SLURM_SUBMIT_DIR=/raid6/homes/kierannp/projects/megnn
cd $SLURM_SUBMIT_DIR

module load anaconda/3.9
conda activate ml

python run_cof.py
