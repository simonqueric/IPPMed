#!/bin/sh

# SLURM options:
#SBATCH --partition=IPPMED-A40              # Choix de partition (obligatoire)
#SBATCH --ntasks=1                    # Exécuter une seule tâche
#SBATCH --time=0-22:00:00     
#SBATCH --gpus=1

python3 ./3D-UNet/train.py
