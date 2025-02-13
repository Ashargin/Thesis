#!/bin/bash
#SBATCH --job-name=runMxtrain_seq    # nom du job
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=runMxtrain_seq_%j.out # nom du fichier de sortie
#SBATCH --error=runMxtrain_seq_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --constraint v100-32g

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load pytorch-gpu/py3/2.5.0

mxfold2 train --model MixC --param sequencewise.pth --save-config sequencewise.conf --gpu 0 --log-dir mxfold2_logdir_sequencewise resources/data_structures/train_sequencewise/filelist.lst
