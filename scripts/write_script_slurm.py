from pathlib import Path

path = Path(r"C:\Work\Thesis\scripts")
module_rnafold = "\nmodule load viennarna/2.5.0"
module_pkiss = "\nmodule load pkiss"
module_probknot = "\nexport PATH=$PATH:$WORK/RNAstructure/exe\nexport DATAPATH=$WORK/RNAstructure/data_tables"

for i in range(1, 56):
    with open(path / f"launch_script{i}.slurm", "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH --job-name=run{i}              # nom du job
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=run{i}_%j.out      # nom du fichier de sortie
#SBATCH --error=run{i}_%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load pytorch-gpu/py3/2.5.0{module_rnafold if i in [14, 20, 39, 45] else ""}{module_pkiss if i in [17, 24, 42, 49] else ""}{module_probknot if i in [18, 25, 43, 50] else ""}

python -u scripts/run_preds{i}.py
"""
        )
