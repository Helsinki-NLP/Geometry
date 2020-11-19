#!/bin/bash
#SBATCH --job-name=geom_europarl
#SBATCH --time=72:00:00
#SBATCH --account=project_2001970
#SBATCH --mem-per-cpu=128G
#SBATCH --partition=small
#SBATCH -o out_%j
#SBATCH -e err_%j


# cd /projappl/project_2001970/Geometry/logs; sbatch ../europarl_compare_embeddings.sh 
curdir=`pwd`
cd /projappl/project_2001970/Geometry/code 
source /projappl/project_2001970/Geometry/env/bin/activate 

SAVEDATA=/scratch/project_2001970/Geometry/embeddings/Europarl_experiments

python compare_contextEmbeddings.py --load_embeddings_path $SAVEDATA/embeddings --outdir $SAVEDATA 
    
    
#python -m ipdb compare_contextEmbeddings.py --load_embeddings_path $SAVEDATA/embeddings --outdir $SAVEDATA --dev_params

echo "moving logfiles"
cd $curdir
sleep 2

mv err_${SLURM_JOBID} compEmbs_europarl_${SLURM_JOBID}_err
mv out_${SLURM_JOBID} compEmbs_europarl_${SLURM_JOBID}_out