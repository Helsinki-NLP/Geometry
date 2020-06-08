#!/bin/bash
#SBATCH --job-name=geom
#SBATCH --account=project_2001970
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=38G
#SBATCH --partition=small
#SBATCH -o out_%j
#SBATCH -e err_%j

# cd /projappl/project_2001970/Geometry_jrvc/logs; sbatch ../compare_embeddings.sh 
curdir=`pwd`
cd /projappl/project_2001970/Geometry_jrvc/code 
source ../env/bin/activate 

SAVEDATA=/scratch/project_2001970/Geometry/embeddings/attempt2

modname=$1



python compare_contextEmbeddings.py --outdir $SAVEDATA --load_embeddings_path $SAVEDATA/embeddings/$modname


echo "moving logfiles"
cd $curdir
sleep 2

mv err_${SLURM_JOBID} err_${modname}_${SLURM_JOBID}
mv out_${SLURM_JOBID} out_${modname}_${SLURM_JOBID}
