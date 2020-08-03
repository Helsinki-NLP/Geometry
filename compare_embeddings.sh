#!/bin/bash
#SBATCH --job-name=geom
#SBATCH --account=project_2001970
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=small
#SBATCH -o out_%j
#SBATCH -e err_%j

# cd /projappl/project_2001970/Geometry_jrvc/logs; sbatch ../compare_embeddings.sh 
curdir=`pwd`
cd /projappl/project_2001970/Geometry/code 
source ../env/bin/activate 

#SAVEDATA=/scratch/project_2001970/Geometry/embeddings/attempt
    
modname=$1
SAVEDATA=$2

echo "running with model: ${modname}"
python compare_contextEmbeddings.py --outdir $SAVEDATA --load_embeddings_path $SAVEDATA/embeddings/${modname}.h5 --save_results


echo "moving logfiles"
cd $curdir
sleep 2

mv err_${SLURM_JOBID} err_${modname}_${SLURM_JOBID}
mv out_${SLURM_JOBID} out_${modname}_${SLURM_JOBID}
