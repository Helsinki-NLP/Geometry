#!/bin/bash
#SBATCH --job-name=europarl_afterAlign
#SBATCH --time=72:00:00
#SBATCH --account=project_2001970
#SBATCH --mem-per-cpu=128G
#SBATCH --partition=small
#SBATCH -o out_%j
#SBATCH -e err_%j



# cd /projappl/project_2001970/Geometry/logs; sbatch ../europarl_compare_embs_after_alignment.sh


cd /projappl/project_2001970/Geometry/code 
source /projappl/project_2001970/Geometry/env/bin/activate 

tgtlang=de
langpair="en-${tgtlang}"
datadir=/scratch/project_2001970/Geometry/data/
SAVEDATA=/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/prealigned



python compare_contextEmbeddings.py --load_embeddings_path $SAVEDATA/embeddings --outdir $SAVEDATA 



echo "moving logfiles"
cd $curdir
sleep 2

mv err_${SLURM_JOBID} compareEmbs_europarl_afterAlign_${SLURM_JOBID}_err
mv out_${SLURM_JOBID} compareEmbs_europarl_afterAlign_${SLURM_JOBID}_out