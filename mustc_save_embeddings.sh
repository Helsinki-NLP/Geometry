#!/bin/bash
#SBATCH --job-name=saveEmbs_mustc
#SBATCH --account=project_2001970
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH -o %x_%j_out
#SBATCH -e %x_%j_err

# cd /projappl/project_2001970/Geometry_jrvc/logs; sbatch ../mustc_save_embeddings.sh 



cd /projappl/project_2001970/Geometry/code 
source ../env/bin/activate 

echo -e "\n DE-EN \n"

python compare_contextEmbeddings.py  --save_results --only_save_embs \
--bert_models None  --huggingface_models de-en \
--data_path ../data/mustc.train.ende50k.de \
--outdir /scratch/project_2001970/Geometry/embeddings/mustc

echo -e "\n EN-DE \n"

python compare_contextEmbeddings.py  --save_results --only_save_embs \
--bert_models None  --huggingface_models en-de \
--data_path ../data/mustc.train.ende50k.en \
--outdir /scratch/project_2001970/Geometry/embeddings/mustc
