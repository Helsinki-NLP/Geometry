#!/bin/bash
#SBATCH --job-name=saveEmbs
#SBATCH --account=project_2001970
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:64
#SBATCH -o saveEmbs_%j_out
#SBATCH -e saveEmbs_%j_err

# cd /projappl/project_2001970/Geometry_jrvc/logs; sbatch ../save_embeddings.sh 
cd /projappl/project_2001970/Geometry_jrvc/code 
source ../env/bin/activate 

SAVEDATA=/scratch/project_2001970/Geometry/embeddings/attempt2

#python compare_contextEmbeddings.py --outdir $SAVEDATA --cuda --save_results --huggingface_models en-de en-fr en-ee  en-fi en-cs
#python compare_contextEmbeddings.py --outdir $SAVEDATA --cuda --save_results --bert_models None --huggingface_models en-it en-sv en-ru en-el


echo -e "Helsinki-NLP/opus-mt-en-et  Helsinki-NLP/opus-mt-en-nl Helsinki-NLP/opus-mt-en-af \n"
python compare_contextEmbeddings.py --outdir $SAVEDATA --cuda --save_results --bert_models None \
 --huggingface_models en-et  en-nl en-af 
echo -e "\n Helsinki-NLP/opus-mt-en-bg  Helsinki-NLP/opus-mt-en-hu Helsinki-NLP/opus-mt-en-sk \n"
python compare_contextEmbeddings.py --outdir $SAVEDATA --cuda --save_results --bert_models None \
 --huggingface_models en-bg  en-hu en-sk 
echo -e "\n Helsinki-NLP/opus-mt-en-jap Helsinki-NLP/opus-mt-en-ro Helsinki-NLP/opus-mt-en-ca  \n"
python compare_contextEmbeddings.py --outdir $SAVEDATA --cuda --save_results --bert_models None \
 --huggingface_models en-jap en-ro en-ca 
echo -e "\n Helsinki-NLP/opus-mt-en-pl  Helsinki-NLP/opus-mt-en-da \n"
python compare_contextEmbeddings.py --outdir $SAVEDATA --cuda --save_results --bert_models None \
 --huggingface_models en-pl  en-da

#srun -p gputest --gres=gpu:v100:1,nvme:32 --time=00:15:00 --account=project_2001970 --mem=64G python compare_contextEmbeddings.py --outdir $SAVEDATA  --cuda --debug --bert_models None --huggingface_models en-et  en-nl en-af
#python compare_contextEmbeddings.py --load_embeddings_path $SAVEDATA/bert-base-uncased.h5 $SAVEDATA/bert-base-cased.h5 --debug
