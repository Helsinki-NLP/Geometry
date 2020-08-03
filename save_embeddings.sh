#!/bin/bash
#SBATCH --job-name=saveEmbs
#SBATCH --account=project_2001970
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH -o %x_%j_out
#SBATCH -e %x_%j_err

# cd /projappl/project_2001970/Geometry_jrvc/logs; sbatch ../save_embeddings.sh 



cd /projappl/project_2001970/Geometry/code 
source ../env/bin/activate 

SAVEDATA=/scratch/project_2001970/Geometry/embeddings/attempt

#python compare_contextEmbeddings.py --outdir $SAVEDATA --cuda --save_results --huggingface_models en-de en-fr en-ee  en-fi en-cs
#python compare_contextEmbeddings.py --outdir $SAVEDATA --cuda --save_results --bert_models None --huggingface_models en-it en-sv en-ru en-el


#rm -rf ~/.cache; sleep 2
#echo "en-af en-bg en-ca en-cs"
#python compare_contextEmbeddings.py --outdir $SAVEDATA --only_save_embs \
#    --cuda --save_results --bert_models None --huggingface_models en-af en-bg en-ca en-cs 

rm -rf ~/.cache; sleep 2
echo "en-da en-de en-et en-fi"
python compare_contextEmbeddings.py --outdir $SAVEDATA --only_save_embs \
    --cuda --save_results --bert_models None --huggingface_models en-da en-de en-et en-fi 

rm -rf ~/.cache; sleep 2
echo "en-fr en-gl en-hu en-is "
python compare_contextEmbeddings.py --outdir $SAVEDATA --only_save_embs \
    --cuda --save_results --bert_models None --huggingface_models en-fr en-gl en-hu en-is 

rm -rf ~/.cache; sleep 2
echo "en-it en-jap en-nl en-ro"
python compare_contextEmbeddings.py --outdir $SAVEDATA --only_save_embs \
    --cuda --save_results --bert_models None --huggingface_models en-it en-jap en-nl en-ro 

rm -rf ~/.cache; sleep 2
echo "en-sk en-sv en-uk en-ru"
python compare_contextEmbeddings.py --outdir $SAVEDATA --only_save_embs \
    --cuda --save_results --bert_models None --huggingface_models en-sk en-sv en-uk en-ru
    
# af-en  bg-en ca-en cs-en da-en de-en et-en fi-en fr-en gl-en hu-en is-en it-en jap-en nl-en ro-en  sk-en sv-en uk-en ru-en

#srun -p gputest --gres=gpu:v100:1,nvme:32 --time=00:15:00 --account=project_2001970 --mem=64G python compare_contextEmbeddings.py --outdir $SAVEDATA  --cuda --debug --bert_models None --huggingface_models en-et  en-nl en-af
#python compare_contextEmbeddings.py --load_embeddings_path $SAVEDATA/bert-base-uncased.h5 $SAVEDATA/bert-base-cased.h5 --debug

