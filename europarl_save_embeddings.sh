#!/bin/bash
#SBATCH --job-name=geom_europarl
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --account=project_2001970
#SBATCH --mem-per-cpu=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH -o out_%j
#SBATCH -e err_%j


# cd /projappl/project_2001970/Geometry/logs; sbatch ../europarl_save_embeddings.sh 
curdir=`pwd`
cd /projappl/project_2001970/Geometry/code 
source /projappl/project_2001970/Geometry/env/bin/activate 

SAVEDATA=/scratch/project_2001970/Geometry/embeddings/Europarl_experiments
datadir=/scratch/project_2001970/Geometry/data/

#tgtlang="de"
#SAVEDATA=$2

# langlist=('bg' 'cs' 'da' 'de' 'el' 'es' 'et' 'fi' 'fr' 'hu' 'it' 'lt' 'lv' 'nl' 'pl' 'pt' 'ro' 'sk' 'sl' 'sv')

langlist=('bg' 'cs' 'da' 'de' 'es' 'et' 'fi' 'fr' 'hu' 'it'                'nl'            'sk'      'sv'  'el' 'ro')


#langlist=( 'fr' 'it' 'es' )
for tgtlang in ${langlist[@]}; do
    rm -rf ~/.cache; sleep 2
    langpair="en-${tgtlang}"

    echo "running with model for language pair: ${langpair}"

    echo -e "\n ####### \n running with model for language pair: ${langpair}" >> err_${SLURM_JOBID}

    #srun --time=48:00:00 --mem-per-cpu=64G --account=project_2001970 --mem-per-cpu=128G --partition=gpu --gres=gpu:v100:1,nvme:32 \
    python compare_contextEmbeddings.py --data_path  ${datadir}/${langpair}.en   ${datadir}/${langpair}.${tgtlang} \
                                         -hfmodels ${langpair} \
                                        --outdir $SAVEDATA --only_save_embs --cuda --debug_mode
#python compare_contextEmbeddings.py --bert_models None --data_path  ${datadir}/${langpair}.en ${datadir}/${langpair}.${tgtlang} -hfmodels ${langpair}  --outdir $SAVEDATA --only_save_embs --dev_params
    echo -e "############################\n"
done





cd /projappl/project_2001970/Geometry/code 
source /projappl/project_2001970/Geometry/env/bin/activate 

tgtlang=de
langpair="en-${tgtlang}"
datadir=/scratch/project_2001970/Geometry/data/
SAVEDATA=/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/prealigned

#srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=128GB --time=01:30:00 \
python compare_contextEmbeddings.py --data_path  ${datadir}/${langpair}.en   ${datadir}/${langpair}.${tgtlang} \
                                    --prealigned_models /scratch/project_2001970/Geometry/aligned_models/intento/align_BERT_2_MTende/best_alignment_network.pt \
                                                        /scratch/project_2001970/Geometry/aligned_models/intento/align_BERT_2_MTdeen/best_alignment_network.pt \
                                                        /scratch/project_2001970/Geometry/aligned_models/intento/align_MTende_2_BERT/best_alignment_network.pt \
                                    --outdir $SAVEDATA --only_save_embs -hfmodels None -bert None \
                                    --prealigned_model_type bert bert en-de --cuda #--debug_mode

#srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=128GB --time=00:30:00 \
python compare_contextEmbeddings.py --data_path  ${datadir}/${langpair}.${tgtlang}   ${datadir}/${langpair}.en \
                                    --prealigned_models /scratch/project_2001970/Geometry/aligned_models/intento/align_MTdeen_2_BERT/best_alignment_network.pt \
                                    --outdir $SAVEDATA --only_save_embs -hfmodels None -bert None \
                                    --prealigned_model_type de-en --cuda



#srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=128GB --time=04:00:00 \
python compare_contextEmbeddings.py --load_embeddings_path $SAVEDATA/embeddings --outdir $SAVEDATA 

echo "moving logfiles"
cd $curdir
sleep 2

mv err_${SLURM_JOBID} saveEmbs_europarl_2nd_${SLURM_JOBID}_err
mv out_${SLURM_JOBID} saveEmbs_europarl_2nd_${SLURM_JOBID}_out