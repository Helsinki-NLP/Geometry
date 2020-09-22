#!/bin/bash
#SBATCH --job-name=Bert-MT
#SBATCH --account=project_2001970
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:16
#SBATCH -o out_%j
#SBATCH -e err_%j

 
# USAGE: 
#          cd /scratch/project_2001970/Geometry/BertMThybrid/testoutput; sbatch /projappl/project_2001970/Geometry/BertMT_hybrid_train.sh de
curdir=`pwd`
cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate
tgtlang=$1 # de | ro
export PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}
export DATA_DIR=/scratch/project_2001970/Geometry/en_${tgtlang}
export MODELNAME=Helsinki-NLP/opus-mt-en-${tgtlang}
export MAX_LEN=128
export BS=16



export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/with_alignment/en-${tgtlang}
echo -e "RUNNING ROUTINE WITH ALIGNMENT \n"
echo -e "   outputs will be stored in: ${outdir} \n"

python BertMT_hybrid_train.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --do_align \
    --val_check_interval 0.01 \
    --adam_eps 1e-06 \
    --num_train_epochs 2 \
    --data_dir $DATA_DIR \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
    --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
    --warmup_steps 500 \
    --freeze_embeds \
    --output_dir ${outdir} \
    --gpus 1 \
    --label_smoothing 0.1 \
    --bert_type 'bert-base-uncased' \
    --mt_mname ${MODELNAME} \
    --sortish_sampler  \
    --num_sents_align 50000 \
    --num_epochs_align 10 


echo -e '\n \n \n '
export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/without_alignment/en-${tgtlang}
echo -e "RUNNING ROUTINE WITHOUT ALIGNMENT \n"
echo -e "   outputs will be stored in: ${outdir} \n"
python BertMT_hybrid_train.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --val_check_interval 0.01 \
    --adam_eps 1e-06 \
    --num_train_epochs 2 \
    --data_dir $DATA_DIR \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
    --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
    --warmup_steps 500 \
    --freeze_embeds \
    --output_dir ${outdir} \
    --gpus 1 \
    --label_smoothing 0.1 \
    --bert_type 'bert-base-uncased' \
    --mt_mname ${MODELNAME} \
    --sortish_sampler  


echo "moving logfiles"
cd $curdir
sleep 2

mv err_${SLURM_JOBID} err_en-${tgtlang}_${SLURM_JOBID}
mv out_${SLURM_JOBID} out_en-${tgtlang}_${SLURM_JOBID}
