#!/bin/bash
#SBATCH --job-name=Bert-MT
#SBATCH --account=project_2001970
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:16
#SBATCH -o out_%j
#SBATCH -e err_%j

 
# USAGE: 
#          cd /projappl/project_2001970/testoutput; sbatch /projappl/project_2001970/Geometry/BertMT_hybrid_train.sh de

cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate
tgtlang=$1 # de | ro
export PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}
export DATA_DIR=/scratch/project_2001970/Geometry/en_${tgtlang}
export MODELNAME=Helsinki-NLP/opus-mt-en-${tgtlang}
export MAX_LEN=128
export BS=32
export outdir=/projappl/project_2001970/testoutput/en-${tgtlang}


python BertMT_hybrid_train.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --val_check_interval 0.25 \
    --adam_eps 1e-06 \
    --num_train_epochs 6 \
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
