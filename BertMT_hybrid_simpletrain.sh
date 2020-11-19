#!/bin/bash

cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate
tgtlang=de
export PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}
export DATA_DIR=/scratch/project_2001970/Geometry/en_${tgtlang}
export MODELNAME=Helsinki-NLP/opus-mt-en-${tgtlang}
export MAX_LEN=128
export BS=32


# LEAR ALIGNMENT MODEL
if false; then

    tgtlang=de
    outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/with_alignment/en-${tgtlang}
    srun --account=project_2001970 --time=10:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
        python BertMT_hybrid_simpletrain.py \
            --data_dir $DATA_DIR \
            --output_dir ${outdir} \
            --gpus 1 \
            --bert_type 'bert-base-uncased' \
            --mt_mname ${MODELNAME} \
            --do_align \
            --num_sents_align 75000 \
            --num_epochs_align 10 \
            --log_every_n_align 250 > ${outdir}/learn_alignment.out 2> ${outdir}/learn_alignment.err 

fi

# Experim. 1:
#           FINETUNE ALIGNED MODEL
tgtlang=de
aligned_BERT_Path=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/with_alignment/en-de/best_alignment_network.pt
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}
#srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=20G --partition=gputest --gres=gpu:v100:1,nvme:16 \
srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 1 \
        --do_predict \
        --val_check_interval 0.25 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} \
        --load_aligned_BERT_path ${aligned_BERT_Path} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err 
        


# Experim. 2:
#           FREEZE EMBEDDINGS LAYER OF BOTH BERT AND MT 
tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}/frozen_embLayers
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}
#srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=20G --partition=gputest --gres=gpu:v100:1,nvme:16 \
srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 1 \
        --do_predict \
        --val_check_interval 0.25 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} \
        --freeze_embeddings     \
        --load_aligned_BERT_path ${aligned_BERT_Path} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err 

# Experim. 3:
#            FREEZE BERT
tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}/frozenBERT
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}
#srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=20G --partition=gputest --gres=gpu:v100:1,nvme:16 \
srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 1 \
        --do_predict \
        --val_check_interval 0.25 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} \
        --freeze_bert     \
        --load_aligned_BERT_path ${aligned_BERT_Path} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err 

# Experim. 4:
#            FREEZE BERT AND EMBEDDINGS LAYER OF MT DECODER
tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}/frozenBERT_and_MTembLayer
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}
#srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=20G --partition=gputest --gres=gpu:v100:1,nvme:16 \
srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 1 \
        --do_predict \
        --val_check_interval 0.25 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} \
        --freeze_bert     \
        --freeze_embeddings \
        --load_aligned_BERT_path ${aligned_BERT_Path} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err 
        
# Experim. 5:
#            FREEZE MT DECODER
tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}/frozenMTdecoder
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}

srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 1 \
        --do_predict \
        --val_check_interval 0.25 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} \
        --freeze_decoder \
        --load_aligned_BERT_path ${aligned_BERT_Path} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err 

# Experim. 6:
#            FREEZE MT DECODER AND BERT EMBEDDINGS LAYER
tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}/frozenMTdecoder_and_BERTembLayer
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}

srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 1 \
        --do_predict \
        --val_check_interval 0.25 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} \
        --freeze_decoder \
        --freeze_embeddings \
        --load_aligned_BERT_path ${aligned_BERT_Path} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err

# Experim. 7:
#            TRAIN (non-aligned) HYBRID MODEL
tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}
BS=24
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}

srun --account=project_2001970 --time=10:00:00 --mem-per-cpu=35G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 6 \
        --do_predict \
        --val_check_interval 0.25 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err 

if false; then
# Experim. 8
#            TRAIN (non-aligned) HYBRID MODEL - RE-INITIALIZE MTdecoder PARAMETERS TO BE RANDOM
tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}/non_pretrained_decoder
BS=24
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}

srun --account=project_2001970 --time=20:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --reinit_decoder \
        --num_train_epochs 15 \
        --do_predict \
        --val_check_interval 0.5 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err 
fi

# EVALUATE
cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate
tgtlang=de
export PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}
export DATA_DIR=/scratch/project_2001970/Geometry/en_${tgtlang}
export MODELNAME=Helsinki-NLP/opus-mt-en-${tgtlang}
export MAX_LEN=128
export BS=32

loadfrom=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}
loadfrom2=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}

# TRANSLATE mustc_testset
srun --account=project_2001970 --time=02:30:00 --mem-per-cpu=12G --partition=gpu --gres=gpu:v100:1,nvme:16 \
python BertMT_hybrid_simpletrain.py --do_predict \
        --data_dir $DATA_DIR \
        --test_batch_size $BS \
        --load_pretrained_BertMT_path ${loadfrom}/best_finetuned_network.pt \
                                    ${loadfrom}/frozenBERT/best_finetuned_network.pt \
                                    ${loadfrom}/frozenBERT_and_MTembLayer/best_finetuned_network.pt \
                                    ${loadfrom}/frozen_embLayers/best_finetuned_network.pt \
                                    ${loadfrom}/frozenMTdecoder/best_finetuned_network.pt \
                                    ${loadfrom}/frozenMTdecoder_and_BERTembLayer/best_finetuned_network.pt \
                                    ${loadfrom2}/best_finetuned_network.pt \
                                    ${loadfrom2}/non_pretrained_decoder/best_finetuned_network.pt 


# TRANSLATE newstest2014
DATA_DIR=/scratch/project_2001970/Geometry/en_de_newstest
srun --account=project_2001970 --time=02:30:00 --mem-per-cpu=12G --partition=gpu --gres=gpu:v100:1,nvme:16 \
python BertMT_hybrid_simpletrain.py --do_predict \
        --data_dir $DATA_DIR \
        --test_batch_size $BS \
        --load_pretrained_BertMT_path ${loadfrom}/best_finetuned_network.pt \
                                    ${loadfrom}/frozenBERT/best_finetuned_network.pt \
                                    ${loadfrom}/frozenBERT_and_MTembLayer/best_finetuned_network.pt \
                                    ${loadfrom}/frozen_embLayers/best_finetuned_network.pt \
                                    ${loadfrom}/frozenMTdecoder/best_finetuned_network.pt \
                                    ${loadfrom}/frozenMTdecoder_and_BERTembLayer/best_finetuned_network.pt \
                                    ${loadfrom2}/best_finetuned_network.pt \
                                    ${loadfrom2}/non_pretrained_decoder/best_finetuned_network.pt > /projappl/project_2001970/Geometry/logs2/eval_newstest2014.out 2> /projappl/project_2001970/Geometry/logs2/eval_newstest2014.err

