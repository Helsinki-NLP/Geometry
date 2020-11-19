#! bin/bash/

#USAGE: 
#            source /projappl/project_2001970/Geometry/europarl_align_models.sh 

source  /projappl/project_2001970/Geometry/env/bin/activate
cd /projappl/project_2001970/Geometry/code

outdir=/scratch/project_2001970/Geometry/aligned_models/europarl
outdir=/scratch/project_2001970/Geometry/aligned_models/intento # DELETE

datadir=/scratch/project_2001970/Geometry/data
mkdir -p $outdir



tgtlang='de'


    ## ALIGN BERT embeddings to MT encoder embeddings (both in english)
    echo " ### aligning BERT embeddings to  MTen${tgtlang} encoder embeddings ### "
    #run --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
    srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
    python do_FT_alignment.py --data_path   ${datadir}/en-${tgtlang}.full.en      \
                              --model        bert-base-cased                                          \
                              --model_base   Helsinki-NLP/opus-mt-en-${tgtlang}                               \
                              --train_nsents 200000 \
                              --epochs 1 \
                              --load_aligned_model ${outdir}/align_BERT_2_MTen${tgtlang}/best_alignment_network.pt \
                              --only_run_tests # > ${outdir}/align_BERT_2_MTen${tgtlang}.out 2> ${outdir}/align_BERT_2_MTen${tgtlang}.err
                              

    ## ALIGN MT encoder embeddings to BERT embeddings  (both in english)
    echo " ### aligning MTen${tgtlang} encoder embeddings to BERT embeddings ### "
    #srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
    srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
    python do_FT_alignment.py --data_path   ${datadir}/en-${tgtlang}.full.en      \
                              --model        Helsinki-NLP/opus-mt-en-${tgtlang}                               \
                              --model_base   bert-base-cased                                          \
                              --train_nsents 200000 \
                              --epochs 1 \
                              --load_aligned_model  ${outdir}/align_MTen${tgtlang}_2_BERT/best_alignment_network.pt \
                              --only_run_tests    #> ${outdir}/align_MTen${tgtlang}_2_BERT.out 2> ${outdir}/align_MTen${tgtlang}_2_BERT.err 



    ## ALIGN BERT embeddings (english) to $tgtlang MT encoder embeddings 
    echo " ### aligning BERT embeddings to  MT${tgtlang}en encoder embeddings ### "
    #srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
    srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
    python do_FT_alignment.py --data_path      ${datadir}/en-${tgtlang}.full.en       \
                              --data_path_tgt  ${datadir}/en-${tgtlang}.full.${tgtlang}       \
                              --model           bert-base-cased                                           \
                              --model_base      Helsinki-NLP/opus-mt-${tgtlang}-en                                \
                              --alignment_file ${datadir}/alignments/en-${tgtlang}.intersect \
                              --train_nsents 200000 \
                              --epochs 1 \
                              --load_aligned_model  ${outdir}/align_BERT_2_MT${tgtlang}en/best_alignment_network.pt \
                              --only_run_tests  #> ${outdir}/align_BERT_2_MT${tgtlang}en.out 2> ${outdir}/align_BERT_2_MT${tgtlang}en.err


    ## ALIGN $tgtlang MT encoder embeddings to english BERT embeddings
    echo " ### aligning MT${tgtlang}en encoder embeddings to BERT embeddings ### "
    #srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
    srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
    python do_FT_alignment.py --data_path      ${datadir}/en-${tgtlang}.full.${tgtlang}       \
                              --data_path_tgt  ${datadir}/en-${tgtlang}.full.en       \
                              --model           Helsinki-NLP/opus-mt-${tgtlang}-en                                \
                              --model_base      bert-base-cased                                           \
                              --alignment_file ${datadir}/alignments/${tgtlang}-en.intersect \
                              --train_nsents 200000 \
                              --epochs 1 \
                              --only_run_tests \
                              --load_aligned_model  ${outdir}/align_MT${tgtlang}en_2_BERT/best_alignment_network.pt\
                              --only_run_tests   # > ${outdir}/align_MT${tgtlang}en_2_BERT.out 2> ${outdir}/align_MT${tgtlang}en_2_BERT.err


# TEST BLEU PERFORMANCE:
cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate
tgtlang=de
export PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}
export DATA_DIR=/scratch/project_2001970/Geometry/en_de
export MODELNAME=Helsinki-NLP/opus-mt-en-de
export MAX_LEN=128
export BS=32


srun --account=project_2001970 --time=02:30:00 --mem-per-cpu=12G --partition=gpu --gres=gpu:v100:1,nvme:16 \
srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=12G --partition=gputest --gres=gpu:v100:1,nvme:16 \
python BertMT_hybrid_simpletrain.py --do_predict \
        --data_dir $DATA_DIR \
        --test_batch_size $BS \
        --bert_type bert-base-cased \
        --load_aligned_BERT_path /scratch/project_2001970/Geometry/aligned_models/intento/align_BERT_2_MTende/best_alignment_network.pt





###################################################
###################################################
###################################################
###################################################
###################################################
#

langlist=('bg' 'cs' 'da' 'de' 'es' 'et' 'fi' 'fr' 'hu' 'it'                'nl'            'sk'      'sv'  'el' 'ro')

outdir=/scratch/project_2001970/Geometry/aligned_models/europarl
langlist=('de')
:'
for tgtlang in ${langlist[@]}; do

    ## ALIGN BERT embeddings to MT encoder embeddings (both in english)
    echo " ### aligning BERT embeddings to  MTen${tgtlang} encoder embeddings ### "
    #run --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
    srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
    python do_FT_alignment.py --data_path   ${datadir}/en-${tgtlang}.full.en      \
                              --model        bert-base-cased                                          \
                              --model_base   Helsinki-NLP/opus-mt-en-${tgtlang}                               \
                              --train_nsents 200000 \
                              --epochs 1 \
                              --outdir      ${outdir}/align_BERT_2_MTen${tgtlang} > ${outdir}/align_BERT_2_MTen${tgtlang}.out 2> ${outdir}/align_BERT_2_MTen${tgtlang}.err
                              

    ## ALIGN MT encoder embeddings to BERT embeddings  (both in english)
    echo " ### aligning MTen${tgtlang} encoder embeddings to BERT embeddings ### "
    #srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
    srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
    python do_FT_alignment.py --data_path   ${datadir}/en-${tgtlang}.full.en      \
                              --model        Helsinki-NLP/opus-mt-en-${tgtlang}                               \
                              --model_base   bert-base-cased                                          \
                              --train_nsents 200000 \
                              --epochs 1 \
                              --outdir      ${outdir}/align_MTen${tgtlang}_2_BERT  > ${outdir}/align_MTen${tgtlang}_2_BERT.out 2> ${outdir}/align_MTen${tgtlang}_2_BERT.err 



    ## ALIGN BERT embeddings (english) to $tgtlang MT encoder embeddings 
    echo " ### aligning BERT embeddings to  MT${tgtlang}en encoder embeddings ### "
    #srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
    srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
    python do_FT_alignment.py --data_path      ${datadir}/en-${tgtlang}.full.en       \
                              --data_path_tgt  ${datadir}/en-${tgtlang}.full.${tgtlang}       \
                              --model           bert-base-cased                                           \
                              --model_base      Helsinki-NLP/opus-mt-${tgtlang}-en                                \
                              --alignment_file ${datadir}/alignments/en-${tgtlang}.intersect \
                              --train_nsents 200000 \
                              --epochs 1 \
                              --outdir         ${outdir}/align_BERT_2_MT${tgtlang}en > ${outdir}/align_BERT_2_MT${tgtlang}en.out 2> ${outdir}/align_BERT_2_MT${tgtlang}en.err


    ## ALIGN $tgtlang MT encoder embeddings to english BERT embeddings
    echo " ### aligning MT${tgtlang}en encoder embeddings to BERT embeddings ### "
    #srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
    srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
    python do_FT_alignment.py --data_path      ${datadir}/en-${tgtlang}.full.${tgtlang}       \
                              --data_path_tgt  ${datadir}/en-${tgtlang}.full.en       \
                              --model           Helsinki-NLP/opus-mt-${tgtlang}-en                                \
                              --model_base      bert-base-cased                                           \
                              --alignment_file ${datadir}/alignments/${tgtlang}-en.intersect \
                              --train_nsents 200000 \
                              --log_every_n_align 500\
                              --epochs 1 \
                              --outdir         ${outdir}/align_MT${tgtlang}en_2_BERT  > ${outdir}/align_MT${tgtlang}en_2_BERT.out 2> ${outdir}/align_MT${tgtlang}en_2_BERT.err
done


'
