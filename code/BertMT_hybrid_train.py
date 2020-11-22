import argparse                                                         
from pathlib import Path                                                
import pytorch_lightning as pl
import numpy as np
import torch                                                            
import time
import transformers                                                     
from tqdm import tqdm


#from utils.BertMT_hybrid import BertMT_hybrid                          
from utils.BertMT_hybrid import BertTranslator, BertSimpleTranslator, MTtranslator


from torch.utils.data import DataLoader
from utils.hf_utils import (
    lmap, 
    save_json, 
    pickle_save, 
    flatten_list, 
    calculate_bleu,
)
''' U S A G E:
cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate
tgtlang=de
export PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}
export ENRO_DIR=/scratch/project_2001970/Geometry/en_ro
export DATA_DIR=/scratch/project_2001970/Geometry/en_de
export MODELNAME=Helsinki-NLP/opus-mt-en-${tgtlang}
export MAX_LEN=128
export trainBS=16
export valBS=8
export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput
srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
  python BertMT_hybrid_train.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --val_check_interval 0.33 \
    --adam_eps 1e-06 \
    --num_train_epochs 6 \
    --data_dir $DATA_DIR \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
    --train_batch_size=$trainBS --eval_batch_size=$valBS \
    --warmup_steps 500 \
    --freeze_embeds \
    --output_dir ${outdir} \
    --gpus 1 \
    --label_smoothing 0.1 \
    --bert_type 'bert-base-uncased' \
    --mt_mname ${MODELNAME} \
    --sortish_sampler  \
    --limit_train_batches 100 \
    --limit_val_batches 20 \
    --only_train_MT \
    --load_aligned_BERT_path /scratch/project_2001970/Geometry/BertMThybrid/testoutput/with_alignment/en-de/best_alignment_network.pt \
    # --num_sents_align 1000 --num_epochs_align 2 --log_every_n_align 25 \
    #--fast_dev_run   #--do_align 

'''                                                   

def load_4_aligning(
    sent_path,
    max_len = 64, 
    max_sent = 200000,
    ):
    '''
    This function emulates load_align_corpus from utils.FTalignment.py
    To use when wanting to encode english sentences with different models, 
    instead of encoding different language sentences with the same model
    '''
    import nltk
    sentences_1 = []
    bad_idx = []
    with open(sent_path) as sent_file:
        for i, line in enumerate(sent_file):
            if i >= max_sent:
                break
          
            sent_1 = nltk.TweetTokenizer().tokenize(line)    
            if len(sent_1) > max_len:
                bad_idx.append(i)
            else:
                sentences_1.append(sent_1)

        alignments = [   np.array([  [j,j] for j,_ in enumerate(sent)  ])   for sent  in sentences_1   ]
        return sentences_1, sentences_1, alignments


#model = reload_model()
def do_finetuning_alignment(
    model,
    model_base,
    data_dir,
    num_sent=50000, 
    bsz=16, 
    num_epochs=10,
    sent_path=None,
    align_path=None,
    outdir='./',
    log_every_n_batches=100,
    validate_alignment=False,
     ):
    '''
    Align embeddings of model with the embeddings of model_base using Cao's alignment method
    IN:
        - model      [str]: name of huggingface model to use (for now, only bert is supported) 
        - model_base [str]: name of huggingface model to use as anchor for alignment
        - sent_path  [str]: path to src-tgt sentences.
                             Lines should be of the form     doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
        - align_path [str]: path to alignement file. Only keeps 1-to-1 alignments. 
                            Lines should be of the form      0-0 1-1 2-4 3-2 4-3 5-5 6-6

    OUT:
        - aligned_state_dict [OrderedDict]: statedict of a BER model with new parameters
    '''
    from utils.FTalignment import WordLevelBert, WordLevelOPUSmt, align_bert_multiple, evaluate_retrieval
    import gc
    print(f'Running alignment routine:')
    print(f'        embeddings from model \"{model}\" are to be aligned with the ones from model \"{model_base}\" ')
    print(f'        using {num_sent} sentences, for {num_epochs} epochs ')
    print(f'        checkpoints will be saved to  {outdir}/alignment_network_ckpt_XX.pt \n', flush=True)

    
    model_base = WordLevelOPUSmt(model_base) # this should be the MT model for us
    model = WordLevelBert(model, do_lower_case=False, outdim=model_base.dim) # this is the one that will have the parameters updated

    data = load_4_aligning(f'{data_dir}/trainalignment.source', max_sent = num_sent)
    dev = None
    if validate_alignment:
        dev = load_4_aligning(f'{data_dir}/val.source')
        print("Word retrieval accuracy before alignment:", evaluate_retrieval(dev, model, model_base), flush=True)

    train = [data]#(sent_1, sent_2, align) for sent_1, sent_2, align in data]
    align_bert_multiple(
        train, 
        model, 
        model_base, 
        num_sentences=num_sent, 
        languages=['dummy'], 
        batch_size=bsz, 
        epochs=num_epochs, 
        outdir=outdir,
        log_every_n_batches=log_every_n_batches,
        devdata=dev
    )
   
    #free memory
    aligned_state_dict = model.state_dict()
    del model, model_base
    gc.collect()
    return aligned_state_dict

def get_checkpoint_callback(
    output_dir,
    metric,
    ):
    """Saves the best model by validation ROUGE2 score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2 and bleu, got {metric}, You can make your own by adding to this function."
        )



def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dummyconfig=transformers.PretrainedConfig.from_pretrained(args.mt_mname)
    dummyconfig.architectures = ["BertMT_hybrid"] 
    dummyconfig.encoder_attention_heads = 12 
    dummyconfig.encoder_layers = 12 
    dummyconfig.model_type = '' 
    dummyconfig.output_hidden_states = args.output_hidden_states 
    dummyconfig.output_attentions = args.output_attentions 
    args.config = transformers.PretrainedConfig.from_dict(dummyconfig.to_dict()) 
    
    device='cpu'
    if args.gpus >= 1:
        print(f'Using {args.gpus} CUDA device(s)')
        device='cuda'

    if args.only_train_MT:
        model = MTtranslator(args)
    else:
        if args.load_aligned_BERT_path:
            print(f'loading pre-aligned Bert embeddings from: {args.load_aligned_BERT_path}',flush=True)
            args.do_align = False # don't redo alignment
            alignedBERT_state_dict = torch.load(args.load_aligned_BERT_path, map_location=torch.device(device))
            if 'state_dict' in alignedBERT_state_dict.keys():
                alignedBERT_state_dict = alignedBERT_state_dict['state_dict']  


        if args.do_align: 
            alignedBERT_state_dict = do_finetuning_alignment(
                model=args.bert_type,
                model_base=args.mt_mname,
                data_dir = args.data_dir,
                num_sent=args.num_sents_align,
                num_epochs=args.num_epochs_align,
                outdir=args.output_dir ,
                log_every_n_batches=args.log_every_n_align,
                validate_alignment=args.validate_alignment,
            )

        #model = BertSimpleTranslator(args)
        model = BertTranslator(args) 
        
        if args.do_align or args.load_aligned_BERT_path:
            # initialize bert & linear projection with the aligned model
            model.model.bert.load_state_dict(alignedBERT_state_dict)
        
        if args.load_pretrained_BertMT_path:
            args.resume_from_checkpoint = args.load_pretrained_BertMT_path
            '''
            print(f'loading finetuned hybrid BERT-MT model from: {args.load_pretrained_BertMT_path}',flush=True)
            pretrained_state_dict = torch.load(load_path, map_location=torch.device(device))
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']  
                model.load_state_dict(pretrained_state_dict)
            '''
    #######################
    logger = True 
    #from pytorch_lightning.loggers import TensorBoardLogger
    #logger = TensorBoardLogger("tb_logs", name="my_hybrid_model")
    train_params = {} 
    if args.gpus and args.gpus > 1: 
        train_params["distributed_backend"] = "ddp" 


    pl.seed_everything(args.seed)
    
    metrics_save_path = Path(args.output_dir) / "MTonly_metrics.json"
    print(f'parameters and metrics to be saved in {metrics_save_path} \n')
    
    trainer = pl.Trainer.from_argparse_args(args, 
        weights_summary=None, 
        logger=logger, 
        default_root_dir=args.output_dir,
        **train_params, 
    ) 
    
    #trainer.tune(model)

    trainer.fit(model) 
    print(f'trainer.tested_ckpt_path: {trainer.tested_ckpt_path}')
    #############
    #pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model
    
    import glob 
    import os
    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    # test() without specifying a model tests using the best checkpoint automatically
    trainer.test()
    print(f'trainer.tested_ckpt_path: {trainer.tested_ckpt_path}')

    return model

    ########

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser = pl.Trainer.add_argparse_args(parser) 
    parser = BertTranslator.add_model_specific_args(parser) 
    
    parser.add_argument("--do_align", action="store_true", help="Whether to run Cao's alignment method before training.")
    parser.add_argument("--validate_alignment", action="store_true", help="If active, will use val.source from --data_dir to validate alignment method.")
    parser.add_argument("--num_sents_align", type=int, default=200000, help="Number of sentences used in Cao's alignment method.")
    parser.add_argument("--num_sents_evalalign", type=int, default=10000, help="Number of sentences to validate Caos' alignment method.")
    parser.add_argument("--num_epochs_align", type=int, default=10, help="Number of epochs for learning Cao's alignment method.")
    parser.add_argument("--log_every_n_align", type=int, default=100, help="How often to report results when doing Cao's alignment method.") 
    parser.add_argument("--load_aligned_BERT_path", type=str, help="Path to an aligned Bert state_dict.")
 

    parser.add_argument("--load_pretrained_BertMT_path", type=str, nargs='+', help="Path to a BertMT hybrid model state_dict.")
    parser.add_argument("--only_train_MT", action="store_true", help="Only train the MT model.")
    parser.add_argument("--reinit_MTdecoder", action="store_true", help="random re-initialization of the MT decoder parameters.")
    parser.add_argument("--reinit_MTencoder", action="store_true", help="random re-initialization of the MT encoder parameters.")
    
    
    args = parser.parse_args()  
    print(f'train bsz: {args.train_batch_size}, eval bsz: {args.eval_batch_size}, test bsz: {args.test_batch_size}')          
    #import ipdb
    #with ipdb.launch_ipdb_on_exception():                                             
    #    main(args)
    main(args)



'''
cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate
tgtlang=de
export PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}
export DATA_DIR=/scratch/project_2001970/Geometry/en_${tgtlang}
export MODELNAME=Helsinki-NLP/opus-mt-en-${tgtlang}
export MAX_LEN=128
export BS=16



export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}
srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 python BertMT_hybrid_train.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --do_align \
    --val_check_interval 0.0007 \
    --adam_eps 1e-06 \
    --num_train_epochs 2 \
    --data_dir $DATA_DIR \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
    --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
    --warmup_steps 500 \
    --freeze_embeds \
    --output_dir ${outdir} \
    --gpus 1 \
    --bert_type 'bert-base-uncased' \
    --mt_mname ${MODELNAME} \
    --num_sents_align 50000 \
    --num_epochs_align 6 \
    --log_every_n_align 250




# TRAIN ONLY MT FROM SCRATCH:

tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/pltrainer_without_alignment/en-${tgtlang}/onlyMT_trainEncDecFromScratch
MAX_LEN=128
DATA_DIR=/scratch/project_2001970/Geometry/en_${tgtlang}
PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}


cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate


#srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:32 --mem=20GB --time=00:15:00  \

srun --account=project_2001970 --time=20:00:00 --mem-per-cpu=42G --partition=gpu --gres=gpu:v100:1,nvme:32 \
 python BertMT_hybrid_train.py     \
 --learning_rate=3e-5               \
 --do_train   --do_predict  --num_train_epochs 50     \
 --max_source_length $MAX_LEN  --max_target_length $MAX_LEN  --val_max_target_length $MAX_LEN  --test_max_target_length $MAX_LEN   \
 --train_batch_size=128  --test_batch_size=128  --eval_batch_size=128  --accumulate_grad_batches 2  \
 --warmup_steps 5000  --lr_scheduler linear --adam_eps 1e-06 --adam_beta1 0.9 --adam_beta2 0.998  --weight_decay 1e-2   --label_smoothing 0.1        \
 --data_dir $DATA_DIR      \
 --output_dir ${outdir}     \
 --gpus 1     \
 --mt_mname Helsinki-NLP/opus-mt-en-de  \
 --sortish_sampler     \
 --reinit_MTencoder \
 --reinit_MTdecoder  \
 --only_train_MT      > ${outdir}/FT_MTonly.out 2> ${outdir}/FT_MTonly.err


# TRAIN only MT DECODER FROM SCRATCH:

tgtlang=de
outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/pltrainer_without_alignment/en-${tgtlang}/onlyMT_trainDecFromScratch
MAX_LEN=128
DATA_DIR=/scratch/project_2001970/Geometry/en_${tgtlang}
PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}

mkdir -p $outdir
cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate


#srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:32 --mem=20GB --time=00:15:00  \

srun --account=project_2001970 --time=20:00:00 --mem-per-cpu=42G --partition=gpu --gres=gpu:v100:1,nvme:32 \
 python BertMT_hybrid_train.py     \
 --learning_rate=3e-5               \
 --do_train   --do_predict  --num_train_epochs 50     \
 --max_source_length $MAX_LEN  --max_target_length $MAX_LEN  --val_max_target_length $MAX_LEN  --test_max_target_length $MAX_LEN   \
 --train_batch_size=128  --test_batch_size=128  --eval_batch_size=128  --accumulate_grad_batches 2  \
 --warmup_steps 5000  --lr_scheduler linear --adam_eps 1e-06 --adam_beta1 0.9 --adam_beta2 0.998  --weight_decay 1e-2   --label_smoothing 0.1        \
 --data_dir $DATA_DIR      \
 --output_dir ${outdir}     \
 --gpus 1     \
 --mt_mname Helsinki-NLP/opus-mt-en-de  \
 --sortish_sampler     \
 --reinit_MTdecoder  \
 --only_train_MT      > ${outdir}/FT_MTonly.out 2> ${outdir}/FT_MTonly.err
 
'''
