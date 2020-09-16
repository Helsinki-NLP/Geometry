import argparse                                                         
from pathlib import Path                                                
import pytorch_lightning as pl
import numpy as np
import torch                                                            

import transformers                                                     

#from utils.BertMT_hybrid import BertMT_hybrid                          
from utils.BertMT_hybrid import BertTranslator, BertSimpleTranslator

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
srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB \
  python BertMT_hybrid_train.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --do_align \
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
     --num_sents_align 1000 \
     --num_epochs_align 2\
     --log_every_n_align 25 \
     --fast_dev_run

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
    num_sent=50000, 
    bsz=16, 
    num_epochs=10,
    sent_path=None,
    align_path=None,
    outdir='./',
    log_every_n_batches=100,
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
    from utils.FTalignment import WordLevelBert, WordLevelOPUSmt, align_bert_multiple
    import gc
    print(f'Running alignment routine:')
    print(f'        embeddings from model \"{model}\" are to be aligned with the ones from model \"{model_base}\" ')
    print(f'        using {num_sent} sentences, for {num_epochs} epochs \n')

    
    model_base = WordLevelOPUSmt(model_base) # this should be the MT model for us
    model = WordLevelBert(model, False, model_base.dim) # this is the one that will have the parameters updated

    data = load_4_aligning(f'{args.data_dir}/train.source', max_sent = num_sent)
    #data = [load_align_corpus(sent_path, align_path, max_sent = num_sent) for sent_path, align_path in zip(sent_paths, align_paths)]
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
        log_every_n_batches=log_every_n_batches
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
    
    #TO DO: 1. Train a BERT model with the finetuning_alignment.py script from Cao (adapt it to have model=BERT and base_model=MTencoder)
    #       2. Modify BertTranslator s.t. if a bert model is passed, then the bert encoder starts with those parameters (Can I use `torch.nn.Module.load_state_dict`?)
    if args.do_align: 
        alignedBERT_state_dict = do_finetuning_alignment(
            model=args.bert_type,
            model_base=args.mt_mname,
            num_sent=args.num_sents_align,
            num_epochs=args.num_epochs_align,
            outdir=args.output_dir ,
            log_every_n_batches=args.log_every_n_align
        )

    model = BertSimpleTranslator(args)
    #model = BertTranslator(args) 
    
    if args.do_align:
        # initialize bert & linear projection with the aligned model
        model.model.bert.load_state_dict(alignedBERT_state_dict)
    logger = True 
    train_params = {} 
    if args.gpus and args.gpus > 1: 
        train_params["distributed_backend"] = "ddp" 


    pl.seed_everything(args.seed)
    
    trainer = pl.Trainer.from_argparse_args(args, 
        weights_summary=None, 
        logger=logger, 
        default_root_dir=args.output_dir,
        **train_params, 
    ) 

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

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    print(f'trainer.tested_ckpt_path: {trainer.tested_ckpt_path}')

    return model
    ########

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser = pl.Trainer.add_argparse_args(parser) 
    parser = BertTranslator.add_model_specific_args(parser) 
    
    parser.add_argument("--do_align", action="store_true", help="Whether to run Cao's alignment method before training.")
    parser.add_argument("--num_sents_align", type=int, default=50000, help="Number of sentences used in Cao's alignment method.")
    parser.add_argument("--num_epochs_align", type=int, default=10, help="Number of epochs for learning Cao's alignment method.")
    parser.add_argument("--log_every_n_align", type=int, default=100, help="Number of epochs for learning Cao's alignment method.")

    args = parser.parse_args() 
    #args.gpus=0  
    print(f'train bsz: {args.train_batch_size}, eval bsz: {args.eval_batch_size}, test bsz: {args.test_batch_size}')          
    #import ipdb
    #with ipdb.launch_ipdb_on_exception():                                             
    #    main(args)
    main(args)