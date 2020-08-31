import argparse                                                         
from pathlib import Path                                                
import pytorch_lightning as pl                                          
import torch                                                            

import transformers                                                     

#from utils.BertMT_hybrid import BertMT_hybrid                          
from utils.BertMT_hybrid import BertTranslator                          

''' U S A G E:
cd  /projappl/project_2001970/Geometry/code
source /projappl/project_2001970/Geometry/env/bin/activate
tgtlang=de
export PYTHONPATH=/scratch/project_2001970/transformers:/scratch/project_2001970/transformers/examples:${PYTHONPATH}
export ENRO_DIR=/scratch/project_2001970/Geometry/wmt_en_ro
export DATA_DIR=/scratch/project_2001970/Geometry/mustc_en_de
export MODELNAME=Helsinki-NLP/opus-mt-en-${tgtlang}
export MAX_LEN=128
export trainBS=16
export valBS=8
export outdir=/projappl/project_2001970/testoutput
srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1 --mem=8GB \
  python BertMT_hybrid_train.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --val_check_interval 0 \
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
    --fast_dev_run

'''
use_cuda=False                                                          


def reload_model(): 
    import utils.BertMT_hybrid 
    from importlib import reload  
    reload(utils.BertMT_hybrid) 
    #from utils.BertMT_hybrid import BertMT_hybrid 
    #model = BertMT_hybrid(config = dummyconfig, bert_type=bert_type, mt_mname=mt_mname, use_cuda=use_cuda) 
    from utils.BertMT_hybrid import BertTranslator 
    model = BertTranslator(args) 
    return model 


#model = reload_model()                                                 


def get_checkpoint_callback(output_dir, metric):
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
    Path(args.output_dir).mkdir(exist_ok=True)
    dummyconfig=transformers.PretrainedConfig.from_pretrained(args.mt_mname)
    dummyconfig.architectures = ["BertMT_hybrid"] 
    dummyconfig.encoder_attention_heads = 12 
    dummyconfig.encoder_layers = 12 
    dummyconfig.model_type = '' 
    dummyconfig.output_hidden_states = args.output_hidden_states 
    dummyconfig.output_attentions = args.output_attentions 
    args.config = transformers.PretrainedConfig.from_dict(dummyconfig.to_dict()) 
      
    model = BertTranslator(args) 

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
      
    args = parser.parse_args() 
    #args.gpus=0  
    print(f'train bsz: {args.train_batch_size}, eval bsz: {args.eval_batch_size}, test bsz: {args.test_batch_size}')          
    import ipdb
    with ipdb.launch_ipdb_on_exception():                                             
        main(args)