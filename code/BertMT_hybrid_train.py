import argparse                                                         
from pathlib import Path                                                
import pytorch_lightning as pl                                          
import torch                                                            

import transformers                                                     

#from utils.BertMT_hybrid import BertMT_hybrid                          
from utils.BertMT_hybrid import BertTranslator                          

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




def main(args):
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

    trainer = pl.Trainer.from_argparse_args(args, 
        weights_summary=None, 
        logger=logger, 
        **train_params, 
    ) 
    trainer.fit(model) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser = pl.Trainer.add_argparse_args(parser) 
    parser = BertTranslator.add_model_specific_args(parser) 
      
    args = parser.parse_args() 
    args.gpus=0                                                         
    main(args)