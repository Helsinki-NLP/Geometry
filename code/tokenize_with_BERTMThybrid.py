import argparse                                                         
from pathlib import Path                                                
import pytorch_lightning as pl
import numpy as np
import numpy as np
import torch                                                            
import time
import transformers                                                     
from tqdm import tqdm
from typing import  List

#from utils.BertMT_hybrid import BertMT_hybrid                          
from utils.BertMT_hybrid import BertTranslator, BertSimpleTranslator


from torch.utils.data import DataLoader
from utils.hf_utils import (
    lmap, 
    save_json, 
    pickle_save, 
    flatten_list, 
    calculate_bleu,
    MarianNMTDataset,
)
                                       

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
    num_dev = 10000,
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

    print(f'Loading models:')    
    model_base = WordLevelOPUSmt(model_base) # this should be the MT model for us
    model = WordLevelBert(model, do_lower_case=False, outdim=model_base.dim) # this is the one that will have the parameters updated

    print(f'Loading data:') 
    data = load_4_aligning(f'{data_dir}/trainalignment.source.shf', max_sent = num_sent)
    dev = None
    if validate_alignment:
        dev = load_4_aligning(f'{data_dir}/val.source', max_sent = num_dev)
        #print("Word retrieval accuracy before alignment:", evaluate_retrieval(dev, model, model_base), flush=True)

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



def get_dataloader(
    model,
    type_path: str, 
    batch_size: int, 
    max_target_length: int, 
    n_obs: int,
    num_workers: int, 
    shuffle: bool = False,
    sampler=None,
    **kwargs,
    ) -> DataLoader:
    
    #from torch.utils.data import DataLoader
    dataset = model.dataset_class(
            tokenizer=model.model.bert_tokenizer ,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            prepare_translation_batch_function=model.model.prepare_translation_batch,
            **kwargs
        )  
    
        
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
        )
        
    return dataloader



metrics_save_path="metrics.json"


def ids_to_clean_text(
        model, 
        tokenizer,
        generated_ids,
        ):

        gen_text = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )

        return lmap(str.strip, gen_text)

def config_optimizer(
    model, 
    learning_rate, 
    adam_eps,
    freeze_decoder=False, 
    freeze_bert=False,
    freeze_embeddings=False,
    ):
    for param in model.model.parameters():
        param.requires_grad = False
        
    if not freeze_decoder:
        for param in model.model.mt_model.model.decoder.parameters():
            param.requires_grad = True
 
    if not freeze_bert:
        for param in model.model.bert.parameters():
            param.requires_grad = True

    if freeze_embeddings:
        for param in model.model.mt_model.model.decoder.embed_tokens.parameters() :
            param.requires_grad = False
        for param in model.model.bert.bert.embeddings.parameters():
            param.requires_grad = False        

        #optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    print(f'Optimizer will update parameters for the decoder:{not freeze_decoder} AND for bert:{not freeze_bert}. freeze_embeddings:{freeze_embeddings}')
    optimizer = transformers.AdamW(
            filter(lambda p: p.requires_grad, model.model.parameters()),
            lr=learning_rate, 
            eps=adam_eps 
        )
    return optimizer

def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    global metrics_save_path 
    metrics_save_path = Path(args.output_dir) / "metrics.json"
    dummyconfig=transformers.PretrainedConfig.from_pretrained(args.mt_mname)
    dummyconfig.architectures = ["BertMT_hybrid"] 
    dummyconfig.encoder_attention_heads = 12 
    dummyconfig.encoder_layers = 12 
    dummyconfig.model_type = '' 
    dummyconfig.output_hidden_states = args.output_hidden_states 
    dummyconfig.output_attentions = args.output_attentions 
    args.config = transformers.PretrainedConfig.from_dict(dummyconfig.to_dict()) 
    
    device='cpu'
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA!", flush=True)
        device='cuda'



    model = BertTranslator(args) 
        
    model = model.to(device)
    

    # define optimizer
    #optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon )
    optimizer = config_optimizer(
            model, 
            learning_rate=args.learning_rate, 
            adam_eps=args.adam_epsilon,
            freeze_decoder=args.freeze_decoder, 
            freeze_bert=args.freeze_bert,
            freeze_embeddings=args.freeze_embeddings,
        )

    # define loss
    loss_fn = model.model.loss_fct1

    dataset_kwargs: dict = dict(
            data_dir=args.data_dir,
            max_source_length=args.max_source_length,
            prefix=model.model.config.prefix or "",
        )

    val_stepcount = 0
    test_stepcount = 0
    allmetrics = {'val':list(),'test':list()}
    # load data
    train_loader = get_dataloader(
                model,
                type_path="train",
                batch_size=1, 
                max_target_length=args.max_target_length, 
                n_obs=args.n_train,
                num_workers = args.num_workers,
                shuffle=False, 
                **dataset_kwargs
            )


        
    print('\n\nTokenizing to find long sentences', flush=True)
    # -------
    # TRAIN:
    # -------
    model.train()
    bad_idx = list()
    for i,batch in enumerate(train_loader):
        if i % 25000 == 0:
            print(f'processed {i} ...')

        if batch['input_ids'].size(1)>=250:
            print(f'Sentence {i} is longer than threshold. Tokd length = {batch["input_ids"].shape}')
            bad_idx.append(i)

    print(f'There are {len(bad_idx)} too long sentences. They are:')
    print(bad_idx)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser = pl.Trainer.add_argparse_args(parser) 
    parser = BertTranslator.add_model_specific_args(parser) 
    
    parser.add_argument("--do_align", action="store_true", help="Whether to run Cao's alignment method before training.")
    parser.add_argument("--validate_alignment", action="store_true", help="If active, will use val.source from --data_dir to validate alignment method.")
    parser.add_argument("--num_sents_align", type=int, default=50000, help="Number of sentences used in Cao's alignment method.")
    parser.add_argument("--num_sents_evalalign", type=int, default=10000, help="Number of sentences to validate Caos' alignment method.")
    parser.add_argument("--num_epochs_align", type=int, default=10, help="Number of epochs for learning Cao's alignment method.")
    parser.add_argument("--log_every_n_align", type=int, default=100, help="How often to report results when doing Cao's alignment method.")
    parser.add_argument("--load_aligned_BERT_path", type=str, help="Path to an aligned Bert state_dict.")
    parser.add_argument("--load_pretrained_BertMT_path", type=str, nargs='+', help="Path to a BertMT hybrid model state_dict.")
    
    
    args = parser.parse_args() 
    #args.gpus=0  
    print(f'train bsz: {args.train_batch_size}, eval bsz: {args.eval_batch_size}, test bsz: {args.test_batch_size}')          
    
    #import ipdb
    #with ipdb.launch_ipdb_on_exception():                                             
    #    main(args)
    main(args)


