import argparse                                                         
from pathlib import Path                                                
import pytorch_lightning as pl
import numpy as np
import numpy as np
import torch                                                            
import time
import transformers                                                     
from tqdm import tqdm


#from utils.BertMT_hybrid import BertMT_hybrid                          
from utils.BertMT_hybrid import BertTranslator, BertSimpleTranslator


from torch.utils.data import DataLoader
from utils.hf_utils import (
    lmap, 
    save_json, 
    pickle_save, 
    flatten_list, 
    calculate_bleu,
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
def report_metrics(metrics):
    print(metrics, flush=True)
    save_json(metrics, metrics_save_path)

def val_routine(model, val_loader, loss_fn, device, return_val_predictions=False, limit_val_batches=None):
    val_losses = list()
    gen_times = list()
    summ_lens = list()
    bleus = list()
    predictions = list()
    targets = list()
    for batch in tqdm(val_loader):
        batch = {k:v.to(device) for k,v in batch.items()}
        lm_labels = batch["decoder_input_ids"].clone()
        
        # s1. forward
        with torch.no_grad():
            t0 = time.time()
            generated_ids = model.model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    MT_attention_mask=batch['MT_attention_mask'],
                    MT_input_ids=batch['MT_input_ids'],
                    token_type_ids=batch['token_type_ids'],
                    use_cache=True
                )
            gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
            preds: List[str] = model.ids_to_clean_text(generated_ids)
            target: List[str] = model.ids_to_clean_text(batch["decoder_input_ids"])   
            
            outputs = model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    MT_input_ids=batch['MT_input_ids'],
                    MT_attention_mask=batch['MT_attention_mask'], 
                    token_type_ids=batch['token_type_ids'],
                    decoder_input_ids=batch['decoder_input_ids'], 
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    use_cache=False
                )

            lm_logits = outputs[0]
            assert lm_logits.shape[-1] == model.model.config.vocab_size

            # s2. compute objective fn
            loss = loss_fn(
                    lm_logits.view(-1, lm_logits.shape[-1]), 
                    lm_labels.view(-1)
                )
            
        summ_len = np.mean(lmap(len, generated_ids))
        bleu=calculate_bleu(preds, target)
        val_losses.append(loss.item())
        gen_times.append(gen_time)
        summ_lens.append(summ_len)
        bleus.append(bleu['bleu'])
        predictions.append(preds)
        targets.append(target)
        
        if isinstance(limit_val_batches,int) and len(val_losses) > limit_val_batches:
            print(f'\nreached limit_val_batches={limit_val_batches}')
            break

    print('end of validation loop: ') 
    base_metrics = {'mean_val_bleu':np.mean(bleus)}
    base_metrics.update(
            mean_val_loss= np.mean(val_losses), 
            mean_gen_time=np.mean(gen_times), 
            mean_gen_len=np.mean(summ_lens), 
            
        )       
    report_metrics(base_metrics)
    if return_val_predictions:
        base_metrics.update(preds=predictions, target=targets)
        print(f'example of predictions on this validation loop: \n{predictions[:2]}')
        print(f'targets of those predicted sentences: \n{targets[:2]}')

    return base_metrics

def config_optimizer(
    model, 
    learning_rate, 
    adam_eps,
    freeze_decoder=False, 
    freeze_bert=False,
    ):
    for param in model.model.parameters():
        param.requires_grad = False
        
    if not freeze_decoder:
        for param in model.model.mt_model.model.decoder.parameters():
            param.requires_grad = True
 
    if not freeze_bert:
        for param in model.model.bert.parameters():
            param.requires_grad = True
        #optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    optimizer = AdamW(
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
        print("Using CUDA!")
        device='cuda'

    #TO DO: 1. Train a BERT model with the finetuning_alignment.py script from Cao (adapt it to have model=BERT and base_model=MTencoder)
    #       2. Modify BertTranslator s.t. if a bert model is passed, then the bert encoder starts with those parameters (Can I use `torch.nn.Module.load_state_dict`?)
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
                num_sent=args.num_sents_align,
                num_epochs=args.num_epochs_align,
                outdir=args.output_dir ,
                log_every_n_batches=args.log_every_n_align
            )

    model = BertSimpleTranslator(args)
    #model = BertTranslator(args) 
    
    if args.do_align or args.load_aligned_BERT_path:
        # initialize bert & linear projection with the aligned model
        model.model.bert.load_state_dict(alignedBERT_state_dict)
    
    model = model.to(device)
    # define optimizer
    #optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon )
    optimizer = config_optimizer(
    model, 
    learning_rate=args.learning_rate, 
    adam_eps=args.adam_epsilon,
    freeze_decoder=args.freeze_decoder, 
    freeze_bert=args.freeze_bert,
    args
    )

    # define loss
    loss_fn = model.model.loss_fct1

    # load data
    dataset_kwargs: dict = dict(
            data_dir=args.data_dir,
            max_source_length=args.max_source_length,
            prefix=model.model.config.prefix or "",
        )


    train_loader = get_dataloader(
            model,
            type_path="train",
            batch_size=args.train_batch_size, 
            max_target_length=args.max_target_length, 
            n_obs=args.n_train,
            num_workers = args.num_workers,
            shuffle=True, 
            **dataset_kwargs
        )

    val_loader = get_dataloader(
            model,
            type_path="val",   
            batch_size=args.eval_batch_size, 
            max_target_length=args.max_target_length, 
            n_obs=args.n_val,
            num_workers = args.num_workers,
            shuffle=False,
            **dataset_kwargs
        )

    # --------------------------
    # TRAIN and VALIDATION loop
    # --------------------------
    print('\n\nStarting train+validation routine', flush=True)
    for epoch in range(args.max_epochs):
        # VALIDATION
        print(f'Running validation loop.')
        model.eval()
        basic_metrics = val_routine(
                model, 
                val_loader, 
                loss_fn, 
                device, 
                return_val_predictions=True,
                limit_val_batches=args.limit_val_batches
            )

        # TRAIN:
        model.train()
        print(f'\n...starting epoch {epoch+1}/{args.max_epochs}', flush=True) 
        losses = list()
        for batch in tqdm(train_loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            lm_labels = batch["decoder_input_ids"].clone()
            
            # s1. forward
            outputs = model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    MT_input_ids=batch['MT_input_ids'],
                    MT_attention_mask=batch['MT_attention_mask'], 
                    token_type_ids=batch['token_type_ids'],
                    decoder_input_ids=batch['decoder_input_ids'], 
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    use_cache=False
                )

            lm_logits = outputs[0]
            assert lm_logits.shape[-1] == model.model.config.vocab_size

            # s2. compute objective fn
            loss = loss_fn(
                    lm_logits.view(-1, lm_logits.shape[-1]), 
                    lm_labels.view(-1)
                )
            coso = loss_fn( lm_logits.view(-1, lm_logits.shape[-1]),  lm_labels.view(-1)    )
            # s3. cleaning the gradient
            model.zero_grad()

            # s4. accumulate the partial derivatives fo the loss w.r.t. parameters
            loss.backward()

            # s5. make a step
            optimizer.step()   #with torch.no_grad(): model.params=model.params-eta*model.params.grad

            losses.append(loss.item())
            # logging
            if len(losses) % 100 == 0:
                print(f'step {len(losses)} of epoch {epoch+1}, train_loss: {torch.tensor(losses).mean():3f}')
            
            if isinstance(args.limit_train_batches,int) and len(losses) > args.limit_train_batches:
                print(f'reached limit_train_batches={args.limit_train_batches}')
                break
            
            if (len(losses) % round(len(train_loader)*args.val_check_interval)) == 0:
                model.eval()
                print(f'running validation at train step {len(losses)} since there are {len(train_loader)} batches and --val_check_interval={args.val_check_interval}', flush=True)
                basic_metrics = val_routine(
                        model, 
                        val_loader, 
                        loss_fn, 
                        device, 
                        return_val_predictions=True,
                        limit_val_batches=args.limit_val_batches
                    )
                model.train()
        # logging
        print(f'Epoch {epoch+1}, train_loss: {torch.tensor(losses).mean():3f}', flush=True)
        

        
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser = pl.Trainer.add_argparse_args(parser) 
    parser = BertTranslator.add_model_specific_args(parser) 
    
    parser.add_argument("--do_align", action="store_true", help="Whether to run Cao's alignment method before training.")
    parser.add_argument("--num_sents_align", type=int, default=50000, help="Number of sentences used in Cao's alignment method.")
    parser.add_argument("--num_epochs_align", type=int, default=10, help="Number of epochs for learning Cao's alignment method.")
    parser.add_argument("--log_every_n_align", type=int, default=100, help="How often to report results when doing Cao's alignment method.")
    parser.add_argument("--load_aligned_BERT_path", type=str, help="Path to an aligned Bert state_dict.")
    

    args = parser.parse_args() 
    #args.gpus=0  
    print(f'train bsz: {args.train_batch_size}, eval bsz: {args.eval_batch_size}, test bsz: {args.test_batch_size}')          
    import ipdb
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

aligned_BERT_Path=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/with_alignment/en-de/best_alignment_network.pt


export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_with_alignment/en-${tgtlang}
echo -e "RUNNING ROUTINE WITH ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
mkdir -p ${outdir}
srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python BertMT_hybrid_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --do_predict \
        --val_check_interval 0.0004 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --bert_type 'bert-base-uncased' \
        --mt_mname ${MODELNAME} \
        --load_aligned_BERT_path ${aligned_BERT_Path}


'''