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

def report_metrics(metrics, type_path):
    if 'preds' in metrics[type_path][-1].keys(): 
        pred_metrics = {'STEP': metrics[type_path][-1]['STEP'], 
                       'preds': metrics[type_path][-1].pop('preds'),
                     'targets': metrics[type_path][-1].pop('target') 
                    }
        save_json(pred_metrics, metrics_save_path.with_name('latest_gen_utterances.json'))
    
        with open(metrics_save_path.with_name('preds.txt'),'w') as f:
            for sent in pred_metrics['preds']:  
                f.write(sent) 
                f.write('\n')
        with open(metrics_save_path.with_name('targets.txt'),'w') as f:
            for sent in pred_metrics['targets']:  
                f.write(sent) 
                f.write('\n')

    if not type_path == 'val':
        for old_key in metrics[type_path][-1].keys():
            new_key = type_path.join(old_key.split('val')) if 'val' in old_key else old_key
            metrics[type_path][-1][new_key] = metrics[type_path][-1].pop(old_key)

    print(type_path, metrics[type_path][-1],flush=True)
    print('')    
    save_json(metrics, metrics_save_path)

def get_dataloader2(
    model,
    tokenizer,
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
            tokenizer=tokenizer ,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            prepare_translation_batch_function=tokenizer.prepare_seq2seq_batch,
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

def val_routine_MT(
    model,
    tokenizer, 
    val_loader, 
    loss_fn, 
    device, 
    val_stepcount, 
    return_val_predictions=False, 
    limit_val_batches=None
    ):
    val_losses = list()
    gen_times = list()
    summ_lens = list()
    bleus = list()
    predictions = list()
    targets = list()
    for batch in val_loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        lm_labels = batch["labels"].clone()
        
        # s1. forward
        with torch.no_grad():
            t0 = time.time()
            generated_ids = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=True
                )
            gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
            # todo: cambiar esta rutina... batch.decode no hace el truco para oraciones en el src-lang
            preds: List[str] = ids_to_clean_text(model, tokenizer, generated_ids)
            target: List[str] = ids_to_clean_text(model, tokenizer, batch["labels"])   
            
            outputs = model(return_dict=True,**batch) 
            #lm_logits = outputs['logits']
            #assert lm_logits.shape[-1] == model.config.vocab_size

            # s2. compute objective fn
            loss = outputs['loss']
            #loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1) )
            
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

    base_metrics = {'STEP':val_stepcount, 'mean_val_bleu':np.mean(bleus).round(decimals=8)}
    base_metrics.update(
            mean_val_loss=np.mean(val_losses).round(decimals=8), 
            mean_gen_time=np.mean(gen_times).round(decimals=3), 
            mean_gen_len=np.mean(summ_lens).round(decimals=3), 
            
        )       
    if return_val_predictions:
        base_metrics.update(preds=predictions, target=targets)

        #print(f'example of predictions on this validation loop: \n{predictions[:2]}')
        #print(f'targets of those predicted sentences: \n{targets[:2]}')

    return base_metrics

def val_routine(
    model, 
    val_loader, 
    loss_fn, 
    device, 
    val_stepcount, 
    return_val_predictions=False, 
    limit_val_batches=None
    ):
    val_losses = list()
    gen_times = list()
    summ_lens = list()
    bleus = list()
    predictions = list()
    targets = list()
    for batch in val_loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        lm_labels = batch["labels"].clone()
        
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
            target: List[str] = model.ids_to_clean_text(batch["labels"])   
            
            outputs = model(return_dict=True, use_cache=False,**batch)
            #lm_logits = outputs['logits']
            #assert lm_logits.shape[-1] == model.config.vocab_size

            # s2. compute objective fn
            loss = outputs['loss']
            #loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1) )
            
        summ_len = np.mean(lmap(len, generated_ids))
        bleu=calculate_bleu(preds, target)
        val_losses.append(loss.item())
        gen_times.append(gen_time)
        summ_lens.append(summ_len)
        bleus.append(bleu['bleu'])
        predictions += preds
        targets += target
        
        if isinstance(limit_val_batches,int) and len(val_losses) > limit_val_batches:
            print(f'\nreached limit_val_batches={limit_val_batches}')
            break

    base_metrics = {'STEP':val_stepcount, 'mean_val_bleu':np.mean(bleus).round(decimals=8)}
    base_metrics.update(
            mean_val_loss=np.mean(val_losses).round(decimals=8), 
            mean_gen_time=np.mean(gen_times).round(decimals=3), 
            mean_gen_len=np.mean(summ_lens).round(decimals=3), 
            
        )       
    if return_val_predictions:
        base_metrics.update(preds=predictions, target=targets)

        #print(f'example of predictions on this validation loop: \n{predictions[:2]}')
        #print(f'targets of those predicted sentences: \n{targets[:2]}')

    return base_metrics

def test_routine(
    model,
    test_loader, 
    loss_fn, 
    device,
    test_stepcount,
    limit_test_batches, 
    load_path=None
    ):
    if load_path:
        print(f'loading finetuned hybrid BERT-MT model from: {load_path}',flush=True)
        pretrained_state_dict = torch.load(load_path, map_location=torch.device(device))
        if 'state_dict' in pretrained_state_dict.keys():
            pretrained_state_dict = pretrained_state_dict['state_dict']  
            model.load_state_dict(pretrained_state_dict)

    model.eval()
    base_metrics = val_routine(
            model, 
            test_loader, 
            loss_fn, 
            device,
            test_stepcount, 
            return_val_predictions=True,
            limit_val_batches=limit_test_batches
        )

    return base_metrics


def weights_init(m):
    if isinstance(m,(transformers.modeling_bart.Attention, transformers.modeling_bart.DecoderLayer, torch.nn.modules.container.ModuleList)):
        for item in m.children():
            item.apply(weights_init)
    elif isinstance(m,torch.nn.modules.normalization.LayerNorm):
        torch.nn.init.uniform_(m.weight.data)
    elif isinstance(m,torch.nn.modules.linear.Identity):
        pass
    else: 
        torch.nn.init.xavier_uniform_(m.weight.data)

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
                num_dev=args.num_sents_evalalign,
                num_epochs=args.num_epochs_align,
                outdir=args.output_dir ,
                log_every_n_batches=args.log_every_n_align,
                validate_alignment=args.validate_alignment,
            )
    

    model = BertSimpleTranslator(args)
    #model = BertTranslator(args) 
        
    if args.do_align or args.load_aligned_BERT_path:
        # initialize bert & linear projection with the aligned model
        model.model.bert.load_state_dict(alignedBERT_state_dict)
        
    model = model.to(device)
    
    if args.reinit_decoder:
        print('Random re-initialization of MT decoder parameters. SEED: 12345')
        torch.manual_seed(12345)
        for m in model.model.mt_model.model.decoder.children():
            weights_init(m)

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
    if args.do_train:
        # load data
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
        # VALIDATION before TRAIN
        # --------------------------

        
        # VALIDATION
        """
        print(f'Running validation for the {val_stepcount}-th time')
        model.eval()
        base_metrics = val_routine(
                model, 
                val_loader, 
                loss_fn, 
                device, 
                val_stepcount,
                return_val_predictions=True,
                limit_val_batches=args.limit_val_batches
            )
        allmetrics['val'].append(base_metrics)
        report_metrics(allmetrics, 'val')
        """
        print('\n\nStarting train+validation routine', flush=True)
        # --------------------------
        # TRAIN and VALIDATION loop
        # --------------------------
        for epoch in range(args.max_epochs):       
            # -------
            # TRAIN:
            # -------
            model.train()
            print(f'\n...starting epoch {epoch+1}/{args.max_epochs}', flush=True) 
            losses = list()
            for batch in train_loader:
                batch = {k:v.to(device) for k,v in batch.items()}
                lm_labels = batch["labels"].clone()
                
                # s1. forward
                outputs = model(return_dict=True,**batch) 
                #lm_logits = outputs['logits']
                #assert lm_logits.shape[-1] == model.config.vocab_size

                # s2. compute objective fn
                loss = outputs['loss']
                #loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1) )
                
                # s3. cleaning the gradient
                model.zero_grad()

                # s4. accumulate the partial derivatives fo the loss w.r.t. parameters
                loss.backward()

                # s5. make a step
                optimizer.step()

                losses.append(loss.item())
                # logging
                if len(losses) % 250 == 0:
                    print(f'step {len(losses)} of epoch {epoch+1}, train_loss: {torch.tensor(losses).mean():3f}', flush=True)
                
                if isinstance(args.limit_train_batches,int) and len(losses) > args.limit_train_batches:
                    print(f'reached limit_train_batches={args.limit_train_batches}')
                    break
                
                if len(losses) == round(10000/args.train_batch_size):
                    print(f'saving into: {args.output_dir}/finetuned_network_step_10k.pt')            
                    #torch.save({'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, f'{args.output_dir}/finetuned_network_step_10k.pt')    
       
                # -----------
                # VALIDATION
                # -----------
                if (len(losses) % int(len(train_loader)*args.val_check_interval)) == 0:
                    model.eval()
                    val_stepcount+=1
                    print(f'Running validation for the {val_stepcount}-th time, at train step {len(losses)} since there are {len(train_loader)} batches and --val_check_interval={args.val_check_interval}', flush=True)
                    base_metrics = val_routine(
                            model, 
                            val_loader, 
                            loss_fn, 
                            device,
                            val_stepcount, 
                            return_val_predictions=True,
                            limit_val_batches=args.limit_val_batches
                    )
                    allmetrics['val'].append(base_metrics)
                    #print('end of validation loop: ', flush=True) 
                    report_metrics(allmetrics, 'val')

                    model.train()
            # logging
            print(f'Epoch {epoch+1}, train_loss: {torch.tensor(losses).mean():3f}', flush=True)
            print(f'saving into: {args.output_dir}/alignment_network_ckpt_{epoch}.pt', flush=True)            
            torch.save({'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict(),}, f'{args.output_dir}/finetune_network_ckpt_{epoch}.pt')
           
        print(f'saving into: {args.output_dir}/best_alignment_network.pt')            
        torch.save({'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),}, f'{args.output_dir}/best_finetuned_network.pt')    
       

    if args.do_predict:
        # -----
        # TEST
        # -----
        test_loader = get_dataloader(
                model,
                type_path="test",   
                batch_size=args.test_batch_size, 
                max_target_length=args.max_target_length, 
                n_obs=args.n_test,
                num_workers = args.num_workers,
                shuffle=False,
                **dataset_kwargs
            )

        # --------------------------
        # 
        # --------------------------
        # TEST
        print(f'Running prediction routine')
                
        if args.do_train:
            base_metrics = test_routine(model, test_loader, loss_fn, device,  test_stepcount, args.limit_test_batches, None)
            allmetrics['test'].append(base_metrics)
            preds: List[str] = base_metrics['preds']                                                         
            target: List[str] = base_metrics['target'] 
            report_metrics(allmetrics, 'test')

            print('computing BLEU on the full test set - the above BLEU is a proxy, it is an average of the BLEU score per batch.')
            bleu=calculate_bleu(preds, target)
            allmetrics['BLEU_on_full_test_set'] = [bleu]
            report_metrics(allmetrics, 'test')

        elif args.load_pretrained_BertMT_path:
            for path in args.load_pretrained_BertMT_path:
                base_metrics = test_routine(model, test_loader, loss_fn, device,  test_stepcount, args.limit_test_batches, path)
                name='_'.join(Path(path).parts[-3:-1])

                allmetrics[f'test_{name}']=list()
                allmetrics[f'test_{name}'].append(base_metrics)
                preds: List[str] = base_metrics['preds']                                                         
                target: List[str] = base_metrics['target'] 
                report_metrics(allmetrics, f'test_{name}')
                print('computing BLEU on the full test set - the above BLEU is a proxy, it is an average of the BLEU score per batch.')
                bleu=calculate_bleu(preds, target)
                allmetrics[f'BLEU_on_full_test_set_{name}'] = [bleu]
                report_metrics(allmetrics, f'BLEU_on_full_test_set_{name}')

        else:
            base_metrics = test_routine(model, test_loader, loss_fn, device,  test_stepcount, args.limit_test_batches, None)
            allmetrics['test'].append(base_metrics)
            preds: List[str] = base_metrics['preds']                                                         
            target: List[str] = base_metrics['target'] 
            report_metrics(allmetrics, 'test')

            print('computing BLEU on the full test set - the above BLEU is a proxy, it is an average of the BLEU score per batch.')
            bleu=calculate_bleu(preds, target)
            print(f'BLEU = {bleu}')
            allmetrics['BLEU_on_full_test_set'] = [bleu]
            report_metrics(allmetrics, 'test')


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
    
    #parser.add_argument("--freeze_decoder", action="store_true", help="freeze decoder parameters during FTing.")
    #parser.add_argument("--freeze_bert", action="store_true", help="freeze BERT parameters during FTing.")
    #parser.add_argument("--freeze_embeddings", action="store_true", help="freeze embedding layer parameters during FTing.")
    parser.add_argument("--reinit_decoder", action="store_true", help="random re-initialization of the MT decoder parameters.")
    
    args = parser.parse_args() 
    #args.gpus=0  
    print(f'train bsz: {args.train_batch_size}, eval bsz: {args.eval_batch_size}, test bsz: {args.test_batch_size}')          
    
    #import ipdb
    #with ipdb.launch_ipdb_on_exception():                                             
    #    main(args)
    main(args)


