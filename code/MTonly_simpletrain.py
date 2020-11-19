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
                                       



def report_metrics(metrics, type_path):
    if 'preds' in metrics[type_path][-1].keys(): 
        pred_metrics = {'STEP': metrics[type_path][-1]['STEP'], 'preds':metrics[type_path][-1].pop('preds')}
        #if 'target' in metrics[type_path][-1].keys():
        pred_metrics.update(targets=metrics[type_path][-1].pop('target'))
        save_json(pred_metrics, metrics_save_path.with_name('latest_gen_utterances.json'))
        with open(metrics_save_path.with_name('preds.txt'),'w') as f:
            for sent in pred_metrics['preds']:  
                f.write(sent) 
                f.write('\n')
        with open(metrics_save_path.with_name('targets.txt'),'w') as f:
            for sent in pred_metrics['targets']:  
                f.write(sent) 
                f.write('\n')

    print(type_path, metrics[type_path][-1],flush=True)
    print('')    
    save_json(metrics, metrics_save_path)

def get_dataloader2(
    model,
    tokenizer,
    type_path: str, 
    batch_size: int, 
    max_source_length:int,
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
            max_source_length=max_source_length,
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


def weights_init(m):
    if isinstance(m,(transformers.modeling_bart.Attention, transformers.modeling_bart.DecoderLayer, torch.nn.modules.container.ModuleList, transformers.modeling_bart.EncoderLayer)):
        for item in m.children():
            item.apply(weights_init)
    elif isinstance(m,torch.nn.modules.normalization.LayerNorm):
        torch.nn.init.uniform_(m.weight.data)
    elif isinstance(m,torch.nn.modules.linear.Identity):
        pass
    else: 
        torch.nn.init.xavier_uniform_(m.weight.data)


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

    print('\nFINETUNNING THE MT SYSTEM... BERT IS NOT INVOLVED IN THIS PART\n')
    mt_tokenizer = transformers.MarianTokenizer.from_pretrained(args.mt_mname) 
    MODEL = transformers.MarianMTModel.from_pretrained(args.mt_mname)
    MODEL.dataset_class = MarianNMTDataset
    MODEL = MODEL.to(device)
    
    if args.reinit_decoder:
        print('Random re-initialization of MT decoder parameters. SEED: 12345')
        torch.manual_seed(12345)
        for m in MODEL.model.decoder.children():
            weights_init(m)
    if args.reinit_encoder:
        print('Random re-initialization of MT encoder parameters. SEED: 12345')
        torch.manual_seed(12345)
        for m in MODEL.model.encoder.children():
            weights_init(m)

    optimizer = transformers.AdamW(
            filter(lambda p: p.requires_grad, MODEL.parameters()),
            lr=args.learning_rate, 
            eps=args.adam_epsilon 
        )

    # define loss    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mt_tokenizer.pad_token_id)
    

    # load data
    dataset_kwargs: dict = dict(
            data_dir=args.data_dir,
            prefix=MODEL.config.prefix or "",
        )


    train_loader = get_dataloader2(
            MODEL,
            mt_tokenizer,
            type_path="train",
            batch_size=args.train_batch_size, 
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length, 
            n_obs=args.n_train,
            num_workers = args.num_workers,
            shuffle=True, 
            **dataset_kwargs
        )

    val_loader = get_dataloader2(
            MODEL,
            mt_tokenizer,
            type_path="val",   
            batch_size=args.eval_batch_size, 
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length, 
            n_obs=args.n_val,
            num_workers = args.num_workers,
            shuffle=False,
            **dataset_kwargs
        )

    allmetrics = {}
    
    # --------------------------
    # TRAIN and VALIDATION loop
    # --------------------------
    if args.do_train:
        val_stepcount=0
        allmetrics.update(val=list())
        # VALIDATION
        
        print(f'Running validation for the {val_stepcount}-th time')
        MODEL.eval()
        base_metrics = val_routine_MT(
                MODEL, 
                mt_tokenizer,
                val_loader, 
                loss_fn, 
                device, 
                val_stepcount,
                return_val_predictions=True,
                limit_val_batches=args.limit_val_batches
            )
        allmetrics['val'].append(base_metrics)
        report_metrics(allmetrics, 'val')
        
        print('\n\nStarting train+validation routine', flush=True)
        for epoch in range(args.max_epochs):       
            # TRAIN:
            MODEL.train()
            print(f'\n...starting epoch {epoch+1}/{args.max_epochs}', flush=True) 
            losses = list()
            for batch in train_loader:
                batch = {k:v.to(device) for k,v in batch.items()}
                lm_labels = batch["labels"].clone()
                
                # s1. forward 
                outputs = MODEL(return_dict=True,**batch) 

                # s2. compute objective fn
                loss = outputs['loss']
                #loss = loss_fn(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1) )
                
                
                # s3. cleaning the gradient
                MODEL.zero_grad()

                # s4. accumulate the partial derivatives of the loss w.r.t. parameters
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
                    MODEL.eval()
                    val_stepcount+=1
                    print(f'Running validation for the {val_stepcount}-th time, at train step {len(losses)} since there are {len(train_loader)} batches and --val_check_interval={args.val_check_interval}', flush=True)
                    base_metrics = val_routine_MT(
                            MODEL, 
                            mt_tokenizer,
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

                    MODEL.train()
            # logging
            print(f'Epoch {epoch+1}, train_loss: {torch.tensor(losses).mean():3f}', flush=True)

    # -----
    # TEST
    # -----        
    if args.do_predict:
        allmetrics.update(test=list())
        test_loader = get_dataloader2(
                MODEL,
                mt_tokenizer,
                type_path="test",   
                batch_size=args.test_batch_size, 
                max_source_length=args.max_source_length,
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
        MODEL.eval()

        base_metrics = val_routine_MT(
                MODEL, 
                mt_tokenizer,
                test_loader, 
                loss_fn, 
                device,
                0, 
                return_val_predictions=True,
                limit_val_batches=args.limit_test_batches
            )
        allmetrics['test'].append(base_metrics)
        preds: List[str] = base_metrics['preds']                                                         
        target: List[str] = base_metrics['target']
        report_metrics(allmetrics, 'test')
        print('computing BLEU on the full test set - the above BLEU is a proxy, it is an average of the BLEU score per batch.')
        bleu=calculate_bleu(preds, target)
        allmetrics[f'BLEU_on_full_test_set'] = [bleu]
        print(f'BLEU = {bleu}')
        report_metrics(allmetrics, 'BLEU_on_full_test_set')
        import sys
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser = pl.Trainer.add_argparse_args(parser) 
    parser = BertTranslator.add_model_specific_args(parser) 
    
    parser.add_argument("--do_align", action="store_true", help="Whether to run Cao's alignment method before training.")
    parser.add_argument("--num_sents_align", type=int, default=50000, help="Number of sentences used in Cao's alignment method.")
    parser.add_argument("--num_epochs_align", type=int, default=10, help="Number of epochs for learning Cao's alignment method.")
    parser.add_argument("--log_every_n_align", type=int, default=100, help="How often to report results when doing Cao's alignment method.")
    parser.add_argument("--load_aligned_BERT_path", type=str, help="Path to an aligned Bert state_dict.")
    
    parser.add_argument("--freeze_decoder", action="store_true", help="freeze decoder parameters during FTing.")
    parser.add_argument("--freeze_bert", action="store_true", help="freeze BERT parameters during FTing.")
    parser.add_argument("--freeze_embeddings", action="store_true", help="freeze embedding layer parameters during FTing.")
    parser.add_argument("--reinit_decoder", action="store_true", help="random re-initialization of the MT decoder parameters.")
    parser.add_argument("--reinit_encoder", action="store_true", help="random re-initialization of the MT encoder parameters.")
    
    args = parser.parse_args() 
    #args.gpus=0  
    print(f'train bsz: {args.train_batch_size}, eval bsz: {args.eval_batch_size}, test bsz: {args.test_batch_size}')          
    import ipdb
    with ipdb.launch_ipdb_on_exception():                                             
        main(args)
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



# only TEST (mustc testset)
srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=20G --partition=gputest --gres=gpu:v100:1,nvme:16 \
    python MTonly_simpletrain.py \
        --do_predict \
        --data_dir $DATA_DIR \
        --test_max_target_length $MAX_LEN \
        --test_batch_size $BS \
        --gpus 1 \
        --output_dir ./outputs/MTonly_mustc \
        --mt_mname ${MODELNAME}

# only TEST (newstest2014)
srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=20G --partition=gputest --gres=gpu:v100:1,nvme:16 \
    python MTonly_simpletrain.py \
        --do_predict \
        --data_dir ${DATA_DIR}_newstest \
        --test_max_target_length $MAX_LEN \
        --test_batch_size $BS \
        --gpus 1 \
        --output_dir ./outputs/MTonly_newstest \
        --mt_mname ${MODELNAME}

# FINE-TUNE
export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}/onlyMT
mkdir -p ${outdir}
echo -e "RUNNING ROUTINE WITHOUT ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "

#srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=20G --partition=gputest --gres=gpu:v100:1,nvme:16 \
srun --account=project_2001970 --time=05:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
srun --account=project_2001970 --time=00:15:00 --mem-per-cpu=20G --partition=gputest --gres=gpu:v100:1,nvme:16 \
    python MTonly_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 1 \
        --do_predict \
        --val_check_interval 0.0004 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --mt_mname ${MODELNAME} > ${outdir}/FT_alignment.out 2> ${outdir}/FT_alignment.err 

# FINE-TUNE WITH RANDOMLY INITIALIZED DECODER
export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}/onlyMT_trainDecFromScratch
mkdir -p ${outdir}
echo -e "RUNNING ROUTINE WITHOUT ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
srun --account=project_2001970 --time=20:00:00 --mem-per-cpu=42G --partition=gpu --gres=gpu:v100:1,nvme:32 \
    python MTonly_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 30 \
        --do_predict \
        --val_check_interval 1 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size 64 --eval_batch_size 64 --test_batch_size 64 \
        --output_dir ${outdir} \
        --gpus 1 \
        --mt_mname ${MODELNAME} \
        --reinit_decoder         > ${outdir}/FT_MTonly.out 2> ${outdir}/FT_MTonly.err 

# FINE-TUNE WITH RANDOMLY INITIALIZED DECODER - BIGGER BSZ
export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}/onlyMT_trainDecFromScratch
mkdir -p ${outdir}
echo -e "RUNNING ROUTINE WITHOUT ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
srun --account=project_2001970 --time=25:00:00 --mem-per-cpu=42G --partition=gpu --gres=gpu:v100:1,nvme:32 \
    python MTonly_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 50 \
        --do_predict \
        --val_check_interval 1 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length 128 --max_target_length 128 --val_max_target_length 128 --test_max_target_length 128 \
        --train_batch_size 128 --eval_batch_size 128 --test_batch_size 128 \
        --output_dir ${outdir} \
        --gpus 1 \
        --mt_mname ${MODELNAME} \
        --reinit_decoder         > ${outdir}/FT_MTonly_bs128.out 2> ${outdir}/FT_MTonly_bs128.err 

# TRAIN SYSTEM FROM SCRATCH
export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}/onlyMT_trainEncDecFromScratch
mkdir -p ${outdir}
echo -e "RUNNING ROUTINE WITHOUT ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
srun --account=project_2001970 --time=10:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:32 \
    python MTonly_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 25 \
        --do_predict \
        --val_check_interval 1 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size 64 --eval_batch_size 64 --test_batch_size 64 \
        --output_dir ${outdir} \
        --gpus 1 \
        --mt_mname ${MODELNAME} \
        --reinit_decoder \
        --reinit_encoder       > ${outdir}/FT_MTonly.out 2> ${outdir}/FT_MTonly.err 


# FINE-TUNE WITH RANDOMLY INITIALIZED ENCODER
export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}/onlyMT_trainEncFromScratch
mkdir -p ${outdir}
echo -e "RUNNING ROUTINE WITHOUT ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
srun --account=project_2001970 --time=10:00:00 --mem-per-cpu=40G --partition=gpu --gres=gpu:v100:1,nvme:16 \
    python MTonly_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 25 \
        --do_predict \
        --val_check_interval 0.25 \
        --adam_eps 1e-06 \
        --data_dir $DATA_DIR \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size $BS --eval_batch_size $BS --test_batch_size $BS \
        --output_dir ${outdir} \
        --gpus 1 \
        --mt_mname ${MODELNAME} \
        --reinit_encoder       > ${outdir}/FT_MTonly.out 2> ${outdir}/FT_MTonly.err 


# FINE-TUNE WITH RANDOMLY INITIALIZED DECODER - BIGGER_DATASET
export outdir=/scratch/project_2001970/Geometry/BertMThybrid/testoutput/simpleTrainer_without_alignment/en-${tgtlang}/onlyMT_trainDecFromScratch
export DATA_DIR=/scratch/project_2001970/Geometry/en_de

mkdir -p ${outdir}
echo -e "RUNNING ROUTINE WITHOUT ALIGNMENT "
echo -e "   outputs will be stored in: ${outdir} "
srun --account=project_2001970 --time=20:00:00 --mem-per-cpu=42G --partition=gpu --gres=gpu:v100:1,nvme:32 \
    python MTonly_simpletrain.py \
        --learning_rate=3e-5 \
        --do_train \
        --num_train_epochs 30 \
        --val_check_interval 1 \
        --adam_eps 1e-06 \
        --data_dir /scratch/project_2001970/Geometry/data/en-de \
        --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
        --train_batch_size 64 --eval_batch_size 64 --test_batch_size 64 \
        --output_dir ${outdir} \
        --gpus 1 \
        --mt_mname ${MODELNAME} \
        --reinit_decoder         > ${outdir}/FTbiggerdata_MTonly.out 2> ${outdir}/FTbiggerdata_MTonly.err 

'''