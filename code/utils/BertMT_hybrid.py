import transformers
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AdamW, 
    BertConfig, 
    BertForMaskedLM, 
    BertModel, 
    BertTokenizer, 
    PreTrainedModel, 
    PretrainedConfig
    )
from transformers.generation_utils import GenerationMixin
from typing import Dict, List, Optional, Tuple, Union, Callable, Iterable
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.hf_utils import label_smoothed_nll_loss, lmap, Seq2SeqDataset, TranslationDataset


class BertEncoder4Hybrid(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.
    Args:
        config: BartConfig
    """

    def __init__(self, bert_type='bert-base-uncased', outdim=512,  output_hidden_states=False):
        super().__init__()

        # BERT: 
        config = BertConfig.from_pretrained(bert_type, output_hidden_states=output_hidden_states)
        self.bert = BertModel.from_pretrained(bert_type, config=config)
        
        self.bert_encdim = self.bert.pooler.dense.in_features

        # LINEAR
        self.linear = nn.Linear(self.bert_encdim, outdim)

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, **kwargs):
        '''
        runs bert and a linear projection
        '''               
        encoder_outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states, **kwargs)

        #linear projection
        hstates=self.linear(encoder_outputs[0]) # [bsz, sentlen, mt_hdim]
        encoder_outputs = (hstates,) + encoder_outputs[1:]
        
        return encoder_outputs

class BertMT_hybrid(PreTrainedModel):
    """
    Encodes with BERT and decodes with specified MT model.
    """
    def __init__(self, 
        config, 
        bert_type='bert-base-uncased', 
        mt_mname='Helsinki-NLP/opus-mt-en-fi', 
        output_hidden_states=False):
        super().__init__(config)
        

        # MT:        
        #config_overrider={'output_attentions':True, 'output_hidden_states':True}
        config_overrider={'output_hidden_states':output_hidden_states}
        self.mt_tokenizer = transformers.MarianTokenizer.from_pretrained(mt_mname)    
        self.mt_model = transformers.MarianMTModel.from_pretrained(mt_mname, **config_overrider)
        self._prepare_bart_decoder_inputs = transformers.modeling_bart._prepare_bart_decoder_inputs
        self.adjust_logits_during_generation =  self.mt_model.adjust_logits_during_generation 
        self._reorder_cache =  self.mt_model._reorder_cache 
        
        # parameters:
        self.mt_hdim = self.mt_model.config.d_model # dimension of the model
        self.output_hidden_states=output_hidden_states
       
        # BERT: 
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.bert = BertEncoder4Hybrid(bert_type, self.mt_hdim)

        self.bert.linear.weight.data.normal_(mean=0.0, std=config.init_std)

        #config = BertConfig.from_pretrained(bert_type, output_hidden_states=output_hidden_states)        
        #self.bert = BertModel.from_pretrained(bert_type, config=config)

        self.bert_encdim = self.bert.bert_encdim

        self.loss_fct1 = nn.CrossEntropyLoss()
        self.loss_fct2 = label_smoothed_nll_loss

    def prepare_translation_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        pad_to_max_length: bool = True,
        return_tensors: str = "pt",
        truncation_strategy="only_first",
        padding="longest",
    ):
        
        bert_encoded_sentences = self.bert_tokenizer(src_texts, padding=True, return_tensors='pt')
        
        mt_encoded_sentences = self.mt_tokenizer.prepare_translation_batch(src_texts=src_texts, tgt_texts=tgt_texts)


        # overwrite some parameters_
        mt_encoded_sentences['MT_input_ids'] = mt_encoded_sentences['input_ids']
        mt_encoded_sentences['MT_attention_mask'] = mt_encoded_sentences['attention_mask']
        mt_encoded_sentences['input_ids'] = bert_encoded_sentences['input_ids']
        mt_encoded_sentences['attention_mask'] =  bert_encoded_sentences['attention_mask']
        mt_encoded_sentences['token_type_ids'] =  bert_encoded_sentences['token_type_ids']


        return mt_encoded_sentences

    def forward(
        self,
        input_ids,
        attention_mask=None,
        MT_input_ids = None,
        MT_attention_mask=None,
        token_type_ids=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **unused,
    ):

        
        if decoder_input_ids is None:
            use_cache = False
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = self._prepare_bart_decoder_inputs(
                config=self.config,
                input_ids=MT_input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask
            )
        else:
            decoder_padding_mask, causal_mask = None, None        

        assert decoder_input_ids is not None

        # run bert
        if encoder_outputs is None:
            bert_encoded_sentences = {'input_ids':input_ids, 'token_type_ids':token_type_ids, 'attention_mask':attention_mask}
            # encoder_outputs is a tuple: (last_hidden_state, pooler_output)
            encoder_outputs = self.bert(**bert_encoded_sentences) # ([bsz,sen_len,bert_encdim], tuple)
        
        
        # TODO: SHOULD WE RESHAPE THINGS TO AVOID THE EMBEDDING FOR THE "[CLS]" TOKEN NEEDED IN BERT ???
        # mt_decoder_outputs consists of (dec_features, layer_state, dec_hidden, dec_attn), given the configuration specs
        mt_decoder_outputs = self.mt_model.model.decoder(input_ids=decoder_input_ids, 
                                    encoder_hidden_states=encoder_outputs[0],
                                    encoder_padding_mask=attention_mask,
                                    decoder_padding_mask=decoder_padding_mask, 
                                    decoder_causal_mask=causal_mask,
                                    decoder_cached_states=decoder_cached_states,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    use_cache=use_cache,
                                ) 
        # Attention and hidden_states will be [] or None if they aren't needed
        mt_decoder_outputs: Tuple =  transformers.modeling_bart._filter_out_falsey_values(mt_decoder_outputs)
        
        outputs = mt_decoder_outputs + encoder_outputs

        # HUOM!: TAKEN FROM transformers.BartForConditionalGeneration forward()
        lm_logits = F.linear(mt_decoder_outputs[0], self.mt_model.model.shared.weight, bias=self.mt_model.final_logits_bias)
        outputs =  (lm_logits,) + outputs[1:]#, mt_decoder_outputs[0]
        if labels is not None:
            
            # TODO(SS): do we need to ignore pad tokens in labels?
            #masked_lm_loss = loss_fct(lm_logits.view(-1, self.mt_model.config.vocab_size), labels.view(-1))
            masked_lm_loss = self.loss_fct1(lm_logits.view(-1, self.mt_model.config.vocab_size), decoder_input_ids.view(-1))      
            outputs = (masked_lm_loss,) + outputs

        return  outputs 

    def get_output_embeddings(self):
        return self.mt_model.get_output_embeddings()

    def get_encoder(self):
        return self.bert
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs, decoder_cached_states = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "MT_attention_mask": kwargs['MT_attention_mask'],
            "MT_input_ids":kwargs['MT_input_ids'],
           "token_type_ids":kwargs['token_type_ids'],
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }







from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    #"polynomial": get_polynomial_decay_schedule_with_warmup, # not supported for now
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


#import sys
#sys.path.insert(0, '/scratch/project_2001970/transformers/examples')
#from lightning_base import BaseTransformer

class BertTranslator(pl.LightningModule):
    """
    Wraps the BertMT_hybrid model for using it with pytorch_lightning.
    """
    def __init__(self, args, **kwargs):
        #config,          bert_type='bert-base-uncased',         mt_mname='Helsinki-NLP/opus-mt-en-fi',          use_cuda=False,         output_hidden_states=False,        , **kwargs):

        super().__init__()
        self.model =  BertMT_hybrid(config = args.config, 
            bert_type=args.bert_type, 
            mt_mname=args.mt_mname, 
            output_hidden_states=args.config.output_hidden_states,
            **kwargs
        )
        self.hparams = args

        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        self.dataset_class = TranslationDataset
        
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids=input_ids, **kwargs)
    
    # huggingface functions
    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.model.mt_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

     # pytorch_lighting reserved functions: 
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        optimizer = AdamW(model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        
        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]
    
    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.train_loader.dataset)
        return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def setup(self, mode):
        if mode == "fit":
            self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.model.mt_tokenizer.pad_token_id
        #source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        #decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
        #lm_labels = target_ids[:, 1:].clone()  # why clone?

        lm_labels = batch["decoder_input_ids"].clone()
        
        outputs = self(input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            MT_input_ids=batch['MT_input_ids'],
            MT_attention_mask=batch['MT_attention_mask'], 
            token_type_ids=batch['token_type_ids'],
            decoder_input_ids=batch['decoder_input_ids'], 
            decoder_attention_mask=batch['decoder_attention_mask'],
            use_cache=False)


        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py
            loss_fct = self.loss_fct1(ignore_index=pad_token_id)
            lm_logits = outputs[0]
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, lm_labels, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        return (loss,)
    
    @property
    def bert_pad(self) -> int:
        return self.model.bert_tokenizer.pad_token_id

    @property
    def mt_pad(self) -> int:
        return self.model.bert_tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.bert_pad).sum() + batch["decoder_input_ids"].ne(self.mt_pad).sum()
        return {"loss": loss_tensors[0], "log": logs}
    
    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)
        sampler = None
        if self.hparams.sortish_sampler and type_path == "train":
            assert self.hparams.gpus <= 1  # TODO: assert earlier
            sampler = dataset.make_sortish_sampler(batch_size)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        if max(scheduler.get_last_lr()) > 0:
            warnings.warn("All learning rates are 0")
        self.lr_scheduler = scheduler
        return dataloader

    @staticmethod
    def add_model_specific_args(parser):
        '''
        TAKEN FROM HUGGINGFACE
        '''
        ### GENERIC ARGUMENTS
        parser.add_argument(
            "--output_dir",
            default="./outputs",
            type=str,
            required=False,
            help="The output directory where the model predictions and checkpoints will be written.",
        )

        parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
        parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
        parser.add_argument(
            "--gradient_accumulation_steps",
            dest="accumulate_grad_batches",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

        ### MODEL SPECIFIC ARGUMENTS
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="/scratch/project_2001970/Geometry/wmt_en_ro",
            required=False,
            help="The input data dir. Should contain 6 files: train.source, train.target, val.source, val.target, test.source, test.target",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")

        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        #parser.add_argument("--src_lang", type=str, default="", required=False)
        #parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
 
        ### BASE TRANSFORMER ARGUMENTS
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout", type=float, help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout", type=float, help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        
        ### ARGUMENTS FOR HYBRID
        parser.add_argument("--bert_type", type=str, default='bert-base-uncased', help="The bert model from huggingface to be used as encoder")
        parser.add_argument("--mt_mname" , type=str, default='Helsinki-NLP/opus-mt-en-de', help="The MT model from huggingface to be used as decoder")
        parser.add_argument("--output_attentions", action="store_true")
        parser.add_argument("--output_hidden_states", action="store_true")
        #parser.add_argument('--gpus', type=int, default=0, help='number of gpus to use')

        return parser

    ###########################################################





if False:

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["decoder_input_ids"].ne(self.pad).sum()
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        rouges = {k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]}
        rouge_tensor: torch.FloatTensor = torch.tensor(rouges[self.val_metric]).type_as(loss)
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {"log": metrics, "preds": preds, f"{prefix}_loss": loss, f"{prefix}_{self.val_metric}": rouge_tensor}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")



    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)