import time
start = time.time()


import torch
import torch.nn as nn
import torch.nn.functional as F



from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, PreTrainedModel, PretrainedConfig
from transformers.generation_utils import GenerationMixin
from typing import Dict, List, Optional, Tuple, Union

#class BertMT_hybrid(nn.Module, GenerationMixin):

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
    def __init__(self, config, bert_type='bert-base-uncased', mt_mname='Helsinki-NLP/opus-mt-en-fi', use_cuda=False, output_hidden_states=False):
        super().__init__(config)
        #super().__init__(config)
        

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
        self.bsz = 256 if use_cuda else 16
        #self.device = 'cpu'

        # BERT: 
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.bert = BertEncoder4Hybrid(bert_type, self.mt_hdim)

        self.bert.linear.weight.data.normal_(mean=0.0, std=config.init_std)

        #config = BertConfig.from_pretrained(bert_type, output_hidden_states=output_hidden_states)        
        #self.bert = BertModel.from_pretrained(bert_type, config=config)

        self.bert_encdim = self.bert.bert_encdim
        if use_cuda:
            #self.device = 'cuda'
            self.cuda()
    
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
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            #masked_lm_loss = loss_fct(lm_logits.view(-1, self.mt_model.config.vocab_size), labels.view(-1))
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.mt_model.config.vocab_size), decoder_input_ids.view(-1))      
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

        
#  -------------------------------------------------------
import transformers
use_cuda = torch.cuda.is_available() 
if use_cuda:
    print("Using CUDA!")
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(non_blocking=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy

bert_type = 'bert-base-uncased'
mt_mname = 'Helsinki-NLP/opus-mt-en-de'

dummyconfig=transformers.PretrainedConfig.from_pretrained(mt_mname)
dummyconfig.architectures = ["BertMT_hybrid"]
dummyconfig.encoder_attention_heads = 12
dummyconfig.encoder_layers = 12
dummyconfig.model_type = ''
dummyconfig = PretrainedConfig.from_dict(dummyconfig.to_dict())

model = BertMT_hybrid(config = dummyconfig,
    bert_type=bert_type, 
    mt_mname=mt_mname, 
    cuda=use_cuda)


text_batch = ["I love Pixar.", "I don't care for Pixar."]
#tokd_batch = tok.prepare_translation_batch(src_texts=text_batch) 
tgttext_batch = ["Ich liebe Pixar.", "Pixar ist mir egal."]

tokd_text = model.prepare_translation_batch(src_texts=text_batch)  
tokd_text2 = model.prepare_translation_batch(src_texts=text_batch, tgt_texts=tgttext_batch)  
#coso = model.mt_model.generate(**tokd_text)
tokd_batch = model.mt_tokenizer.prepare_translation_batch(src_texts=text_batch)

elapsed_time_fl = (time.time() - start) 
print(f'Initialization required : {elapsed_time_fl} seconds')

 
start = time.time()
fwdpass_output = model(**tokd_text)  
elapsed_time_fl = (time.time() - start) 
print(f'forward pass needed : {elapsed_time_fl} seconds')

start = time.time()
mt_gentext = model.mt_model.generate(**tokd_batch)
elapsed_time_fl = (time.time() - start) 
print(f'generation with MT model needed : {elapsed_time_fl} seconds')

start = time.time()
gentext = model.generate(**tokd_text)  
elapsed_time_fl = (time.time() - start) 
print(f'generation with Hybrid model needed : {elapsed_time_fl} seconds')



if False:
    '''
    TODO: SHOULD WE ALWAYS DROP THE 1st EMB FROM THE BERT MODELS?
    > model.bert_tokenizer.convert_ids_to_tokens(tokd_text['input_ids'][0])     
    ['[CLS]', 'i', 'love', 'pi', '##xa', '##r', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

    > model.mt_tokenizer.convert_ids_to_tokens(tokd_text['MT_input_ids'][0])    
    ['▁I', '▁love', '▁P', 'ix', 'ar', '.', '</s>', '<pad>', '<pad>', '<pad>', '<pad>']

    '''

    bert_tokd_batch = model.bert_tokenizer(text_batch, padding=True, return_tensors='pt')



    gentext = model.mt_model.generate(tokd_batch)


    bert_last_hstate, dec_features = model(text_batch)
    bert_last_hstate, dec_features = model(text_batch, tgt_texts=tgttext_batch)


    #for i in range(0, len(src_texts), model.bsz):


    ## ---------------------------------------------------
    ## get the LOSS of the MT model:

    #tokd_batch = model.mt_tokenizer.prepare_translation_batch(src_texts=text_batch, tgt_texts=tgttext_batch)

    #model_outputs = model.mt_model(**tokd_batch) # [2, 11, 58101] = [bsz, maxsentlen, vocabsz]
    #labels_size=tokd_batch['input_ids'].shape[1]- tokd_batch['decoder_input_ids'].shape[1] # [bsz, longest_sent]
    #labels = torch.ones_like(tokd_batch['input_ids']) * (-100)
    #labels = torch.cat((tokd_batch['decoder_input_ids'],labels[:,:labels_size]),dim=1)
    #loss, pred_scores, hstates = model.mt_model(input_ids=tokd_batch['input_ids'], attention_mask=tokd_batch['attention_mask'], labels=labels)
    ## ---------------------------------------------------

    #enc_outputs = encoder.forward(**tokd_batch) # [2, 11, 512] = [bsz, maxsentlen, hdim]

    # --

    # config_overrider={'output_attentions':True, 'output_hidden_states':True}
    # mt  = transformers.MarianMTModel.from_pretrained(modelname, **config_overrider)

    #emb_layer = mt.model.shared
    #encoder = mt.model.encoder

    #encoder = mt.get_encoder()
    #decoder = mt.model.decoder

    #from transformers import AdamW
    #optimizer = AdamW(mt.model.parameters(), lr=1e-5)

    #model_outputs = mt(**tokd_batch) # [2, 11, 58101] = [bsz, maxsentlen, vocabsz]
     
    # --

    # ---------------------------------------------------










if False:


    def keep_1to1(alignments):
        if len(alignments) == 0:
            return alignments
        
        counts1 = np.zeros(np.max(alignments[:,0]) + 1)
        counts2 = np.zeros(np.max(alignments[:,1]) + 1)
        
        for a in alignments:
            counts1[a[0]] += 1
            counts2[a[1]] += 1
        
        alignments2 = []
        for a in alignments:
            if counts1[a[0]] == 1 and counts2[a[1]] == 1:
                alignments2.append(a)
        return np.array(alignments2)

    def load_alignments(align_path, max_sent, bad_idx):
        alignments = []
        with open(align_path) as align_file:
            """Lines should be of the form
            0-0 1-1 2-4 3-2 4-3 5-5 6-6
            
            Only keeps 1-to-1 alignments.
            
            Result:
            [
            [[0,0], [1,1], ...],
            ...
            ]
            """
            # need to only keep 1-1 alignments
            for i, line in enumerate(align_file):
                if i >= max_sent:
                    break
                
                if i not in bad_idx:
                    alignment = [pair.split('-') for pair in line.split()]
                    alignment = np.array(alignment).astype(int)
                    alignment = keep_1to1(alignment)
                    alignments.append(alignment)

        return alignments

    def load_align_corpus(sent_path, align_path, max_len = 64, max_sent = np.inf):
        sentences_1 = []
        sentences_2 = []
        bad_idx = []
        with open(sent_path) as sent_file:
            """Lines should be of the form
            doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
            
            Result: 
            sentences_1 = ['doch jetzt ist der Held gefallen .', ...  ]
            
            sentences_2 = ['but now the hero has fallen .', ... ]
            
            If sentences are already in sub-tokenized form, then max_len should be
            512. Otherwise, sentence length might increase after bert tokenization.
            (Bert has a max length of 512.)
            """
            for i, line in enumerate(sent_file):
                if i >= max_sent:
                    break
                
                sent_1 = line[:line.index("|||")] #.split()
                sent_2 = line[line.index("|||"):].strip().strip('||| ') #.split()[1:]
                
                #if len(sent_1) > max_len or len(sent_2) > max_len:
                if len(sent_1.split()) > max_len or len(sent_2.split()) > max_len:
                    bad_idx.append(i)
                else:
                    sentences_1.append(sent_1)
                    sentences_2.append(sent_2)
        
        if align_path is None:
            return [sentences_1, sentences_2, bad_idx]
        
        alignments = load_alignments(align_path, max_sent, bad_idx)
                    
        return [sentences_1, sentences_2, alignments]
        



            

    # ENCODERS: 
    #     BERT MODEL:  WITH TOKENIZER AND FORWARD PASS
    #     MT MODEL:    WITH TOKENIZER AND FORWARD PASS

    # DECODER:
    #     MT MODEL: 

    bertEncoder = Loader.bertModel()
    mtEncoder = Loader.huggingfaceModel('Helsinki-NLP/opus-mt-en-de')
    #mtDecoder = 

    num_sent = 250 # size of experiment - to be splitted into train & dev
    languages = ['de']
    sent_paths = ['/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.de-en.token.clean.reverse', 
                  '/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.bg-en.token.clean.reverse']
    align_paths = ['/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.de-en.intersect.reverse', 
                   '/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.bg-en.intersect.reverse']
    batch_size = 8
    num_epochs = 2

    # Load data for MT
    data = [load_align_corpus(sent_path, None, max_sent = num_sent) for sent_path in sent_paths]

    #num_dev = 10
    #dev = [(sent_1[:num_dev], sent_2[:num_dev], align[:num_dev]) for sent_1, sent_2, align in data]
    #train = [(sent_1[num_dev:], sent_2[num_dev:], align[num_dev:]) for sent_1, sent_2, align in data]




    def finetune_mt_sys(train_data, encoder, decoder, 
                            num_sentences, languages, batch_size, 
                            splitbatch_size = 4, epochs = 1,
                            learning_rate = 0.00005, learning_rate_warmup_frac = 0.1):
        # Adam hparams from Attention Is All You Need
        trainer = torch.optim.Adam([param for param in model.parameters() if
                                    param.requires_grad], lr=1., 
                                   betas=(0.9, 0.98), eps=1e-9)
                                   
        # set up functions to do linear lr warmup
        def set_lr(new_lr):
            for param_group in trainer.param_groups:
                param_group['lr'] = new_lr
        learning_rate_warmup_steps = int(num_sentences * learning_rate_warmup_frac)
        warmup_coeff = learning_rate / learning_rate_warmup_steps
        def schedule_lr(iteration):
            iteration = iteration + 1
            if iteration <= learning_rate_warmup_steps:
                print("Warming up, iter {}/{}".format(iteration, learning_rate_warmup_steps))
                set_lr(iteration * warmup_coeff)
                
        
        total_processed = 0
        for epoch in range(epochs):
            for i in range(0, num_sentences, batch_size):
                loss = None
                encoder.train(); decoder.eval()
                schedule_lr(total_processed // (len(languages)))
                for j, language in enumerate(languages):
                    sent_1, sent_2, align = train_data[j]
                    ii = i % len(sent_1) # cyclic list - datasets may be diff sizes
                    ss_1, ss_2 = sent_1[ii:ii+batch_size], sent_2[ii:ii+batch_size]
                    aa = align[ii:ii+batch_size]
                    
                    # split batch to reduce memory usage
                    for k in range(0, len(ss_1), splitbatch_size):
                        s_1 = ss_1[k:k+splitbatch_size]
                        s_2 = ss_2[k:k+splitbatch_size]
                        a = aa[k:k+splitbatch_size]
                        
                        '''
                        COMPUTE LOSS
                        labels = torch.tensor([1,0]).unsqueeze(0)
                        outputs = model(input_ids, attention_mask=attention_mask)
                        loss = F.cross_entropy(labels, outputs[0])
                        '''
                print("Sentences {}-{}/{}, Loss: {}".format(
                        i, min(i+batch_size, num_sentences), num_sentences, loss))
                loss.backward()
                trainer.step()
                trainer.zero_grad()
                    
        torch.save({'state_dict': model.state_dict(),
                    'trainer' : trainer.state_dict(),}, 'best_network.pt')













