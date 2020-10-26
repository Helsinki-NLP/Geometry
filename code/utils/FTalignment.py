
'''
This code was provided by Steven Cao and has some modifications that were required for matching this project's objecitves
If using this code, please do cite: 
    Steven Cao, Nikita Kitaev & Dan Klein (2020). MULTILINGUAL ALIGNMENT OF CONTEXTUAL WORD REPRESENTATIONS. ICLR 2020

'''
import numpy as np
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import time

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




class WordLevelOPUSmt(nn.Module):
    """
    Runs an OPUS-mt model on sentences but only keeps the last subword embedding for
    each word.
    """
    def __init__(self, model):
        super().__init__()
        self.mt_tokenizer = transformers.MarianTokenizer.from_pretrained(model)
        self.mt_model = transformers.MarianMTModel.from_pretrained(model)

        
        self.dim = self.mt_model.config.d_model
        self.max_len = self.mt_tokenizer.model_max_length
        self.pad_token_id = self.mt_tokenizer.pad_token_id
        self.device='cpu'

        if use_cuda:
            self.device='cuda'
            self.cuda()
    
    def forward(self, sentences, include_clssep = True):
        batch_size = 128
        ann_full = None
        for i in range(0, len(sentences), batch_size):
            ann = self.annotate(sentences[i:i+batch_size], 
                                include_sep = include_clssep)
            if ann_full is None:
                ann_full = ann
            else:
                ann_full = torch.cat((ann_full, ann), dim = 0)
        return ann_full
    
    def annotate(self, sentences, include_sep = True):
        """
        Input: sentences, which is a list of sentences
            Each sentence is a list of words.
            Each word is a string.
        Output: an array with dimensions (packed_len, dim).
            packed_len is the total number of words, plus 1 for each sentence
            for the ending token '</s>'.
        """
        if include_sep:
            packed_len = sum([(len(s) + 1) for s in sentences])
        else:
            packed_len = sum([len(s) for s in sentences])
                
        '''
        # tokenize with OPUS-mt tokenizer which adds </s> tokens
        sents = [' '.join(ss) for ss in sentences]
        all_inputs=self.mt_tokenizer.prepare_translation_batch(sents)
        all_inputs = {k:v.to(self.device) for k,v in all_inputs.items()}
        max_sent = all_inputs['input_ids'].size(1)
        '''

        # Each row is the token ids for a sentence, padded with pad_token_id.
        all_input_ids = np.full((len(sentences), self.max_len), self.pad_token_id, dtype = int)
        # Mask with 1 for real tokens and 0 for padding.
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for the last subword for each word.
        all_end_mask = np.zeros((len(sentences), self.max_len), dtype = int)

        # tokenize to get word splits
        max_sent=0
        for s_num, sentence in enumerate(sentences):
            tokens = []
            end_mask = []
            for word in sentence:
                word_tokens = self.mt_tokenizer.tokenize(word)
                assert len(word_tokens) > 0, \
                    "Unknown word: {} in {}".format(word, sentence)
                for _ in range(len(word_tokens)):
                    end_mask.append(0)
                end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append("</s>")
            end_mask.append(int(include_sep))
            input_ids = self.mt_tokenizer.convert_tokens_to_ids(tokens)
            
            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1
            all_end_mask[s_num, :len(end_mask)] = end_mask
            max_sent = max(max_sent, len(input_ids))

        all_input_ids = all_input_ids[:, :max_sent]
        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids))
        all_input_mask = all_input_mask[:, :max_sent]
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask))
        all_end_mask = all_end_mask[:, :max_sent]
        all_end_mask = from_numpy(np.ascontiguousarray(all_end_mask))


        # encode
        features = self.mt_model.model.encoder(
            input_ids = all_input_ids,
            attention_mask = all_input_mask.bool(),
            return_dict=True,
            )
        features=features['last_hidden_state']

        # for each word, only keep last encoded token.
        all_end_mask = all_end_mask.to(torch.uint8).unsqueeze(-1)
        features_packed = features.masked_select(all_end_mask.bool())
        features_packed = features_packed.reshape(-1, features.shape[-1])
        
        assert features_packed.shape[0] == packed_len, "Features: {}, \
            Packed len: {}".format(features_packed.shape[0], packed_len)
        
        return features_packed

def get_bert(bert_model, bert_do_lower_case):
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case = bert_do_lower_case)
    bert = BertModel.from_pretrained(bert_model)
    return tokenizer, bert

class WordLevelBert_old(nn.Module):
    """
    Runs BERT on sentences but only keeps the last subword embedding for
    each word.
    """
    def __init__(self, model, do_lower_case, outdim=512):
        super().__init__()
        self.bert_tokenizer, self.bert = get_bert(model, do_lower_case)
        self.dim = self.bert.pooler.dense.in_features
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings
        self.project_output = False
        # LINEAR
        if outdim != self.dim:
            self.project_output = True
            self.linear = nn.Linear(self.dim, outdim)
        if use_cuda:
            self.cuda()
    
    def forward(self, sentences, include_clssep = True):
        batch_size = 128
        ann_full = None
        for i in range(0, len(sentences), batch_size):
            ann = self.annotate(sentences[i:i+batch_size], 
                                include_clssep = include_clssep)
            if ann_full is None:
                ann_full = ann
            else:
                ann_full = torch.cat((ann_full, ann), dim = 0)
        return ann_full
    
    def annotate(self, sentences, include_clssep = True):
        """
        Input: sentences, which is a list of sentences
            Each sentence is a list of words.
            Each word is a string.
        Output: an array with dimensions (packed_len, dim).
            packed_len is the total number of words, plus 2 for each sentence
            for [CLS] and [SEP].
        """
        if include_clssep:
            packed_len = sum([(len(s) + 2) for s in sentences])
        else:
            packed_len = sum([len(s) for s in sentences])
        
        # Each row is the token ids for a sentence, padded with zeros.
        all_input_ids = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for real tokens and 0 for padding.
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for the last subword for each word.
        all_end_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        max_sent = 0
        # tokenize with bert tokenizer and add [CLS] and [SEP] tokens
        for s_num, sentence in enumerate(sentences):
            tokens = []
            end_mask = []
            tokens.append("[CLS]")
            end_mask.append(int(include_clssep))
            for word in sentence:
                word_tokens = self.bert_tokenizer.tokenize(word)
                assert len(word_tokens) > 0, \
                    "Unknown word: {} in {}".format(word, sentence)
                for _ in range(len(word_tokens)):
                    end_mask.append(0)
                end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append("[SEP]")
            end_mask.append(int(include_clssep))
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            
            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1
            all_end_mask[s_num, :len(end_mask)] = end_mask
            max_sent = max(max_sent, len(input_ids))
        all_input_ids = all_input_ids[:, :max_sent]
        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids))
        all_input_mask = all_input_mask[:, :max_sent]
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask))
        all_end_mask = all_end_mask[:, :max_sent]
        all_end_mask = from_numpy(np.ascontiguousarray(all_end_mask))
        
        # all_input_ids: num_sentences x max_sentence_len
        features, _ = self.bert(all_input_ids, attention_mask = all_input_mask.bool(),
                                output_all_encoded_layers = False)
        del _

        if self.project_output:
            features = self.linear(features)
        # for each word, only keep last encoded token.
        all_end_mask = all_end_mask.to(torch.uint8).unsqueeze(-1)
        features_packed = features.masked_select(all_end_mask.bool())
        features_packed = features_packed.reshape(-1, features.shape[-1])
        
        assert features_packed.shape[0] == packed_len, "Features: {}, \
            Packed len: {}".format(features_packed.shape[0], packed_len)
        
        return features_packed


class WordLevelBert(nn.Module):
    """
    Runs BERT on sentences but only keeps the last subword embedding for
    each word.
    """
    def __init__(self, bert_type, do_lower_case, outdim=512):
        super().__init__()

        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_type)
        self.bert = transformers.BertModel.from_pretrained(bert_type)

        self.dim = self.bert.pooler.dense.in_features
        self.max_len = self.bert.embeddings.position_embeddings.num_embeddings
        self.project_output = False
        # LINEAR
        if outdim != self.dim:
            self.project_output = True
            self.linear = nn.Linear(self.dim, outdim)
        if use_cuda:
            self.cuda()
    
    def forward(self, sentences, include_clssep = True):
        batch_size = 128
        ann_full = None
        for i in range(0, len(sentences), batch_size):
            ann = self.annotate(sentences[i:i+batch_size], 
                                include_clssep = include_clssep)
            if ann_full is None:
                ann_full = ann
            else:
                ann_full = torch.cat((ann_full, ann), dim = 0)
        return ann_full
    
    def annotate(self, sentences, include_clssep = True):
        """
        Input: sentences, which is a list of sentences
            Each sentence is a list of words.
            Each word is a string.
        Output: an array with dimensions (packed_len, dim).
            packed_len is the total number of words, plus 2 for each sentence
            for [CLS] and [SEP].
        """
        if include_clssep:
            packed_len = sum([(len(s) + 2) for s in sentences])
        else:
            packed_len = sum([len(s) for s in sentences])
        
        # Each row is the token ids for a sentence, padded with zeros.
        all_input_ids = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for real tokens and 0 for padding.
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for the last subword for each word.
        all_end_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        max_sent = 0
        # tokenize with bert tokenizer and add [CLS] and [SEP] tokens
        for s_num, sentence in enumerate(sentences):
            tokens = []
            end_mask = []
            tokens.append("[CLS]")
            end_mask.append(int(include_clssep))
            for word in sentence:
                word_tokens = self.bert_tokenizer.tokenize(word)
                assert len(word_tokens) > 0, \
                    "Unknown word: {} in {}".format(word, sentence)
                for _ in range(len(word_tokens)):
                    end_mask.append(0)
                end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append("[SEP]")
            end_mask.append(int(include_clssep))
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            
            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1
            all_end_mask[s_num, :len(end_mask)] = end_mask
            max_sent = max(max_sent, len(input_ids))
        all_input_ids = all_input_ids[:, :max_sent]
        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids))
        all_input_mask = all_input_mask[:, :max_sent]
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask))
        all_end_mask = all_end_mask[:, :max_sent]
        all_end_mask = from_numpy(np.ascontiguousarray(all_end_mask))
        
        # all_input_ids: num_sentences x max_sentence_len
        features = self.bert(all_input_ids, attention_mask = all_input_mask.bool(), return_dict=True)

        if self.project_output:
            features = self.linear(features.last_hidden_state)
        # for each word, only keep last encoded token.
        all_end_mask = all_end_mask.to(torch.uint8).unsqueeze(-1)
        features_packed = features.masked_select(all_end_mask.bool())
        features_packed = features_packed.reshape(-1, features.shape[-1])
        
        assert features_packed.shape[0] == packed_len, "Features: {}, \
            Packed len: {}".format(features_packed.shape[0], packed_len)
        
        return features_packed

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

def load_align_corpus(sent_path, align_path, max_len = 64, max_sent = np.inf):
    sentences_1 = []
    sentences_2 = []
    bad_idx = []
    with open(sent_path) as sent_file:
        """Lines should be of the form
        doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
        
        Result: 
        [
        ['doch', 'jetzt', ...],
        ...
        ]
        
        [
        ['but', 'now', ...],
        ...
        ]
        
        If sentences are already in sub-tokenized form, then max_len should be
        512. Otherwise, sentence length might increase after bert tokenization.
        (Bert has a max length of 512.)
        """
        for i, line in enumerate(sent_file):
            if i >= max_sent:
                break
            
            sent_1 = line[:line.index("|||")].split()
            sent_2 = line[line.index("|||"):].split()[1:]
            
            if len(sent_1) > max_len or len(sent_2) > max_len:
                bad_idx.append(i)
            else:
                sentences_1.append(sent_1)
                sentences_2.append(sent_2)
    
    if align_path is None:
        #return sentences_1, sentences_2, None
        # encoding ENGLISH with both, BERT and OPUS-mt:
        alignments = [   np.array([  [j,j] for j,_ in enumerate(sent)  ])   for sent  in sentences_1   ]
        return sentences_1, sentences_1, alignments
        

    
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
                
    return sentences_1, sentences_2, alignments
    
def partial_sums(arr):
    for i in range(1, len(arr)):
        arr[i] += arr[i-1]
    arr.insert(0, 0)
    return arr[:-1]

def pick_aligned(sent_1, sent_2, align, cls_sep = True):
    """
    sent_1, sent_2 - lists of sentences. each sentence is a list of words.
    align - lists of alignments. each alignment is a list of pairs (i,j).
    """
    extra_tokens = 2 if cls_sep else 0
    idx_1 = partial_sums([len(s) + extra_tokens for s in sent_1]) #adjust indexes as if concat all sentences into a list of words
    idx_2 = partial_sums([len(s) + extra_tokens for s in sent_2])
    align = [a + [i_1, i_2] for a, i_1, i_2 in zip(align, idx_1, idx_2)] # adjust alignment to new indexes
    align = reduce(lambda x, y: np.vstack((x, y)), align)
    
    if cls_sep:
        align = align + 1 # to account for extra [CLS] at beginning
        # also add cls and sep as alignments
        cls_idx = np.array(list(zip(idx_1, idx_2)))
        sep_idx = (cls_idx - 1)[1:]
        sep_idx_last = np.array([(sum([len(s) + 2 for s in sent_1]) - 1,
                        sum([len(s) + 2 for s in sent_2]) - 1)])
        align = np.vstack((align, cls_idx, sep_idx, sep_idx_last))
    
    # returns idx_1, idx_2
    # pick out aligned tokens using ann_1[idx_1], ann_2[idx_2]
    return align[:, 0], align[:, 1]
    
def align_bert_multiple(
    train, 
    model, 
    model_base, 
    num_sentences, 
    languages, 
    batch_size,
    splitbatch_size = 4, 
    epochs = 1,
    learning_rate = 0.00005, 
    learning_rate_warmup_frac = 0.1,
    outdir='./',
    log_every_n_batches=100,
    ):
    # Adam hparams from Attention Is All You Need
    t0 = time.time()
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
            if ((iteration-1)/batch_size % log_every_n_batches == 0):
                print("Warming up, iter {}/{}".format(iteration, learning_rate_warmup_steps))
            set_lr(iteration * warmup_coeff)
            
    model_base.eval() # freeze and remember initial model # model_base should be the MT model for us
    
    total_processed = 0
    for epoch in range(epochs):
        print(f'\n ...starting epoch {epoch+1}/{epochs}', flush=True) 
        for i in range(0, num_sentences, batch_size):
            loss = None
            model.train()
            schedule_lr(total_processed // (len(languages)))
            for j, language in enumerate(languages):
                sent_1, sent_2, align = train[j]
                ii = i % len(sent_1) # cyclic list - datasets may be diff sizes
                ss_1, ss_2 = sent_1[ii:ii+batch_size], sent_2[ii:ii+batch_size]
                aa = align[ii:ii+batch_size]
                
                # split batch to reduce memory usage
                for k in range(0, len(ss_1), splitbatch_size):
                    s_1 = ss_1[k:k+splitbatch_size]
                    s_2 = ss_2[k:k+splitbatch_size]
                    a = aa[k:k+splitbatch_size]
                    
                    # pick out aligned indices in a packed representation
                    idx_1, idx_2 = pick_aligned(s_1, s_2, a, False)
                    
                    # compute vectors for each position, pack the sentences
                    # result: packed_len x dim
                    ann_1, ann_2 = model(s_1, False), model(s_2, False)
                    ann_2_base = model_base(s_2, False)
                    
                    loss_1 = F.mse_loss(ann_1[idx_1], ann_2_base[idx_2])
                    loss_2 = F.mse_loss(ann_2, ann_2_base)
                    loss_batch = loss_1 + loss_2
                    if loss is None: 
                        loss = loss_batch
                    else: 
                        loss += loss_batch
                total_processed += len(ss_1)
            
            if (i/batch_size % log_every_n_batches == 0):
                print(f'Sentences {i}-{min(i+batch_size, num_sentences)}/{num_sentences}, Loss: {loss:.4f}, Time: {time.time()-t0:.1f} secs', flush=True) 
            
            loss.backward()
            trainer.step()
            trainer.zero_grad()
    
    print(f'saving into: {outdir}/best_alignment_network.pt')            
    torch.save({'state_dict': model.state_dict(),
                'trainer' : trainer.state_dict(),}, f'{outdir}/best_alignment_network.pt')

def normalize(vecs):
    norm = np.array([np.linalg.norm(vecs)])
    norm[norm < 1e-5] = 1
    normalized = vecs / norm
    return normalized
    
def hubness_CSLS(ann_1, ann_2, k = 10):
    """
    Computes hubness r(x) of an embedding x, or the mean similarity of x to 
    the K closest neighbors in Y. Used for the CSLS metric:
    CSLS(x, y) = 2cos(x,y) - r(x) - r(y)
    which penalizes words with high hubness, or a dense neighborhood.
    
    Uses k = 10, similarly to https://arxiv.org/pdf/1710.04087.pdf.
    """
    ann_1, ann_2 = normalize(ann_1), normalize(ann_2)
    sim = ann_1.dot(ann_2.T) # words_1 x words_2
    neighbors_1 = np.sort(sim, axis = 1)[:, -k:] # words_1 x k
    neighbors_2 = np.sort(sim.T, axis = 1)[:, -k:] # words_2 x k
    return np.mean(neighbors_1, axis = 1), np.mean(neighbors_2, axis = 1)

def bestk_idx_CSLS(x, vecs, vec_hubness, k = 5):
    """
    Looks for the k closest vectors using the CSLS metric, which is cosine
    similarity with a hubness penalty.
    
    Usage:
        hub_1, hub_2 = hubness_CSLS(vecs_1, vecs_2)
        # get word translations for vecs_1[0]
        best_k = bestk_idx_CSLS(vecs_1[0], vecs_2, hub_2)
    """
    x, vecs = normalize(x), normalize(vecs)
    sim = 2 * vecs.dot(x) - vec_hubness
    return np.argsort(-sim)[:k]

def evaluate_retrieval(dev, model):
    sent_1, sent_2, align = dev
    idx_1, idx_2 = pick_aligned(sent_1, sent_2, align)
    model.eval()
    with torch.no_grad():
        ann_1 = model(sent_1)[idx_1].detach().cpu().numpy()
        ann_2 = model(sent_2)[idx_2].detach().cpu().numpy()
    hub_1, hub_2 = hubness_CSLS(ann_1, ann_2)
    matches_1 = [bestk_idx_CSLS(ann, ann_2, hub_2)[0] for ann in ann_1]
    matches_2 = [bestk_idx_CSLS(ann, ann_1, hub_1)[0] for ann in ann_2]
    # acc_1 = 1, if matches_1 predicts [0,1,2,3,..,len(matches_1)] - bc we are giving the sentences in order
    acc_1 = np.sum(np.array(matches_1) == np.arange(len(matches_1))) / len(matches_1)
    acc_2 = np.sum(np.array(matches_2) == np.arange(len(matches_2))) / len(matches_2)
    return acc_1, acc_2


if __name__ == '__main__':

    num_sent = 20 # size of experiment - to be splitted into train & dev
    languages = ['de', 'bg']
    sent_paths = ['/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.bg-en.token.clean.reverse', 
                  '/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.de-en.token.clean.reverse' ]
    align_paths = ['/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.bg-en.intersect.reverse',
                   '/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.de-en.intersect.reverse']
    batch_size = 8
    num_epochs = 2

    model_base = WordLevelBert('bert-base-multilingual-cased', False) # this should be the MT model for us
    model = WordLevelBert('bert-base-multilingual-cased', False) # this is the one that will have the parameters updated

    model_base2 = WordLevelOPUSmt('Helsinki-NLP/opus-mt-en-fi') 
    model2 = WordLevelBert('bert-base-multilingual-cased', False, model_base2.dim) # this should be the MT model for us

    data = [load_align_corpus(sent_path, align_path, max_sent = num_sent) for
            sent_path, align_path in zip(sent_paths, [None,None])]

    #data = [load_align_corpus(sent_path, align_path, max_sent = num_sent) for
    #        sent_path, align_path in zip(sent_paths, align_paths)]


    num_dev = 10
    dev = [(sent_1[:num_dev], sent_2[:num_dev], align[:num_dev]) for sent_1, sent_2, align in data]
    train = [(sent_1[num_dev:], sent_2[num_dev:], align[num_dev:]) for sent_1, sent_2, align in data]

    print('aligninig...')
    align_bert_multiple(train, model2, model_base2, num_sent, languages, batch_size, epochs=num_epochs)
    statedict = model2.state_dict()

    for lang, dev_lang in zip(languages, dev):
        print(lang)
        print("Word retrieval accuracy:", evaluate_retrieval(dev_lang, model))
    align_bert_multiple(train, model, model_base, num_sent, languages, batch_size, epochs=num_epochs)
    for lang, dev_lang in zip(languages, dev):
        print(lang)
        print("Word retrieval accuracy:", evaluate_retrieval(dev_lang, model))
