
"""
Simple argument parser. Can specify model to load, metrics to compute, datasets to use, sampling size, treatment of of tokenization & subword units, etc.
"""

import argparse

def get_parser():
    """
    Add arguments in alphabetical order.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--apply_bpe", action='store_true',
                        help="Does your model needs BPEd data as input?")

    parser.add_argument("--bpe_model", required=False,
                        help='path to read onmt_model')

    parser.add_argument("--cuda", action='store_true',
                        help="Whether to use a cuda device.")
    
    parser.add_argument("--data_path", required=False, type=str, default="../data/STS/allSTS.txt",
                        help="Dataset used to sample from. Defaults to STS datasets.")

    parser.add_argument("--debug_mode", action='store_true',
                        help="Launch ipdb debugger if script crashes.")
    
    parser.add_argument("--dev_params", action='store_true',
                        help="set params and options small enough to develop faster.")

    parser.add_argument("--intrasentsim_samplesize", "-intra_ss", required=False, type=int, default=500,
                        help='size of the sentence sample to be extracted for computing the intra-sentence similarity')

    parser.add_argument("--isotropycorrection_samplesize", required=False, type=int, default=1000,
                        help='size of the words sample to be extracted for computing the isotropy correction factor')

    parser.add_argument("--load_embeddings_path", type=str, required=False, nargs='+', 
                        help="name of the huggingface embeddings(s) to use. If more than one, separate with a space"\
                              "if not specified, will compute embeddings for all specified models in bert, huggingface & onmt"\
                              "and save them into ../embeddings/[modelname].pkl")

    parser.add_argument("--load_w2s_path", type=str, required=False, 
                        help="path to the word to sentence indexer. Saved as default in ../embeddings/w2s.pkl")

    parser.add_argument("--onmt_model", type=str, required=False,
                        help='path to read onmt_model')

    parser.add_argument("--outdir", type=str, required=False, default='../results/',
                        help='path to dir where outputs will be saved')

    parser.add_argument("--huggingface_models", '-hfmodels', type=str, required=False, nargs='+', default='Helsinki-NLP/opus-mt-en-de',
                        help="name of the huggingface model(s) to use. If more than one, separate with a space")

    parser.add_argument("--bert_model",  "-bert", type=str, required=False, default="bert-base-uncased",
                        help="Which BERT to use for contextualized embeddings [bert_base_uncased | bert_cased].")

    parser.add_argument("--selfsim_samplesize", "-self_ss",required=False, type=int, default=2500,
                        help='size of the words sample to extract for computing the self-similarity')

    parser.add_argument("--save_results", action='store_true',
                        help="if active, will save results into --outdir.")

    return parser
