
"""
Simple argument parser. Can specify model to load, metrics to compute, datasets to use, sampling size, treatment of of tokenization & subword units, etc.
"""

import argparse

def get_parser():
    """
    Add arguments in alphabetical order.
    """
    parser = argparse.ArgumentParser()

    # MODELS
    parser.add_argument("--bert_models",  "-bert", type=str, required=False,  nargs='+', default=["bert-base-uncased","bert-base-cased"],
                        help="Which BERT to use for contextualized embeddings [bert-base-uncased | bert-base-cased | bert-base-multilingual-uncased | bert-base-multilingual-cased].")
    
    parser.add_argument("--huggingface_models", '-hfmodels', type=str, required=False, nargs='+', default=['en-fi'],
                        help="src-tgt languages for the huggingface model(s) to use; separate with a space"\
                              "Defaults to Helsinki-NLP/opus-mt-en-fi ."\
                              " See all available models on https://huggingface.co/models"\
                              "If --data_path includes src and tgt,  only need to specify src-tgt and will compute "\
                              "embeddings for both directions: src-tgt and tgt-src")

    parser.add_argument("--prealigned_models", '-loadmodels', type=str, required=False, nargs='+', default=[None],
                        help="Path to prealigned model. \
                             Need to specify if BERT or a Helsinki-NLP/opus-mt-$src-$tgt Model\
                             using --prealigned_model_type")

    parser.add_argument("--prealigned_model_types", type=str, required=False, nargs='+', default=[''],
                        help="For each prealigned model, specify which model type it is to be used \
                              'bert' or '$src-$tgt' for a Helsinki-NLP/opus-mt-$src-$tgt model")
    
    # SETTINGS
    parser.add_argument("--max_nsents", required=False, type=int, default=-1,
                        help='size of the amount of sents to load when using dev_mode')
   
    # DATA & PATHS
    parser.add_argument("--data_path", required=False, type=str, nargs="+", default=["../data/STS/allSTS.txt"],
                        help="Dataset used to extract embedding from. Defaults to STS datasets."\
                              "Either give path to 1 file (only source) and the system will compute embeddings with the specified models with this file"\
                              "or give 2 files (source and target) and the system will compute embeddings for src-tgt and tgt-src")
   
    parser.add_argument("--load_embeddings_path", type=str, required=False, nargs='+', 
                        help="name of the huggingface embeddings(s) to use. If more than one, separate with a space"\
                              "if not specified, will compute embeddings for all specified models"\
                              "and save them into [outdir]/embeddings/[data_path_basename]_[modelname].pkl")

    parser.add_argument("--outdir", type=str, required=False, default='../outputs/',
                        help='path to dir where outputs will be saved')


    ##### THINGS ABOUT SAMPLING
    parser.add_argument("--intrasentsim_samplesize", "-intra_ss", required=False, type=int, default=500,
                        help='size of the sentence sample to be extracted for computing the intra-sentence similarity')

    parser.add_argument("--isotropycorrection_samplesize", required=False, type=int, default=1000,
                        help='size of the words sample to be extracted for computing the isotropy correction factor')

    parser.add_argument("--selfsim_samplesize", "-self_ss",required=False, type=int, default=2500,
                        help='size of the words sample to extract for computing the self-similarity')



    # OTHER
    parser.add_argument("--cuda", action='store_true',
                        help="Whether to use a cuda device.")

    parser.add_argument("--debug_mode", action='store_true',
                        help="Launch ipdb debugger if script crashes.")

    parser.add_argument("--dev_params", action='store_true',
                        help="set params and options small enough to develop faster.")
    
    parser.add_argument("--only_save_embs", action='store_true',
                        help="if active, will exit after computing and saving the embeddings.")

    parser.add_argument("--plot_results", action='store_true',
                        help="if active, will plot results after computing metrics.")

    #parser.add_argument("--save_results", action='store_true',
    #                    help="if active, will save results into --outdir.")

    #parser.add_argument("--use_samples", action='store_true',
    #                    help="use samples instead of the whole dataset.")

    return parser
