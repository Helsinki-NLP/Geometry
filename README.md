# Comparing the internal representations of contextualized embeddings (LM & MT)

### Requirements & Instalation 
You need to clone this repo
```
cd path/to/your/git/repos
git clone https://github.com/Helsinki-NLP/Geometry.git
cd Geometry
```
We strongly recommend to make the setup in a virtual environment. 
This is done by:
```
#export PYTHONUSERBASE=/projappl/project_2001970
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
#export PYTHONPATH=/projappl/project_2001970/lib/python3.7/site-packages/
python3 -m venv env
source env/bin/activate
```
_You can confirm you’re in the virtual environment by checking the location of your Python interpreter, it should point to the env directory:_
```
which python
```
Now that you’re in your virtual environment you can install packages without worrying too much about version control.

First you need to have installed [`torch`](https://pytorch.org/get-started/locally/) according to your system requirements
```
nvidia-smi
# python3 -m pip install torch torchvision
# srun -p gputest --gres=gpu:v100:1 --mem=4G --account=<account_name> python3 -m pip install torch torchvision
```

After installing pytorch, you can run: 
```
pip install -r requirements.txt
```

### Usage

To replicate our experiments, you will first need to clone [moses decoder](https://github.com/moses-smt/mosesdecoder.git) and [fast_align](https://github.com/clab/fast_align.git); install [eflomal](https://github.com/Helsinki-NLP/eflomal.git)
; and  download the parallel data and the STS datasets - can do by running our adapted script from [SentEval](https://github.com/facebookresearch/SentEval):
```
cd path/to/your/git/repos
git clone https://github.com/clab/fast_align.git
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/Helsinki-NLP/eflomal.git

cd eflomal
make
sudo make install
python3 setup.py install # your venv needs to be activated at this step

cd ../Geometry
source getSTSdb.sh /path/to/where/you/cloned/mosesdecoder
source get_data_and_learn_alignment.sh ./data/ # you can also give another directory to download the parallel corpora
```


You can replicate our experiments by running: 
```
source env/bin/activate
cd code
python compare_context_embeddings.py --save_results
```

You can also override the default options using the command line flags. 
This way, you can compute the measures for your own pretrained model. 
```
optional arguments:
  -h, --help        show this help message and exit
  --apply_bpe       Does your model needs BPEd data as input?
  --bpe_model       path to read onmt_model
  --cuda            Whether to use a cuda device.
  --data_path       Dataset used to sample from. Defaults to STS datasets.
  --debug_mode                     Launch ipdb debugger if script crashes.
  --intrasentsim_samplesize        size of the sentence sample to be extracted for
                                   computing the intra-sentence similarity
  --isotropycorrection_samplesize  size of the words sample to be extracted for computing
                                   the isotropy correction factor
  --onmt_model        path to read onmt_model
  --outdir            path to dir where outputs will be saved
  --huggingface_model name of the huggingface model to use
  --bert_model        Which BERT to use for contextualized embeddings
                      [bert_base_uncased | bert_cased].
  --selfsim_samplesize  size of the words sample to extract for computing the self-similarity
  --save_results        if active, will save results into --outdir.

```

### Contact
Report bugs, contribute, give feedback or somply reach out. We are happy if you decide to contact us. 

Please cite, if using this work: 
``` 
<Hopefully, we will have a preprint soon>
```

### Acknowledgements


