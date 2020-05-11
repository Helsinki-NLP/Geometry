# Comparing the internal representations of contextualized embeddings (LM & MT)

### Requirements & Instalation 
You need to clone this repo
```
git clone https://github.com/Helsinki-NLP/Geometry.git
cd Geometry
```
We strongly recommend to make the setup in a virtual environment. 
This is done by:
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
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

To replicate our experiments, you need to download the STS datasets. A straightforward way to do that is running
```
source getSTSdb.sh path/to/mosesdecoder
```
You will need to clone [moses decoder](https://github.com/moses-smt/mosesdecoder.git) `git clone https://github.com/moses-smt/mosesdecoder.git`
The previous script has been adapted from [SentEval](https://github.com/facebookresearch/SentEval).

You can replicate our experiments by running: 
```
python code/compare_context_embeddings.py
```

You can also override the options using the command line options. 
This way, you can compute the measures for your own pretrained model. 
`<when code is ready: paste the HELP message here>`

### Contact
Report bugs, contribute, give feedback or somply reach out. We are happy if you decide to contact us. 

Please cite, if using this work: 
``` 
<Hopefully, we will have a preprint soon>
```

### Acknowledgements


