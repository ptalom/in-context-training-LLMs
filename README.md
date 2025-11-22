
###### This repo contains the code of our paper : 
##### How Are Transformers Sensitive to Coherence? A case study on Matrix completion

## Goals of the project
We test the performance of Transformers by changing coherence controller alpha/beta

### coherence-based sampling
- High incoherent Matrix (alpha=beta=0)
- 50% coherent, 50% incoherent (alpha=beta=0.5)
- High coherent Matrix (alpha=beta=1)

## Setup
```bash
git clone https://github.com/ptalom/in-context-training-LLMs.git
cd in-context-training-LLMs
cd src
```

## Getting started
Check the `requirements.txt` file to configure the appropriated environment

## To change configurations
- For Matrix Completion task, edit the yaml configuration file : `src/conf/matrix_completion.yaml`
You can start by cloning our repository and following the steps below.

Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```


- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. 

## Training 
You can try `python src/train.py --config conf/matrix_completion.yaml` for a quick training run.

## Contributors
- Patrick C. Talom
- Pascal Jr Notsawo Tikeng

Inspired by the article : *"What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (Garg et al., 2022)"*