# Installation/Setup

1. Install our Supercharged OpenAI Development API (soda) Python package

```bash
# Move into the soda directory
cd soda

# Install the requirements (e.g. openai)
pip install -r requirements.txt

# Install the soda package
pip install -e .

# Move back into the root directory
cd ..

# Move into the classifiers directory
cd classifiers

# Run a paired completion experiment
python paired_completion.py
```

## Configuring Inference

We ran experiments on an NVIDIA H100 instance on Lambda Labs, using VLLM to create an inference server. We then edited `soda/soda/openai/text.py` to point to the VLLM server. We have configured `text.py` to instead point to DeepInfra, which supports the echo features required for our method. You will need to set an environment variable `DEEPINFRA_API_KEY` to a valid API key for DeepInfra, or alternatively you can edit `text.py` to point to another service (ChatGPT/Claude will be happy to help you with this if you're unfamiliar with the process).

## Running Experiments

Ensure you start these steps from the root directory (i.e. the directory containing this README file).

Note that you might need to set `OPENAI_API_KEY` to a valid OpenAI API key, and/or `DEEPINFRA_API_KEY` to a valid DeepInfra API key, depending on which experiments you are running.

### Baseline Experiments

```bash
# Move back to the root directory (if necessary)
# cd ../ # If necessary

# Move into the experiments directory
cd classifiers

# Run the baseline classification experiments
bash run_classifiers.sh
```

### Embedding Experiments

Note that you will need to install `fasttext` to use the fasttext embeddings (though we find the OpenAI embeddings are significantly better).

```bash
# Install fasttext if you want to use the fasttext embeddings
pip install fasttext

# Download cc.en.300.bin from (e.g.) https://www.kaggle.com/datasets/sanyatargrenkin/cc-en-300-bin
```

Embedding experiments can be run with `run_embeddings.sh`. This script uses papermill to run the experiments using the Jupyter notebook `embedding.ipynb`.

### Paired Completion Experiments

The code used for the paired completion experiments is split into three files in the `classifiers/` directory:

- `new_method_experiment.ipynb` contains the code for running individual experiments
- `import_data.py` contains the code for importing the data
- `analyze.py` contains the paired completion implementation

We use papermill for our experiments, which is a tool for running Jupyter notebooks from the command line. We have provided a script to run the paired completion experiments from the paper as `run_new_method_sentences_experiment.sh`.

We also prepared a script, `paired_completion.py`, which can be used to run individual experiments. You can modify the parameters at the top of the script, and/or create your own datasets, by following the data structure found in the JSON files in `gpt-4-only-corpora`. While the script seems to work, note that we used the Jupyter notebook/papermill approach for the experiments in our paper.

## Datasets

The datasets used in the experiments are located in the `gpt-4-only-corpora` directory. As the name implies, these datasets were generated using GPT-4 (specifically `gpt-4-turbo-preview`).
