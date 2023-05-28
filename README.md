# :boom: hyped
![Publish Master](https://github.com/ndoll1998/hyped/workflows/PyPI/badge.svg)

A collection of data pipelines to ease the training of transformer models

## installation

The package can be installed from PyPI:

```bash
pip install -U hyped
```

Alternatively you can also install the package from source by cloning this repository

```bash
git clone https://github.com/ndoll1998/hyped.git
cd hyped
pip install -e .
```

## dependencies

`hyped` combines the 🤗 huggingface libraries to implement pipelines that go from data preparation to model training and finally evaluation. The following table shows which library is used as backend to which pipeline stage.

| stage | backend | version |
|:-----:|:-------:|:-------:|
| data preparation | [datasets](https://github.com/huggingface/datasets) | `>=2.12.0` |
| model training | [adapter-transformers](https://github.com/adapter-hub/adapter-transformers) | `>=3.2.1` |
| model evaluation | [evaluate](https://github.com/huggingface/evaluate) | `>=0.4.0` |

## usage

The purpose of `hyped` is to train complex transformer models with minimal effort. The typical ML pipeline consists of three main stages, namely a `data preparation`, a `model training` and a `model evaluation` stage. Each one of these stages can be executed by the `hyped` command line interface.

The following demonstates how to train a text classification model on the [`imdb`](https://huggingface.co/datasets/imdb) dataset by going through all three stages of the ML pipeline (from `examples/run.sh`).

```bash
# data preparation stage
# runs the data preparation pipeline specified in the
# configuration file and saves the prepared dataset
# to the specified location
python -m hyped.stages.prepare \
    -c examples/text_cls/imdb/distilbert_data.json \
    -o output/distilbert_data

# model training stage
# trains a model on the prepared data generated by the
# previous stage
python -m hyped.stages.train \
    -c examples/text_cls/imdb/distilbert_run.json \
    -d output/distilbert_data \
    -o output/model

# model evaluation stage
# evaluates the trained model on the test split of the
# prepared dataset generated earlier
python -m hyped.stages.evaluate \
    -c examples/text_cls/imdb/distilbert_run.json \
    -d output/distilbert_data \
    -m output/model/best-model
```

## features

`hyped` currently implements all components required to train and evaluate models for the following NLP tasks:

 - Text Classification
 - Named Entity Recognition
 - Multi-label Classification
 - Causal Language Modeling

### adapters

`hyped` is build upon [`adapter-transformers`](https://docs.adapterhub.ml/), which allows the configuration of a variety of model architectures. See the following excerpt for how to configure an adapter model in the run configuration (from [`examples/cls/imdb/distilbert_run.json`](examples/cls/imdb/distilbert_run.json)):

```json
{
    "model": {
        "pretrained_ckpt": "distilbert-base-uncased",
        
        "adapter_name": "imdb",
        "adapter": {
            "train_adapter": true,
            "adapter_config": "pfeiffer"
        },

        "heads": {
            "cls": {
                "head_type": "hyped-cls-head",
                "label_column": "labels"
            }
        }
    }
}
```

The `adapter` field in the above configuration mirrors the [`AdapterArguments`](https://docs.adapterhub.ml/classes/adapter_training.html#transformers.adapters.training.AdapterArguments) class and can specify all it's values (e.g. `load_adapter` for loading pretrained adapters). Furthermore, by setting `train_adapter` to True, only the paramters of the adapter and head are trained, while the pretrained encoder parameters are frozen.

Alternatively, the adapter configuration can also be omitted, in which case no adapter is used and the encoder output is directly passed to the prediction heads.

### dvc

`hyped` is designed to be easily integratable into existing data pipelines. Thus, it fits well with [`dvc`](https://dvc.org/) pipelines. See [`examples/dvc`](examples/dvc) for an example setup.

### deepspeed

`hyped` supports distributed training allowing for both pre-training and fine-tuning of very large language models. See [`examples/lm/run.json`](examples/lm/run.json) for an example configuration. To lauch a distributed training you need to use `hyped`'s command line interface:

```bash
deepspeed --no_python hyped train -c config.json -d data/ -o model/
```

## roadmap

`hyped` is currently in its early stages of development. Over time, the goal is to expand its applicability to a diverse range of tasks and setups. The following features are planned for future development:

 - ~~support adapter training~~
 - ~~support transformers models~~
 - support more tasks
   - Masked Language Modeling
   - ~~Causal Language Modeling~~
   - nested Named Entity Recognition
   - Question Answering
 - support multi-modal encoders
   - LayoutLM
   - LiLT
 - support distributed training/inference
   - ~~deepspeed~~
   - pytorch DDP and FSDP
