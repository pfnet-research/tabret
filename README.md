# TabRet: Pre-training Transformer-based Tabular Models for Unseen Columns (ICLR 2023 Workshop on ME-FoMo)

This repository contains the code for our machine learning paper titled "TabRet: Pre-training Transformer-based Tabular Models for Unseen Columns" ([link](https://arxiv.org/abs/2303.15747)). In this README, we provide information about the environment we used and instructions on how to run the code to reproduce our experiments.

Feel free to report [issues](https://github.com/pfnet-research/tabret/issues).

## Environment

The experiments in our paper were conducted using the following environment:

- Operating System: Ubuntu 22.04.1 LTS
- CUDA compiler version: 11.7

## Installation

To install the dependencies for this project, please run the following command:

```bash
pip install -r requirements.txt
```

## Preparing the Data

Before running the experiment, you need to prepare the data that will be used for pre-training and fine-tuning the model. The data can be placed under any folder, and you can specify the folder with the `data_dir` option. By default, the data directory is set to `datasets/`.

For more information on data types and detailed data placement, please refer to the files for each type of data under `data/datasets/`.

*How to create data for pre-training*

Download the data from `https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system`. After unzipping, rename the folder to `raw` and move it directly under `data/`. Then, run `data/BRFSS.ipynb` accordingly to generate `all.csv`. Put this file in the `data_dir` before running the pre-training script.


## Running the Experiments

The learning execution phase is divided into pre-training and fine-tuning. Here's how to run each phase:

### Pre-training

To run the pre-training phase, please run the following command:

```bash
python main_pre.py
```
This will create a `{date}/{time}` directory under `outputs/` where the trained model will be stored. Note that you can specify any output directory by specifying `hydra.run.dir`.

You can modify the configurations for the pre-training phase in the conf/pre.yaml file.

*A specific example is shown below.*
<details><summary><em>[Click to expand]</em></summary>

<br>

```bash
python main_pre.py data=BRFSS \
pre_conf=tabret model=tabret \
model/encoder=dropout_01_6blocks \
pre_conf.mask_ratio=0.7 \
batch_size=8192 eval_batch_size=8192 \
mixed_fp16=true num_workers=15 \
hydra.run.dir=output/Pre/BRFSS/tabret/dropout_01_6blocks/0.7
```

</details>

### Fine-tuning

To run the fine-tuning phase, please run the following command:

```bash
python main_fine.py fine_conf.pre_path={pre-trained model path} seed={n}
```


Make sure to replace `{pre-trained model path}` with the path to the previously trained model. You can modify the configurations for the fine-tuning phase in the `conf/fine.yaml` file.

Alternatively, you can run optimization during fine-tuning using Optuna by running the following command:

```bash
python main_optuna.py fine_conf.pre_path={pre-trained model path} seed={n}
```

Again, make sure to replace `{pre-trained model path}` with the path to the previously trained model.

*A specific example is shown below.*
<details><summary><em>[Click to expand]</em></summary>

<br>

```bash
python main_optuna.py data=Diabetes \
model=tabret model/encoder=dropout_01_6blocks \
fine_conf=ret/tabret \
fine_conf.pre_path=output/Pre/BRFSS/tabret/dropout_01_6blocks/0.7/checkpoints_pre/best_model \
study_name=Diabetes/tabret/1 seed=1 \
hydra.run.dir=outputs/Fine/Diabetes/tabret/1
```

</details>