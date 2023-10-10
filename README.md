# Crystal: Introspective Commonsense Reasoning

This repo hosts the code for the paper, [Crystal: Introspective Reasoners Reinforced with Self-Feedback](https://arxiv.org/abs/2310.04921), presented at EMNLP 2023.

## Resources

**Model**: Our Crystal models are now on huggingface model hub! [[large]](https://huggingface.co/liujch1998/crystal-large) [[3b]](https://huggingface.co/liujch1998/crystal-3b) [[11b]](https://huggingface.co/liujch1998/crystal-11b)

Usage: Please see Crystal's [huggingface model card](https://huggingface.co/liujch1998/crystal-11b)

## Setup

Create and activate the Conda environment:
```
conda env create -f environment.yml
conda activate crystal
```

Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install).

### Download data

**Download the UQA data**: Go to `/data/` and run `python download_uqa.py`

## Training the Crystal model

The Crystal model is trained in two stages.
For simplicity, we show the process and code for training the Crystal-large model.

### Stage I: Imitation Learning

We trained this stage using 8x V100 GPUs, each has 32G memory.

First, generate silver knowledge from GPT-3.

If you would like to use our pre-generated data, you can download a copy of our pre-generated knowledge.
Go to `/data/` and run `gdown 10c2Mmjd3Nc6FgoWkUjqy-Ysy41D7qzch`
Alternatively, you can download the `knowledge_gkp.zip` file from [our Google Drive folder](https://drive.google.com/drive/folders/1K_lp3oKHk4AL1hrFrAPFo80pDkWQHyGs?usp=drive_link), unzip it and put it under `/data/`

Otherwise, you can generate the knowledge yourself by going to the `/scripts/` directory and run
```
sh generate_knowledge_gkp.sh
```
Remember to set the `OPENAI_API_KEY` envvar beforehand, and be ready to spend a lot of money ;)

Then, you can start Stage I training by going to the `/sbatch/` directory and run
```
sh train_imitation_large.sh
```
You can track the training in wandb.
The best model ckpt will be saved under `/runs_stageI/`.

### Stage II: Reinforcement Learning

We trained this stage using 8x V100 GPUs, each has 32G memory.

To train Stage II with the default setting, go to the `/sbatch/` directory and run
```
sh train_crystal_large.sh
```
Before you run this script, make sure to edit it and fill in the path to the best StageI model ckpt.
You can track the training in wandb.
The best model ckpt will be saved under `/runs/`.

## Running inference

To run inference with the default setting, go to the `/sbatch/` directory and run
```
sh eval_crystal_large.sh
```
This will evaluate the dev split of all seen datasets.
You can view the output knowledge in `[PATH_TO_MODEL_CKPT]/../knowledge/` and the inference results in `[PATH_TO_MODEL_CKPT]/../inference/`.

## Citation

If you find this repo useful, please cite our paper:
```
@article{Liu2023CrystalIR,
  title={Crystal: Introspective Reasoners Reinforced with Self-Feedback},
  author={Jiacheng Liu and Ramakanth Pasunuru and Hannaneh Hajishirzi and Yejin Choi and Asli Celikyilmaz},
  journal={ArXiv},
  year={2023},
  volume={abs/2310.04921}
}
```
