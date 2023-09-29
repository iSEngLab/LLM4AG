# LLM4Assertion

This repository contains the code needed to reproduce the experiments.

## How to set up the running environment

Please run the following commands to install the necessary packages for running the replication package.

```
pip install pipreqs
pipreqs /the/root/entry --encoding=utf-8 --force
pip install -r requirements.txt 
```

## How to run the replication package

We have organized the code into corresponding folders based on the research questions in the paper. You need to download the datasets provided in the paper, and then use the data processing scripts we have provided to generate datasets for each RQ. These datasets will be used for training and generation.

Each sub-folder contains a file named run.py. It contains the shell commands used for training and testing the corresponding model. To replicate the results of our paper, run the following commands:

```
cd /the entry you want
python run.py
```
