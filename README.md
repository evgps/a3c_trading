# a3c_trading
Trading with recurrent actor-critic reinforcement learning

## Configuration: `config.py`
This file contains all the pathes and gloabal variables to be set up

## Dataset: `load_data.py`
After setting `config.py` please run this file to download and preprocess the data need for training and evaluation

## Training: `train.py`
Run this file, preferrable in `tmux`. During training it will create files in `tensorboard_dir` and in `model_dir`

## Testing: `test.ipynb`
`Jupyter notebook` contains all for picturing
