# A3C trading
Trading with recurrent actor-critic reinforcement learning

<object data="https://github.com/evgps/a3c_trading/blob/master/supply_materials/Full_UML.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/evgps/a3c_trading/blob/master/supply_materials/Full_UML.pdf">
        This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/evgps/a3c_trading/blob/master/supply_materials/Full_UML.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Configuration: `config.py`
This file contains all the pathes and gloabal variables to be set up

## Dataset: `load_data.py`
After setting `config.py` please run this file to download and preprocess the data need for training and evaluation

## Environment: `trader_gym.py`
OpenAI.gym-like environment class

## Model: `A3C_class.py`
This file is containing `AC_network`, `Worker` and `Test_Worker` classes

## Training: `A3C_training.py`
Run this file, preferrable in `tmux`. During training it will create files in `tensorboard_dir` and in `model_dir`

## Testing: `A3C_testing.ipynb`
`Jupyter notebook` contains all for picturing
