# Directories
TENSORBOARD_DIR = '/mnt/a3c_data/tb/'
MODEL_DIR = '/mnt/a3c_data/model/'
DATA_DIR = '/mnt/a3c_data/data/'
PLOTS_DIR = '/mnt/a3c_data/plots/'

# Data files
POSTFIX = "['RTS-12.15']w5k"
TEST_POSTFIX = "['RTS-3.16', 'RTS-6.16']w5k"
POSTFIX_REAL = "['RTS-12.15']"
TEST_POSTFIX_REAL = "['RTS-3.16', 'RTS-6.16']"

# Network architechture
EXTRA_DENSE = False
N_HIDDEN = 64
DROPOUT = True
TRAINING = False
COOL_V = True
COOL_A = False
DEP = 1
GAMMA = .8
LOAD_MODEL = False
LR = 1e-4
COMISSION = 20  # rubls
PRICE_MAG = 1 / 5000  # среднне различие между макс и мин ценой за 5000 шагов
