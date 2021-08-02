import pandas as pd
import numpy as np

import pickle

locked = False
training_flag = False
first_time_flag = True

rolling_df = np.asarray(pd.read_csv('data/CONTEST_DATA_MIN_SP_1.csv', header = None) \
    .iloc[-500*21:]).reshape(21, 500, 48).transpose((1, 2, 0))

training_df = pd.read_csv('/root/AlphasData/alphas23.csv').set_index('Unnamed: 0')

hf_loaded_model = pickle.load(open('Models/gb10_final.sav', 'rb'))
hhf_loaded_model = pickle.load(open('Models/gb5_final.sav', 'rb'))

