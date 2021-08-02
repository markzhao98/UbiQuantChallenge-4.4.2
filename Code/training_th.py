import threading
import pandas as pd
import numpy as np
import time

import saver
from supp import *

from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor

class TrainingThread(threading.Thread):

    def __init__(self,name=None):
        threading.Thread.__init__(self,name=name)

    def run(self):
        if saver.first_time_flag:
            time.sleep(400)
            saver.first_time_flag = False
        
        """
        training data preparing
        """
        saver.locked = True
        data = pd.DataFrame(saver.training_df.copy())
        saver.locked = False
        
        data.rename(columns={0: 'tick', 1: 'stock', 2:'close', 3:'alpha1', 4:'alpha2', 5:'alpha3', 
                     6:'alpha4', 7:'alpha5', 8:'alpha6', 9:'alpha7', 
                     10:'alpha8', 11:'alpha9', 12:'alpha10', 13:'alpha11', 
                     14:'alpha12', 15:'alpha13', 16:'alpha14', 17:'alpha15', 
                     18:'alpha16', 19:'alpha17', 20:'alpha18', 21:'alpha19', 
                     22:'alpha20', 23:'alpha21', 24:'alpha22', 25:'alpha23'}, inplace=True)

        data['obj10'] = data[['stock', 'close']].groupby('stock') \
            .transform(lambda x: laggingF(abs2percF(x.values, 10), 1))
        
        data['obj5'] = data[['stock', 'close']].groupby('stock') \
            .transform(lambda x: laggingF(abs2percF(x.values, 5), 1))
        
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        y10_train = data['obj10'].values
        y5_train = data['obj5'].values
        X_train = data.iloc[:, 3:-2].values
        
        """TODO
        training processing
        """

        print("==== First Model Training Begins ====")

        gb10 = GradientBoostingRegressor(loss = 'huber', 
                               learning_rate = 0.25, 
                               n_estimators = 25, 
                               verbose = True, 
                               random_state = 2020)
        gb10.fit(X_train, y10_train)

        print("==== Second Model Training Begins ====")

        gb5 = GradientBoostingRegressor(loss = 'huber', 
                               learning_rate = 0.25, 
                               n_estimators = 25, 
                               verbose = True, 
                               random_state = 2020)
        gb5.fit(X_train, y5_train)

        saver.hf_loaded_model = gb10
        saver.hhf_loaded_model = gb5
        
        """
        process endding
        """

        print("==== New Models Updated ====")

        saver.training_flag = False
        return

