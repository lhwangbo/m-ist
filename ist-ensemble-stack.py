import winsound
import numpy as np
import time
import sklearn
import os
import random
from joblib import dump, load
from sklearn.metrics import *
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import *
from my_eval_functions import set_seeds, get_clf_eval, dingdong, printtimer
# Written by Lee HWANGBO, MD. Dec 2021.

def ensemble_stack(level_zero, x, y_label):
    set_seeds(0)
    clf = StackingClassifier (estimators=level_zero,
                                final_estimator = MLPClassifier(hidden_layer_sizes=(5,),
                                                                random_state=123,
                                                                max_iter=100000),
                                  cv=5,
                                  n_jobs=-1,
                                  passthrough=False
                                  )
    clf.fit(x,y_label)
    return clf

##### BEGIN

print('Starting Ensemble Learning')
start_time = printtimer(time.time())
set_seeds(123)

##### READING DF #####
df_final = load('df_final.joblib')
df = df_final[0]
df_label = df_final[1]
df_train, df_test, df_train_label, df_test_label = train_test_split (df, df_label, test_size=0.3, random_state=123)

# READS INDIVIDUAL LEVEL 0 MODELS
clflist = load('MortalityOutcomeModels.joblib')
clfnamelist = load('ClassifierNameList.joblib')
models = []

################# ENSEMBLE TRAINING ###############
for i in range (0,len(clflist)):
    models.append( (clfnamelist[i], clflist[i].best_estimator_) )

ensemble_model = ensemble_stack(models, df_train, df_train_label)
dump(ensemble_model, 'EnsembleModel.joblib')

### TO STDOUT

################# VALIDATION #################
print('*****************************************************************************************')
print('               VALIDATION SET')
print('*****************************************************************************************')
print('********** Ensemble [KNN + XGB + SVM + NB + RF + ANN + LR] ******************************')
print('*****************************************************************************************\n\n')
get_clf_eval(df_test_label.tolist(), ensemble_model.predict_proba(df_test)[:,1].tolist())

printtimer(start_time)

# dingdong()
