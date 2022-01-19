import time
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import *
# A custom-made library for reporting
from my_eval_functions import set_seeds, get_clf_eval, dingdong, printtimer
# Written by Lee HWANGBO, MD. Dec 2021.

##### BEGIN
print('Loading dataframe, base, and ensemble classifiers')
start_time = printtimer(time.time())
set_seeds(123)

# READS DF
df_final = load('df_final.joblib')
df = df_final[0]
df_label = df_final[1]
df_train, df_test, df_train_label, df_test_label = train_test_split (df, df_label, test_size=0.3, random_state=123)

# READS INDIVIDUAL BASE MODELS (Lv 0)
clflist = load('MortalityOutcomeModels.joblib')
clfnamelist = load('ClassifierNameList.joblib')
# READS STACKING ENSEMBLE MODEL (Lv 1)
ensemble_model = load('EnsembleModel.joblib')

### TO STDOUT
print('*****************************************************************************************')
print('               TRAINING SET\n')
print('=========================================================================================')
for i in range (0,len(clflist)):
    print('\n***** INDIVIDUAL MODEL (best): ', clfnamelist[i])
    get_clf_eval(df_train_label.tolist(), clflist[i].best_estimator_.predict_proba(df_train)[:, 1].tolist())
    printtimer(start_time)

print('********** Ensemble [KNN + XGB + SVM + NB + RF + ANN + LR] ******************************')
print('*****************************************************************************************\n\n')
get_clf_eval(df_train_label.tolist(), ensemble_model.predict_proba(df_train)[:,1].tolist())

################# VALIDATION #################
print('*****************************************************************************************')
print('               VALIDATION SET\n')
print('=========================================================================================')
for i in range (0,len(clflist)):
    print('\n***** INDIVIDUAL MODEL (best): ', clfnamelist[i])
    get_clf_eval(df_test_label.tolist(), clflist[i].best_estimator_.predict_proba(df_test)[:, 1].tolist())
    printtimer(start_time)
print('=========================================================================================')
print('********** Ensemble [KNN + XGB + SVM + NB + RF + ANN + LR] ******************************')
print('*****************************************************************************************\n\n')
get_clf_eval(df_test_label.tolist(), ensemble_model.predict_proba(df_test)[:,1].tolist())

printtimer(start_time)
dingdong()
