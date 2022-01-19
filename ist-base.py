import numpy as np
import xgboost as xgb
import time
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import *
from numpy import logspace
from joblib import load, dump
# My custom library
from my_eval_functions import set_seeds, report, get_clf_eval, dingdong, printtimer
# Written by Lee HWANGBO, MD. Dec 2021.

# DIVISION FOR EACH ML METHODS CODED with 10-fold CV and either grid or randomized search (multithreading enabled)
def runKNN(df_train, df_train_label, df_test, df_test_label, num_iter=64):
    # RUN KNN
    parameters = {'n_neighbors': np.linspace(1,500,500).astype(int)}
    knn_model = KNeighborsClassifier()
    clf = RandomizedSearchCV(knn_model, parameters, n_iter=num_iter, n_jobs=14,
                             cv=StratifiedKFold(n_splits=5, shuffle=True),
                             scoring='accuracy',
                             verbose=1, refit=True, random_state=123)
    clf.fit(df_train, df_train_label)
    proba = clf.predict_proba(df_test)
    proba = proba [:,1]
    proba = proba.tolist()
    pred = clf.predict(df_test)
    pred = pred.tolist()
    df_test_label = df_test_label.tolist()

    report(clf.cv_results_, n_top=5)
    get_clf_eval(df_test_label, pred, proba)
    return clf


def runXGB(df_train, df_train_label, df_test, df_test_label, num_iter=256):
    parameters = {'nthread':[1], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'eval_metric': ['auc'],
                  'learning_rate': np.logspace (-0.5,-2,30),
                  'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'min_child_weight': np.linspace(1,300,200).astype(int),
                  'subsample': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
                  'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'n_estimators': np.linspace(50,1000,25).astype(int),
                  'missing':[-999],
                  'seed': [123,321]}
    xgb_model = xgb.XGBClassifier(use_label_encoder=False) # An empty XGB classifier initiated
    num_iter_=num_iter
    clf = RandomizedSearchCV(xgb_model, parameters, n_iter=num_iter_, n_jobs=14,
                       cv=StratifiedKFold(n_splits=5, shuffle=True),
                       scoring='accuracy',
                       verbose=1, refit=True, random_state=123)
    clf.fit(df_train, df_train_label)
    proba = clf.predict_proba(df_test)
    proba = proba[:,1]
    proba = proba.tolist()
    pred = clf.predict(df_test)
    pred = pred.tolist()
    df_test_label = df_test_label.tolist()
    report(clf.cv_results_, n_top=5)
    get_clf_eval(df_test_label, pred, proba)
    return clf


def runSVM(df_train, df_train_label, df_test, df_test_label, num_iter=128):
    # {'C': [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 250, 500, 750, 1000],
    param   = {'C': logspace(-2, 3, num=100),
              'gamma': logspace(-4,4,num=100),
              'kernel': ['rbf'],
               'random_state': [123,321]
               }
    svm_model = SVC(probability=True)
    clf = RandomizedSearchCV(svm_model, param, n_iter=num_iter, n_jobs=15,  # Thread usual 14 + 1, additionally, empirical under Xeon Silver 4110 CPU 16 cores.
                       cv=StratifiedKFold(n_splits=5, shuffle=True),
                       scoring='roc_auc',
                       verbose=1, refit=True, random_state=123)
    clf.fit(df_train, df_train_label)
    # xgbproba = clf.predict_proba(df_test)
    proba = clf.predict_proba(df_test)
    proba = proba[:,1]
    proba = proba.tolist()
    pred = clf.predict(df_test)
    pred = pred.tolist()
    df_test_label = df_test_label.tolist()
    report(clf.cv_results_, n_top=5)
    get_clf_eval(df_test_label, pred, proba)
    return clf


def runNB(df_train, df_train_label, df_test, df_test_label):
    param = {'var_smoothing': logspace(1, -15, num=512)}
    nb_model = GaussianNB()
    set_seeds(123)
    clf = GridSearchCV (nb_model, param, n_jobs=14,
                       cv=StratifiedKFold(n_splits=5, shuffle=True),
                       scoring='accuracy',
                       verbose=1, refit=True)
    clf.fit(df_train, df_train_label)
    proba = clf.predict_proba(df_test)
    proba = proba[:,1]
    proba = proba.tolist()
    pred = clf.predict(df_test)
    pred = pred.tolist()
    df_test_label = df_test_label.tolist()
    report(clf.cv_results_, n_top=5)
    get_clf_eval(df_test_label, pred, proba)
    return clf


def runRF(df_train, df_train_label, df_test, df_test_label, num_iter=256):
    param = {'bootstrap': [True, False],
             'max_depth': np.linspace(1,20,20).astype(int),
             'max_features': ['auto', 'sqrt'],
             'min_samples_leaf': np.linspace(1,10,10).astype(int),
             'min_samples_split': np.linspace(1,10,10).astype(int),
             'n_estimators': np.linspace(30,2000,200).astype(int),
             'random_state': [123,321]}
    rf_model = RandomForestClassifier()
    clf = RandomizedSearchCV (rf_model, param,
                              n_iter=num_iter, n_jobs=14,
                              cv=StratifiedKFold(n_splits=5, shuffle=True),
                              scoring='accuracy',
                              verbose=1, refit=True, random_state=123)
    clf.fit(df_train, df_train_label)
    proba = clf.predict_proba(df_test)
    proba = proba[:,1]
    proba = proba.tolist()
    pred = clf.predict(df_test)
    pred = pred.tolist()
    df_test_label = df_test_label.tolist()
    report(clf.cv_results_, n_top=5)
    get_clf_eval(df_test_label, pred, proba)
    return clf


def runANN(df_train, df_train_label, df_test, df_test_label, num_iter=256):
    param = {'hidden_layer_sizes': [(2,),(3,),(4,),(5,), (6,), (7,), (8,), (9,), (10,),
                                    (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,)
                                    ],
             'activation': ['tanh', 'identity', 'logistic', 'relu'],
             'solver': ['sgd', 'adam', 'lbfgs'],
             'alpha': np.logspace(-1, -5, 100),
             'learning_rate': ['constant', 'adaptive'],
             'random_state':[123,321]}
    ann_model = MLPClassifier(max_iter=512)
    clf = RandomizedSearchCV (ann_model, param,
                              n_iter=num_iter, n_jobs=15,
                              cv=StratifiedKFold(n_splits=5, shuffle=True),
                              scoring='neg_brier_score',
                              verbose=1, refit=True,random_state=123)
    clf.fit(df_train, df_train_label)
    proba = clf.predict_proba(df_test)
    proba = proba[:,1]
    proba = proba.tolist()
    pred = clf.predict(df_test)
    pred = pred.tolist()
    df_test_label = df_test_label.tolist()
    report(clf.cv_results_, n_top=5)
    get_clf_eval(df_test_label, pred, proba)
    return clf


def runLR(df_train, df_train_label, df_test, df_test_label):
    param = {'solver': ['newton-cg', 'liblinear', 'saga', 'sag'],
             'C': logspace(3,-8,64)}
    lr_model = LogisticRegression()
    set_seeds(123)
    clf = GridSearchCV (lr_model, param,
                        n_jobs=14,
                        cv=StratifiedKFold(n_splits=5, shuffle=True),
                        scoring='accuracy',
                        verbose=1, refit=True)
    clf.fit(df_train, df_train_label)
    proba = clf.predict_proba(df_test)
    proba = proba[:,1]
    proba = proba.tolist()
    pred = clf.predict(df_test)
    pred = pred.tolist()
    df_test_label = df_test_label.tolist()
    report(clf.cv_results_, n_top=5)
    get_clf_eval(df_test_label, pred, proba)
    return clf


################################
## READ JOBLIB file (dataframe)
set_seeds(123)
df_joblib = load ('df_final.joblib')
df = df_joblib[0]
df_label = df_joblib[1]
df_train, df_test, df_train_label, df_test_label = train_test_split (df, df_label, test_size=0.3, random_state=123)

# NOW START ANALYSIS FOR EACH ML
start_time = printtimer(time.time())

print('----------KNN----------')
clf_knn = runKNN(df_train, df_train_label, df_test, df_test_label)
printtimer(start_time)

print('----------SVM----------')
clf_svm = runSVM(df_train,df_train_label,df_test,df_test_label,1024)
printtimer(start_time)

print('----------XGB----------')
clf_xgb = runXGB(df_train,df_train_label,df_test,df_test_label,4096)
printtimer(start_time)

print('----------Naive Bayes, Gaussian----------')
clf_nb = runNB(df_train,df_train_label,df_test,df_test_label)
start_time = printtimer(start_time)

print('----------Random Forest----------')
clf_rf = runRF(df_train,df_train_label,df_test,df_test_label,2048)
printtimer(start_time)

print('----------ANN----------')
clf_ann = runANN(df_train,df_train_label,df_test,df_test_label,4096)
printtimer(start_time)

print('----------Logistic Regression----------')
clf_lr = runLR(df_train,df_train_label,df_test,df_test_label)
printtimer(start_time)

##### SAVES clf #####
print('SAVING CLASSIFIERS')
clfnamelist = ['KNN', 'XGB', 'SVM', 'NB', 'RF', 'ANN', 'LR']
clfallD = [clf_knn, clf_xgb, clf_svm, clf_nb, clf_rf, clf_ann, clf_lr]
dump (clfnamelist, 'ClassifierNameList.joblib')
dump (clfallD,'MortalityOutcomeModels.joblib')

printtimer(start_time)
dingdong()
### EOF
