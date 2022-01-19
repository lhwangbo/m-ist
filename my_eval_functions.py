import winsound
import numpy as np
import time
import sklearn
import os
import random
from sklearn.metrics import *
# Written by Lee HWANGBO, MD. Dec 2021.

### SOUND DING DONG ###
def dingdong():
    duration = 500
    winsound.Beep(262, duration)
    winsound.Beep(330, duration)
    winsound.Beep(392, duration)
    winsound.Beep(524, duration)

### TIMER
def printtimer(t0):
    t1 = time.time()
    print('\n********** TIME ELAPSED (s)=', t1 - t0, '\n')
    return t0


### SET RANDOM SEED FIXED
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

### Hyperparameter function
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")

def ci95 (p_hat, n): ##### FOR PROPORTIONS - No more used, using bootstrap (naive) instead.
    half_interval = 1.96 * np.sqrt(p_hat * (1-p_hat) /n)
    output = '%f (%f, %f)' % (p_hat, p_hat - half_interval, p_hat + half_interval)
    return output

### EVALUATION FUNCTION
def get_clf_eval(y_test, y_prob, n_bootstraps=10000, verbosity=0):
    #####from confusion matrix calculate accuracy
    ### The BELOW shows REFINED results with ROC with argmax (Youden J index, i.e., max sens+spec-1 == tpr-fpr
    # PART I. CALCULATING AUC
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_test, y_prob)
    argmax_i = np.argmax(tpr-fpr)
    threshold_best = threshold [argmax_i]
    # Bootstrapping
    bootstrapped_auroc = []
    bootstrapped_accuracy = []
    bootstrapped_sensitivity = []
    bootstrapped_specificity = []
    bootstrapped_PPV = []
    bootstrapped_NPV = []
    bootstrapped_LRpos = []
    bootstrapped_LRneg = []
    bootstrapped_youdenJ = []
    bootstrapped_F1 = []
    bootstrapped_threshold = []
    rng = np.random.RandomState(1)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_test), len(y_test))
        y_test_temp = []
        y_prob_temp = []
        for k in range(0, len(y_test)):
            y_test_temp.append(y_test[indices[k]])
            y_prob_temp.append(y_prob[indices[k]])
        fpr_temp, tpr_temp, threshold_temp = sklearn.metrics.roc_curve (y_test_temp, y_prob_temp)
        argmax_temp = np.argmax(  tpr_temp-fpr_temp - ( abs(tpr_temp-(1-fpr_temp)) * 0.25 ) ) # argmax(Youden J)
        threshold_temp_best = threshold_temp[argmax_temp]
        y_pred_temp = np.where(y_prob_temp >= threshold_temp_best, 1, 0)
        cm_temp = confusion_matrix (y_test_temp, y_pred_temp)
        bootstrapped_auroc.append(roc_auc_score(y_test_temp, y_prob_temp))
        bootstrapped_youdenJ.append(tpr_temp[argmax_temp]-fpr_temp[argmax_temp])
        bootstrapped_accuracy.append((cm_temp[0, 0]+cm_temp[1, 1])/(np.sum(cm_temp)))
        bootstrapped_sensitivity.append(tpr_temp[argmax_temp])
        bootstrapped_specificity.append(1-fpr_temp[argmax_temp])
        bootstrapped_PPV.append(cm_temp[1,1]/(cm_temp[1,1]+cm_temp[0,1]))
        bootstrapped_NPV.append(cm_temp[0,0]/(cm_temp[0,0]+cm_temp[1,0]))
        bootstrapped_LRpos.append( tpr_temp[argmax_temp] / fpr_temp[argmax_temp] )
        bootstrapped_LRneg.append( (1 - tpr_temp[argmax_temp]) / (1 - fpr_temp[argmax_temp]) )
        bootstrapped_F1.append (cm_temp[1,1] / (cm_temp[1,1] + ((cm_temp[0,1]+cm_temp[1,0])/2) ))
        bootstrapped_threshold.append (threshold_temp_best)
    bootstrapped_auroc.sort()
    bootstrapped_youdenJ.sort()
    bootstrapped_accuracy.sort()
    bootstrapped_sensitivity.sort()
    bootstrapped_specificity.sort()
    bootstrapped_PPV.sort()
    bootstrapped_NPV.sort()
    bootstrapped_LRpos.sort()
    bootstrapped_LRneg.sort()
    bootstrapped_F1.sort()
    bootstrapped_threshold.sort()

    metric_names = ['AUROC', 'Youden J', 'Accuracy',
                    'Sensitivity', 'Specificity',
                    'PPV', 'NPV', 'LR+', 'LR-', 'F1']
    metric_bootstrapped = [bootstrapped_auroc, bootstrapped_youdenJ, bootstrapped_accuracy,
                           bootstrapped_sensitivity, bootstrapped_specificity,
                           bootstrapped_PPV, bootstrapped_NPV, bootstrapped_LRpos, bootstrapped_LRneg, bootstrapped_F1]

    auc_ci95_lower = bootstrapped_auroc[int(0.025*n_bootstraps)]
    auc_ci95_upper = bootstrapped_auroc[int(0.975*n_bootstraps)]
    youdenJ_ci95_lower = bootstrapped_youdenJ[int(0.025*n_bootstraps)]
    youdenJ_ci95_upper = bootstrapped_youdenJ[int(0.975*n_bootstraps)]
    accuracy_ci95_lower = bootstrapped_accuracy[int(0.025*n_bootstraps)]
    accuracy_ci95_upper = bootstrapped_accuracy[int(0.975*n_bootstraps)]
    sensitivity_ci95_lower = bootstrapped_sensitivity[int(0.025*n_bootstraps)]
    sensitivity_ci95_upper = bootstrapped_sensitivity[int(0.975*n_bootstraps)]
    specificity_ci95_lower = bootstrapped_specificity[int(0.025*n_bootstraps)]
    specificity_ci95_upper = bootstrapped_specificity[int(0.975*n_bootstraps)]
    ppv_ci95_lower = bootstrapped_PPV[int(0.025*n_bootstraps)]
    ppv_ci95_upper = bootstrapped_PPV[int(0.975*n_bootstraps)]
    npv_ci95_lower = bootstrapped_NPV[int(0.025*n_bootstraps)]
    npv_ci95_upper = bootstrapped_NPV[int(0.975*n_bootstraps)]
    LRpos_ci95_lower = bootstrapped_LRpos[int(0.025*n_bootstraps)]
    LRpos_ci95_upper = bootstrapped_LRpos[int(0.975*n_bootstraps)]
    LRneg_ci95_lower = bootstrapped_LRneg[int(0.025*n_bootstraps)]
    LRneg_ci95_upper = bootstrapped_LRneg[int(0.975*n_bootstraps)]
    F1_ci95_lower = bootstrapped_F1[int(0.025*n_bootstraps)]
    F1_ci95_upper = bootstrapped_F1[int(0.975*n_bootstraps)]
    threshold_best = bootstrapped_threshold[int(0.50*n_bootstraps)]
#########################################
    youdenJ = tpr [argmax_i] - fpr [argmax_i]
    AUC_refined = roc_auc_score(y_test, y_prob)
    y_pred_best = np.where(y_prob >= threshold_best, 1, 0)
    cm = confusion_matrix (y_test, y_pred_best)
    accuracy_r = accuracy_score(y_test, y_pred_best)
    recall_r = recall_score(y_test, y_pred_best)
    specificity_r = 1 - fpr[argmax_i]
    F1_r = f1_score(y_test, y_pred_best)
    ppv_r = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    npv_r = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    lrpos_r = recall_r / fpr[argmax_i]
    lrneg_r = (1 - recall_r) / (1 - fpr[argmax_i])
    ##### IF PRINTOUT NOT REQUIRED THEN
    if verbosity==-1:
        return metric_names, metric_bootstrapped
        #####
    # PART II. PRINTING OUT
    print('**********classifier.predict_proba() -> argmax (Youden J index)-ROC ANALYSIS**********')
    print('***** AUC = ', AUC_refined)
    print('          95CI by bootstrapping 10000 times: (%f, %f)' % (auc_ci95_lower, auc_ci95_upper))
    print('***** Thresholod (best Youden J) = ', threshold_best)
    print('***** Youden J = %f (%f to %f)' % (youdenJ, youdenJ_ci95_lower, youdenJ_ci95_upper))
    print('***** Accuracy (95CI) = %f (%f to %f)' % (accuracy_r, accuracy_ci95_lower, accuracy_ci95_upper))
    print('***** Sensitivity (95CI) = %f (%f to %f)' % (recall_r, sensitivity_ci95_lower, sensitivity_ci95_upper))
    print('***** Specificity (95CI) = %f (%f to %f)' % (specificity_r,
          specificity_ci95_lower, specificity_ci95_upper))
    print('***** PPV (95CI) = %f (%f to %f)' % (ppv_r, ppv_ci95_lower, ppv_ci95_upper))
    print('***** NPV (95CI) = %f (%f to %f)' % (npv_r, npv_ci95_lower, npv_ci95_upper))
    print('***** LR pos (95CI) = %f (%f to %f)' % (lrpos_r, LRpos_ci95_lower, LRpos_ci95_upper))
    print('***** LR neg (95CI) = %f (%f to %f)' % (lrneg_r, LRneg_ci95_lower, LRneg_ci95_upper))
    print('***** F1 (95CI) = %f (%f to %f)' % (F1_r, F1_ci95_lower, F1_ci95_upper))
    print('** CONFUSION MATRIX REFINED **\n', cm)
    if verbosity == 0:
        return metric_names, metric_bootstrapped

    # ELSE IF VERBOSITY > 0
    # FOR USE on R project (pROC, DeLong's method)
    print('-----All Prob-----')
    print(y_prob)
    print('All true y')
    print(y_test)
    return metric_names, metric_bootstrapped