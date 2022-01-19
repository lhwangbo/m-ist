# m-ist
Machine-learning derived mortality prediction model for patients with acute ischaemic stroke based on IST-1 dataset

Using seven base learner implementations of Scikit Learn, my colleagues and I have developed this stacking ensemble learning model to predict six-month mortality for ischaemic stroke patients. This work has not been peer-reviewed yet. And when the peer-review process is complete, I will change this README file accordingly.

We have used three libraries, Scikit Learn, XGBoost, and Pandas along with their dependencies

I will update this file as soon as the manuscript gets published.

Our team developed both base and ensemble ML models using Scikit-learn (version 1.0.1) and XGBoost (version 1.5.0) libraries on Python (version 3.9.7).

16th Jan 2022
Lee Hwangbo, MD

-----
Python source files

ist-base.py // Seven base ML methods are hyperparameter-tuned. The seven individual learners are exported into a file (MortalityOutcomeModels.joblib).
ist-ensemble-stack.py // Stacking ensemble learner using an MLP as a learner. This is exported into a file (EnsenbleModel.joblib)
ist-study-table-roc.py // Coded for our research article. The output to stdout contains comprehensive statistics regarding performance on both train and validation sets. 
my_eval_functions.py // A collection of helper functions.

Joblib files
ClassifierNameList.joblib // Contains classifier names of seven base ML methods
df_final.joblib // Contains data of both train and validation sets (we used random seed '123' for train-test-split)
EnsembleModel.joblib // Final stacking ensemble learner
MortalityOutcomeModels.joblib // Seven individual base learners

-----
