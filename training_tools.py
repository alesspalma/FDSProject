from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import numpy as np

# ---------------------------------------------------------------------------- #
#             All the tools for manage the dataset and train models            #
# ---------------------------------------------------------------------------- #


def get_splitted_dataset(df, x_column, y_column, num_entries=None):

    if num_entries is not None and num_entries < len(df):
        rnd_idx = np.random.choice(len(df), num_entries, replace=False)
    else:
        rnd_idx = df.index
    X = df.iloc[rnd_idx][x_column].values
    X = np.concatenate(X).reshape(len(X),-1)
    Y = df.iloc[rnd_idx][y_column].values

    return train_test_split(X, Y, train_size=0.8, stratify=Y)


def run_complement_NB(Xtrain, Ytrain):
    complement_NB_model =  ComplementNB()                       
    complement_NB_model.fit(Xtrain, Ytrain)                  
    return complement_NB_model


def run_gaussian_NB(Xtrain, Ytrain):
    gaussian_NB_model =  GaussianNB()                       
    gaussian_NB_model.fit(Xtrain, Ytrain)                  
    return gaussian_NB_model


def run_multi_NB(Xtrain, Ytrain):
    multi_NB_model =  MultinomialNB()                       
    multi_NB_model.fit(Xtrain, Ytrain)                
    return multi_NB_model


def run_logistig_regression(Xtrain, Ytrain):
    model_LR = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model_LR.fit(Xtrain, Ytrain)                  
    return model_LR


def run_rfc(Xtrain, Ytrain):
    cpus = multiprocessing.cpu_count()
    rfc_model =RandomForestClassifier(n_estimators=100, n_jobs=cpus)
    rfc_model.fit(Xtrain,Ytrain)
    return rfc_model
