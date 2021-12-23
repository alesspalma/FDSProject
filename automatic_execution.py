import preprocessing
import manage_results
import training_tools
import os
import pandas as pd
from time import time


# ---------------------------------------------------------------------------- #
#        Two methods to automatically train models and save the results        #
# ---------------------------------------------------------------------------- #


def save_run_metrics(model_names,y_models, Xtest, Xtrain, Ytest, Ytrain, weighted_idfs, base_path):
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    elif not os.path.isdir(base_path):
        raise ValueError(f"base_path attribute must be a directory or not exist. {base_path} is a file")

    predictions = pd.DataFrame()
    predictions['Xtest'] = Xtest
    predictions['Ytest'] = Ytest
    
    for i in range(len(y_models)):
        predictions[model_names[i]] = y_models[i]
        weighted_idfs[i].to_csv(os.path.join(base_path, f"{model_names[i]}_weighted_idf.csv"))
    
    predictions.to_csv(os.path.join(base_path, 'predictions.csv'), index=False)
    
    train = pd.DataFrame()
    train['Xtrain'] = Xtrain
    train['Ytrain'] = Ytrain
    train.to_csv(os.path.join(base_path, 'train_set.csv'), index=False) 


def exe(in_path, out_path, lower, lemma, set_size=None, logistic=False):
    df  = preprocessing.read_dataset(in_path)
    if lower:
        preprocessing.lower_case(df)
    if lemma:
        preprocessing.lemmatize(df)
    
    word_freq = pd.Series(' '.join(df['text']).split()).value_counts()
    word_freq= word_freq[(word_freq>4)]
    preprocessing.clean_dataset(df, word_freq)

    df = df[df.text != ''].reset_index(drop=True)

    idf = preprocessing.compute_idf(df.text.values)
    df['tf_idf'] = df.text.apply(lambda d: preprocessing.tf_idf(d, idf))
    

    Xtrain, Xtest, Ytrain, Ytest = preprocessing.get_splitted_dataset(df, x_column= 'tf_idf', y_column='y', num_entries = set_size)
    print('obtained test and train set')
    if logistic:
        print("Starting to create logistic regression model...")
        start=time()
        lr_model = training_tools.run_logistic_regression(Xtrain, Ytrain)
        print(f"Fitting...")
        y_model_LR = lr_model.predict(Xtest)             # 4. predict on new data
        print(f"Done in {round((time()-start), 3)} seconds\n")    
        Y_MODELS = [y_model_LR]
        MODEL_NAMES = ['Logistic regression']
        WEIGHTED_IDFS = [manage_results.get_LR_weighted_df(idf, lr_model)]
    else:
        Y_MODELS = []
        MODEL_NAMES = []
        WEIGHTED_IDFS = []

    print("Starting to create complement NB model...")
    start=time()
    complement_NB_model = training_tools.run_complement_NB(Xtrain, Ytrain)
    print(f"Fitting...")
    y_model_complement_NB = complement_NB_model.predict(Xtest)             # 4. predict on new dat
    print(f"Done in {round((time()-start), 3)} seconds\n")

    print("Starting to create multinomial NB model...")
    start=time()
    multi_NB_model = training_tools.run_multi_NB(Xtrain, Ytrain)
    print(f"Fitting...")
    y_model_multi_NB = multi_NB_model.predict(Xtest)             # 4. predict on new data
    print(f"Done in {round((time()-start), 3)} seconds\n")


    Y_MODELS = [y_model_complement_NB, y_model_multi_NB] + Y_MODELS
    MODEL_NAMES = ['Complement Naive Bayes', 'Multinomial Naive Bayes'] + MODEL_NAMES

    complement_NB_params = manage_results.get_NB_weighted_df(idf, complement_NB_model, 'complement_NB')
    multi_NB_params = manage_results.get_NB_weighted_df(idf, multi_NB_model, 'multinomial_NB')

    WEIGHTED_IDFS = [complement_NB_params, multi_NB_params] + WEIGHTED_IDFS

    Xtrain_list = list(Xtrain)
    Xtest_list = list(Xtest)
    save_run_metrics(MODEL_NAMES, Y_MODELS, Xtest_list, Xtrain_list, Ytest, Ytrain, WEIGHTED_IDFS, out_path)
    print('saved')