import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from matplotlib.font_manager import FontProperties

# ---------------------------------------------------------------------------- #
#        All the tools for restore and for visualize the models's result       #
# ---------------------------------------------------------------------------- #



def make_confusion_matrices(Ytest, y_models=None, model_names=None, fig_axs=None, cms=None):
    '''
        Given a ground truth and a list of predictions or confusion matrices,
        it returns the corresponding heatmaps.
    '''
    if fig_axs is None:
        fig, axs = plt.subplots(1, len(y_models), sharex='col', sharey='row', figsize=(30,5))
    else:
        fig, axs = fig_axs

    if cms is None:
        if y_models is None:
            raise ValueError(f"You must pass either a list of confusion matrixes OR a list of predictions!")
        cms = []
        for i in range(len(y_models)):
            cms.append(confusion_matrix(Ytest, y_models[i], normalize='true'))
    
    tot_matrixes = len(cms) if cms is not None else len(y_models)
    print(tot_matrixes)     
    for i in range(tot_matrixes):
        sns.heatmap(
            cms[i],
            annot=True,
            ax=axs.flat[i] if tot_matrixes > 1 else axs,
            cbar= (tot_matrixes==1),
            vmin=0,
            vmax=max(map(np.max, cms)),
            cmap="YlGn")

        if y_models is not None:
            accuracy = f"accuracy:{round(accuracy_score(Ytest, y_models[i]), 3)}"
        else:
            accuracy = ''
        
        if tot_matrixes >1:
            axs.flat[i].set_title(f"{model_names[i]} {accuracy}", fontsize=16)
        else:
            axs.set_title(f"{model_names[i]} {accuracy}", fontsize=16)

    if tot_matrixes > 1:
        fig.colorbar(axs.flat[0].collections[0], ax=axs.ravel().tolist())

def get_report_dfs(y_models, Ytest, target_names=['positive', 'negative', 'normal']):
    '''
        Given a list of predictions and a groung truth returns a list of dataframes
        (one for prediction) that are the classification report of such predicions
    '''

    labels = np.arange(len(target_names))
    report_dfs = []
    for i in range(len(y_models)):
        clf_report = classification_report(Ytest,
                                        y_models[i],
                                        labels=labels,
                                        target_names=target_names,
                                        output_dict=True)
        report_dfs.append(pd.DataFrame((clf_report)))
    return report_dfs

def make_classification_report(Ytest, y_models=None, model_names=None, target_names=['positive', 'negative', 'normal'], fig_axs=None, report_dfs=None, out_path=None):

    if fig_axs is None:
        fig, axs = plt.subplots(1, len(y_models), sharex='col', sharey='row', figsize=(30,5))
    else:
        fig,axs = fig_axs
    
    if report_dfs is None:
        if y_models is None:
            raise ValueError(f"You must pas either a list of report dataframes or a list of y models")
        report_dfs=get_report_dfs(y_models, Ytest, target_names)

    tot_reports = len(report_dfs)

    for i in range(tot_reports):
        sns.heatmap(
            report_dfs[i].iloc[:-1,:].T,
            annot=True,
            ax=axs.flat[i] if tot_reports > 1 else axs,
            cbar=(tot_reports==1),
            vmin=0,
            vmax=max(map(lambda df: df.iloc[:-1,:].values.max(), report_dfs)),
            cmap="YlGn"
        )
        if tot_reports > 1:
            axs.flat[i].set_title(model_names[i])
        else:
            axs.set_title(model_names[i])


    if tot_reports > 1:
        fig.colorbar(axs.flat[0].collections[0], ax=axs.ravel().tolist())

    if out_path is not None:
        plt.savefig(out_path)


def load_from_dir(base, only_weight=False):
    '''
        Given a directory, returns the results stored in that directory in a more
        managable way

        Parameters
        ----------
            base: the path of the directory where the log file can be found
            only_weight: boolean if you want to read only the weight log (weight of the model)
        Returns
        -------
            Ytest: The ground truth Y where the model has been tested
            MODEL_NAMES: A list containing the names of the models
            Y_MODELS: A list of list containing the prediction of each model
            WEIGHTED_IDFS: A list of dataframe in the form:
                idx=word, class0_weights, class1_weights,class3_weights 

    '''
    MODEL_NAMES = []
    Y_MODELS = []
    WEIGHTED_IDFS = []
    for fname in os.listdir(base):
        if fname=='predictions.csv' and not only_weight:
            pred = pd.read_csv(os.path.join(base, fname), header='infer')

        elif '_weighted_idf.csv' in fname:
            model_name = fname[:-len('_weighted_idf.csv')]
            MODEL_NAMES.append(model_name)
            if not only_weight:
                Y_MODELS.append(pred[model_name].values)
            WEIGHTED_IDFS.append(pd.read_csv(os.path.join(base, fname), header='infer', index_col=0))
    if not only_weight:
        Ytest = pred.Ytest.values
    else:
        Ytest = None
    return Ytest, MODEL_NAMES, Y_MODELS, WEIGHTED_IDFS


def get_matrices_from_dir(base_dir, case_sensitives=[True, False], stop_words= [True, False], lemmatized=[True, False], out_path=None):
    '''
        Given a directory and a combination of features plots the confusion matrixes of the 
        stored results in the directory.
        Parameters
        ----------
        base_dir: The directory where the log files can be found
        case_sensitive: list of boolean selecting which file consider
        stop_words: list of boolean selecting which file consider
        lemmatized: list of boolean selecting which file consider
    '''
    if '20k' in base_dir:
        ncol = 3
    else:
        ncol=2

    n_row = len(case_sensitives)*len(stop_words)*len(lemmatized)
    fig, axs = plt.subplots(n_row, ncol, sharex='col', sharey='row', figsize=(30,5*n_row))

    cms = []
    tot_names = []
    y_labels = []

    for dir_name in os.listdir(base_dir):
        
        if '20k' in dir_name or 'total' in dir_name:
            cs = 'sensitive' in dir_name
            lemm = 'lemm' in dir_name
            stop = 'stop' in dir_name

            if (cs in case_sensitives) and (stop in stop_words) and (lemm in lemmatized):
                Ytest, MODEL_NAMES, Y_MODELS, _ = load_from_dir(os.path.join(base_dir, dir_name))
                for i in range(len(Y_MODELS)):
                    cms.append(confusion_matrix(Ytest, Y_MODELS[i], normalize='pred'))
                tot_names+=MODEL_NAMES
                
                row_title = 'case sensitive' if cs else 'lower case'
                row_title += ' lemmatized' if lemm else ''
                row_title += ((' with ' if stop else ' without ') + 'stopwords')
                y_labels.append(row_title)

    print(len(cms))
    make_confusion_matrices(Ytest, None, tot_names, cms=cms, fig_axs=(fig, axs))

    for i in range(len(y_labels)):
        axs.flat[i*ncol].set_ylabel(y_labels[i])

    if out_path is not None:
        plt.savefig(out_path)



def get_reports_from_dir(base_dir, case_sensitives=[True, False], stop_words= [True, False], lemmatized=[True, False], out_path=None):
    '''
        Given a directory and a combination of features plots the classification report of the 
        stored results in the directory.
        Parameters
        ----------
        base_dir: The directory where the log files can be found
        case_sensitive: list of boolean selecting which file consider
        stop_words: list of boolean selecting which file consider
        lemmatized: list of boolean selecting which file consider
    '''
    
    if '20k' in base_dir:
        ncol = 3
    else:
        ncol=2
    n_row = len(case_sensitives)*len(stop_words)*len(lemmatized)
    fig, axs = plt.subplots(n_row, ncol, sharex='col', sharey='row', figsize=(30,5*n_row))
    report_dfs = []
    tot_names = []
    y_labels = []

    for dir_name in os.listdir(base_dir):
        if '20k' in dir_name or 'total' in dir_name:

            cs = 'sensitive' in dir_name
            lemm = 'lemm' in dir_name
            stop = 'stop' in dir_name
            if (cs in case_sensitives) and (stop in stop_words) and (lemm in lemmatized):

                Ytest, MODEL_NAMES, Y_MODELS, _ = load_from_dir(os.path.join(base_dir, dir_name))

                report_dfs += get_report_dfs(Y_MODELS, Ytest)
                print(report_dfs[-1])
                tot_names+=MODEL_NAMES

                
                row_title = 'case sensitive' if cs else 'lower case'
                row_title += ' lemmatized' if lemm else ''
                row_title += ((' with ' if stop else ' without ') + 'stopwords')
                y_labels.append(row_title)

    print(len(report_dfs))
    make_classification_report(Ytest, y_models=None, model_names= tot_names, target_names=['positive', 'negative', 'normal'], fig_axs=(fig, axs), report_dfs=report_dfs)


    for i in range(len(y_labels)):
        axs.flat[i*ncol].set_ylabel(y_labels[i])

    if out_path is not None:
        plt.savefig(out_path)


def get_weighted_plots_from_dir(base_dir):
    _, MODEL_NAMES, _, WEIGHTED_IDFS = load_from_dir(base_dir, True)
    fig, axs = plt.subplots(len(WEIGHTED_IDFS), 1, figsize=(30,5*len(WEIGHTED_IDFS)), sharex=False,constrained_layout=True)
    prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
    plt.rcParams['font.family'] = prop.get_family()

    # For each dfataframe of each class
    for i in range(len(WEIGHTED_IDFS)):
        # Plot the relative classe's weights plot
        get_multiclass_plot(
            pd.DataFrame(
                WEIGHTED_IDFS[i].loc[
                get_params_index(
                            WEIGHTED_IDFS[i], WEIGHTED_IDFS[i].columns[-3:], 50)
                    ].iloc[:,-3:]
                ),
                title=MODEL_NAMES[i],
                ax=axs.flat[i] if len(WEIGHTED_IDFS)>1 else axs)


# ---------------------------------------------------------------------------- #
# Playing with the weight of the models

#The three following methods takes as input a dataframe that has the unique words
# Of the embedded vector as index and return a dataframes where is added
# the columns containing the weight that a specific model has given to such words
def get_NB_weighted_df(df, model, model_name=''):
    ret = df[:]
    for i in range(3):
        ret[f'class_{i}_weights'+f'_{model_name}' if model_name != '' else ''] = model.feature_log_prob_[i, :]
    return ret


def get_LR_weighted_df(df, model):
    ret = df[:]
    for i in range(model.coef_.shape[0]):
        ret[f'class_{i}_LR_importance'] = model.coef_[i,:]
    return ret


def get_rfc_weighted_df(df, model):
    ret = df[:]
    ret[f'feature_rfc_importance'] = model.feature_importances_
    return ret

# ---------------------------------------------------------------------------- #
# Plotting the weights

def get_params_index(df, class_names, n):
    '''
        Returns the top n indexes for a three columns dataframe

        Parameters
        ----------
            df: The dataframe from which extract the indexes
            class_names: The list of the classes from which take the first n results
            n: The numer of entries to be considered
    '''
    df0 = df.sort_values(by=class_names[0], ascending=False).head(n)[class_names[0]]
    df1 = df.sort_values(by=class_names[1], ascending=False).head(n)[class_names[1]]
    df2 = df.sort_values(by=class_names[2], ascending=False).head(n)[class_names[2]]
    return np.unique(np.concatenate((df0.index, df1.index, df2.index)))


def get_multiclass_plot(values_to_plot, title=None, ax=None):
    '''
        Given a dataframe with three or one columns, it plots the value of each columns
        as y values and uses the index of the dataframe as x labels
        
        Parameters
        ----------
            values_to_plot: the dataframe with header 'parameter' or ['positive', 'negative', 'neutral']
            title: the title of the plot
            ax: the axes over which plot
    '''
    n_val = len(values_to_plot)
    n_classes = values_to_plot.shape[-1]

    if ax is None:
        _, ax = plt.subplots(figsize=(30,10))
    
    labels = ['parameter'] if n_classes == 1 else ['positive', 'negative', 'neutral']
    for i in range(n_classes):
        ax.plot(np.arange(n_val), values_to_plot.iloc[:, i], label=labels[i])
        _ = ax.set_xticks(ticks= np.arange(n_val))
        _ = ax.set_xticklabels(labels=values_to_plot.index.values, rotation=70)
    
    
    ax.legend(loc='upper center', fontsize='large', numpoints=5)
    ax.set_title(title, fontsize=20)