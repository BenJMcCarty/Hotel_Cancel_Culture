'''Name: Classification

Description: Functions designed to assist with the creation and evaluation of classification modeling.

By Ben McCarty (bmccarty505@gmail.com)'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics._plot.precision_recall_curve import PrecisionRecallDisplay


## Generating scores for later comparisons

def model_scores(model, X_train, y_train, X_test, y_test):

    y_hat_train = model.predict(X_train)
    prob_train = model.predict_proba(X_train)

    y_hat_test = model.predict(X_test)
    prob_test = model.predict_proba(X_test)

    base_train_ll = metrics.log_loss(y_train, prob_train).round(2)
    base_test_ll = metrics.log_loss(y_test, prob_test).round(2)

    base_train_score = model.score(X_train, y_train).round(2)
    base_test_score = model.score(X_test,y_test).round(2)


    return base_train_score, base_test_score, base_train_ll, base_test_ll

##
def plot_comparison_hist(data, feature, target = None, bins = 'auto', alt_name = None, save_fig=False):

    feature_name = feature.replace("_", " ").replace("'", "").title()
    
    if len(list(data[feature].unique())) > 10:
        ax = sns.histplot(data=data, x= data[feature] \
                           .value_counts(ascending=False),
                          hue = target, bins=bins)
    else:
        ax = sns.histplot(data=data, x= feature, hue = target, bins = bins)
    
    if list((data[feature].unique())) == [0,1]:
        plt.xticks([0, 1], ['No', 'Yes'])

    if alt_name == None:
        ax.set(title = f'Total Counts of Reviews for {feature_name}',
            xlabel = feature_name)
    else:
        ax.set(title = f'Total Counts of Reviews for {alt_name}',
            xlabel = alt_name)
        
    ax.legend(('Less than 4', '4 or Greater'),fontsize= 'medium', 
              title = 'Rating', title_fontsize = 'large', loc = 0);

    if save_fig == True:
        plt.savefig(f'{feature}_importance.png');

##
def plot_comparison_count(data,feature, target = 'review_scores_rating',
                    save_fig=False, print_target = None):

    feature_name = feature.replace("_", " ").title()
    
    
    if len(list(data[feature].unique())) > 10:
        ax = sns.countplot(data=data, x= feature,
                          hue = target)
    else:
        ax = sns.countplot(data=data, x= feature, hue = target)
    
    if list((data[feature].unique())) == [0,1]:
        plt.xticks([0, 1], ['No', 'Yes'])

    if print_target != None:
        target_name = target.replace("_", " ").title()
        ax.set(title = f'Comparing {feature_name} by {target_name}',
            xlabel = feature_name,ylabel = f'Count of {target_name}')
    else:
        ax.set(title = f'Comparing {feature_name} by Number of Reviews ',
            xlabel = feature_name,ylabel = f'Number of Reviews')


    ax.legend(('Less than 4', '4 or Greater'),fontsize= 'medium', 
              title = 'Rating', title_fontsize = 'large', loc = 0);

    if save_fig == True:
        plt.savefig(f'{feature}_importance.png');

##
def plot_depths(fitted_model, verbose = False):
    depths = []

    for i in fitted_model.estimators_:
        depths.append(i.get_depth())

    print(f'\nThe maximum depth is: {np.max(depths)}\n')

    ax = sns.histplot(depths)

    ax.set(title = 'Tree Depths Used in RandomForestClassifier',
           xlabel = 'Depths', ylabel = 'Number of Trees')
    ax.axvline(np.mean(depths), label = f'Mean: {np.mean(depths):.0f}',
               color='k')

    plt.legend(loc=0);
    plt.show
    
    if verbose == True:
        return depths

##
def plot_importances(model, X_train_df, count = 10, return_importances = False, model_name = None, save_fig = False):
    """Given a fitted classification model with feature importances, creates a 
    horizontal barplot of the top 10 most important features.

    Args:
        model (SKlearn model): pre-fit classification model
        X_train_df (pd.DataFrame: DF used for modeling - provides index to pair with values
        count (int, optional): Top # importances. Defaults to 10.
        return_importances (bool, optional): If True, returns the feature importances as a Series. Defaults to False.

    Returns:
        - Prints bar chart always
        If return_importances = True:
            pd.Series: feature importances as a Pandas 
    """    

    col_names =  list(map(lambda x: x.title().replace('_', ' '), list(X_train_df.columns)))

    importances = pd.Series(model.feature_importances_, index= col_names)
    
    importances = importances.sort_values(ascending = False)

    ax = importances[:count].plot(kind= 'barh')
    ax.set(title=f'Top {count} Most Important Predictors', xlabel='importance')
    
    plt.show()
    
    if save_fig == True:
        plt.savefig(f'{model_name}_feat_imp.png')

    plt.close()

    if return_importances == True:
        return importances


def cf_rpt_results(y_true, y_preds, metric):

        ## Getting clf rpt as dict -> df
    cr_df = pd.DataFrame(metrics.classification_report(y_true, y_preds,
                                                   output_dict=True))

    ## Rounding all values to 2 decimals
    cr_df = cr_df.applymap(lambda x: round(x, 2))

    ## adding blank col b/t 1, "accuracy"
    cr_df.insert(2,column=" ", value=" ")
    
    # Transposing df
    cr_df = cr_df.T

    ## Converting 'accuracy' values to strings to replace values w/ blank strings
    for value in range(len(cr_df.loc['accuracy':,])):
        cr_df.loc['accuracy'][value] = round(cr_df.loc['accuracy'][value],2)
        cr_df.loc['accuracy'][value] = str(cr_df.loc['accuracy'][value])

    ## Converting remaining values back to floats
    cr_df.loc['accuracy'][2:4] = pd.to_numeric(cr_df.loc['accuracy'][2:4])

    ## Copying "support" value from macro average
    cr_df.loc['accuracy','support'] = cr_df.loc['macro avg','support']

    cond = [metric == 'accuracy',
            metric == 'precision',
            metric == 'recall',
            metric == 'f1',
            metric == 'balanced accuracy',
            metric == 'balanced precision',
            metric == 'balanced recall']

    choice = [cr_df.loc['accuracy'][2],
              cr_df.loc['1'][1],
              cr_df.loc['1'][0],
              cr_df.loc['1'][2],
              cr_df.loc['macro avg'][1],
              cr_df.loc['macro avg'][0],
              cr_df.loc['macro avg'][1]
    ]

    return np.select(cond, choice, None)


##
def evaluate_classification(model,X_train, y_train, X_test, y_test,
                            metric = 'accuracy', verbose = True, 
                            cmap='Blues',normalize='true',figsize=(10,4), labels=None):                 
    """[summary]
    
    Adapted from:
    https://github.com/jirvingphd/Online-DS-FT-022221-Cohort-Notes/blob/master/Phase_3/topic_25_logistic_regression/topic_25_pt2_LogisticRegression_titanic-v2-SG.ipynb
    
    Args:
        model (model): Trained classification model
        X_train (Dataframe): X-train data
        y_train (Dataframe): y-train data
        X_test (Dataframe): X-test data
        y_test (Dataframe): y-test data
        metric (string): Text string to specify preferred performance metric
        verbose (bool, optional): 0, 1, 2. Defaults to True.
        labels (string, optional): Classes for predictions. Defaults to None.
        cmap (str, optional): Color pallette for confusion matrix. Defaults to 'Blues'.
        normalize (str, optional): Option for normalizing confusion matrix. Defaults to 'true'.
        figsize (tuple, optional): Sizing for ROC-AUC plot. Defaults to (10,4).

    Returns:
        [type]: [description]
    """                    

    print('\n|' + '----'*8 + ' Classification Metrics ' + '---'*11 + '--|\n')

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    prob_train = model.predict_proba(X_train)
    prob_test = model.predict_proba(X_test)  

    ### --- Scores --- ###

    train_score = cf_rpt_results(y_train, y_hat_train, metric)
    test_score = cf_rpt_results(y_test, y_hat_test, metric)

    print(f'Training {metric} score: {train_score}')
    print(f'Testing {metric} score: {test_score}')

    if verbose == 1:
        
        difference = train_score - test_score
        
        if difference > 0:
            print(f"\t- The training score is larger by {np.abs(difference):.2f} points.")
        elif difference < 0:
            print(f"\t- The training score is smaller by {np.abs(difference):.2f} points.")
        else:
            print(f"\t- The scores are the same size.")

    else:
        pass

    ### --- Log Loss --- ###

    print(f"\nTraining data log loss: {metrics.log_loss(y_train, prob_train):.2f}")
    print(f"Testing data log loss: {metrics.log_loss(y_test, prob_test):.2f}\n")

    if verbose == 2:
        if metrics.log_loss(y_test, prob_test) >= .66:
            print('\tThe log loss for the testing data is high, indicating a poorly-performing model.')
        elif metrics.log_loss(y_test, prob_test) <= .33:
            print('\tThe log loss for the testing data is low, indicating a well-performing model.')
        else:
            print('\tThe log loss for the testing data is moderate, indicating a weakly-performing model.')
    else:
        pass

    ### --- Clasification Reports --- ###
    
    print('\n\n|' + '----'*7 + ' Classification Report - Testing Data ' + '---'*8 + '-|\n')
    print(metrics.classification_report(y_test, y_hat_test,
                                    target_names=labels))

    fig, ax = plt.subplots(ncols=2, figsize = figsize)
    metrics.plot_confusion_matrix(model, X_test,y_test,cmap=cmap,
                            normalize=normalize, display_labels=labels,
                            ax=ax[0])

    curve = metrics.plot_roc_curve(model, X_test,y_test,ax=ax[1])
    curve.ax_.grid()
    curve.ax_.plot([0,1],[0,1], ls=':')
    plt.tight_layout()
    plt.show()

    print('\n|' + '----'*7 + ' Classification Report - Training Data ' + '---'*8 + '|\n')
    print(metrics.classification_report(y_train, y_hat_train,
                                target_names=labels))

    fig, ax = plt.subplots(ncols=2, figsize = figsize)
    metrics.plot_confusion_matrix(model, X_train,y_train,cmap=cmap,
                            normalize=normalize, display_labels=labels,
                            ax=ax[0])

    curve = metrics.plot_roc_curve(model, X_train,y_train,ax=ax[1])
    curve.ax_.grid()
    curve.ax_.plot([0,1],[0,1], ls=':')
    plt.tight_layout()
    plt.show()

    return None
