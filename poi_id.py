

import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import logging
import time
import datetime
import math

from scipy import stats
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
import sklearn.pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif # Features are numerical and sometimes negative
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import combinations
from sklearn import tree
from sklearn import decomposition

import numpy as np

### User Functions
def log_info(message):
    ts = time.time()
    logging.info(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')+" "+message)

def init_logging():
    logging.basicConfig(format="%(message)s",
                        level=logging.INFO,
                        filename="history.log",
                        filemode="w")

def plotGridSearchCV(model, scoring, results, parameter):

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV for Classifier " + model)
    plt.xlabel(parameter)
    plt.ylabel("Score")
    ax = plt.axes()

    X_axis = np.array(results["param_"+parameter])

    for scorer, color in zip(sorted(scoring), ['g', 'k', 'r']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            # ax.fill_between(X_axis, sample_score_mean - sample_score_std,
            #                 sample_score_mean + sample_score_std,
            #                 alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()

def getFeatures(features_list, clf):

    log_info("============== Features Selected ==============")
    features_selected_ind = clf.named_steps['feature_selection'].get_support()
    features_list_wo_poi = features_list[1:]
    features_selected = [features_list_wo_poi[i] for i, selected in enumerate(features_selected_ind) if selected == True]

    for feature_selected in features_selected:
        log_info(feature_selected)

    return [features_selected, features_selected_ind]

def logOutliers(data_dict, features_list):

    df = pd.DataFrame.from_dict(data=data_dict, orient='index')
    df.drop(['email_address', 'poi'], axis=1, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Plot distribution of feature values
    # ax = plt.subplot(111)
    # df.plot(kind='box', subplots=True, layout=(4,5), sharex=False, sharey=False)


    # Print out largest values
    log_info("============== Largest feature values ==============")
    for feature in df.keys():
        log_info("{:<25}: {:<25}=\t{:,.0f}".format(feature, df[feature].idxmax(), df[feature].max(),0))

def best_config(name, model, parameters, features_train, labels_train):

    log_info('Grid search for... ' + name)
    select = SelectKBest(f_classif, k=10)
    pca = decomposition.PCA()
    steps = [('pca', pca),
             ('feature_selection', select),
             ('classifier', model)]
    pipeline = sklearn.pipeline.Pipeline(steps)

    scoring = ['accuracy', 'recall', 'precision', 'f1']
    sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.1, random_state=42)
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scoring, refit='recall',
                      cv=sss, return_train_score=True)
    cv.fit(features_train, labels_train)
    results = cv.cv_results_
    results_df = pd.DataFrame(results)

    # Plot scoring as a function of # of features selected
    # plotGridSearchCV(name, scoring, results, "feature_selection__k")

    log_info('Best hyperparameters: ' + str(cv.best_params_))

    return [name, cv.best_score_, cv.best_estimator_]


# List of candidate family classifiers with parameters for grid
# search [name, classifier object, parameters].
def candidate_families():
    candidates = []
    nb_tuned_parameters = dict(feature_selection__k=range(4,6,1))
    candidates.append((["Naive Bayes", GaussianNB(), nb_tuned_parameters]))


    # svm_tuned_parameters = dict(feature_selection__k=range(4,20,1))
    # candidates.append(["SVM", SVC(), svm_tuned_parameters])

    dt_tuned_parameters = dict(feature_selection__k=range(10, 14, 1),
                               classifier__min_samples_split=range(2, 20, 2),
                               classifier__max_depth=range(2, 20, 2))
    candidates.append(["Decision Tree", tree.DecisionTreeClassifier(), dt_tuned_parameters])

    # rf_tuned_parameters = dict(feature_selection__k=range(10,12,1))
    # candidates.append(["RandomForest",
    #                    RandomForestClassifier(n_jobs=-1),
    #                    rf_tuned_parameters])

    # knn_tuned_parameters = [{"n_neighbors": [1, 3, 5, 10, 20]}]
    # candidates.append(["kNN", KNeighborsClassifier(),
    #                    knn_tuned_parameters])

    return candidates

# Returns the best model from a set of model families given
# training data using cross-validation.
def best_model(classifier_families, features_train, labels_train):

    best_quality = -0.1
    best_classifier = None
    classifiers = []

    log_info("============== Searching for best hyperparameter ==============")
    for name, model, parameters in classifier_families:
        classifiers.append(best_config(name,
                                       model,
                                       parameters,
                                       features_train,
                                       labels_train))

    log_info("============== Searching for best classifier ==============")
    for name, quality, classifier in classifiers:
        log_info('Considering classifier... %s: F1 score %.4f' % (name, quality))
        if (quality > best_quality):
            best_quality = quality
            best_classifier = [name, classifier]

    log_info('Best classifier... ' + best_classifier[0])
    return best_classifier[1]

def make_meshgrid(x, y, n=100):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/n),
                         np.arange(y_min, y_max, (y_max-y_min)/n))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def plotDecisionBoundary(clf, features_test, labels_test):
    features_selected = clf.named_steps['feature_selection'].get_support()
    features_idx = [i for i, selected in enumerate(features_selected) if selected==True]
    features_comb = combinations(features_idx, 2)

    fig, sub = plt.subplots(nCr(len(features_idx), 2)/2 + 1, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    axs = sub.flatten()

    idx_subplot = 0
    for idx_x, idx_y in features_comb:
        X0 = np.asarray([features_test[i][idx_x] for i in np.arange(len(features_test))])
        X1 = np.asarray([features_test[i][idx_y] for i in np.arange(len(features_test))])
        xx, yy = make_meshgrid(X0, X1)

        axs[idx_subplot].scatter(X0, X1, c=labels_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

        #plot_contours(axs[idx_subplot], clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        idx_subplot += 1


def dataCleaning(data_dict):
    df = pd.DataFrame(data_dict)
    df = df.transpose()

    # Convert NaN to 0 for financial fields
    fields = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances',
               'other', 'expenses', 'director_fees', 'total_payments', 'exercised_stock_options', 'restricted_stock',
              'restricted_stock_deferred', 'total_stock_value']
    df[fields] = df[fields].apply(pd.to_numeric, args=('coerce',))
    df[fields] = df[fields].fillna(value=0)

    # Remove non-persons
    df = df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'])
    
    return(df.to_dict(orient='index'))


init_logging()


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict = dataCleaning(data_dict)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = data_dict.values()[0].keys()
features_list.remove('poi')
features_list.remove('email_address')
features_list.insert(0, 'poi')

### Task 2: Remove outliers
logOutliers(data_dict, features_list)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

models = candidate_families()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
#pred = cv.predict(features_test)
#report = sklearn.metrics.classification_report(labels_test, pred)

best_clf = best_model(models, features_train, labels_train)
getFeatures(features_list, best_clf)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(best_clf, my_dataset, features_list)















































