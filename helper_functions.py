from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score, f1_score
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

plt.style.use('ggplot')


# FUNCTIONS USED IN THE EDA PROCCES


def populate_df(df):
    """ This function turns the characters dataframe into dummy varaibles dataframe"""
    uniques = pd.unique(df.values.ravel('K'))
    zeros = np.zeros(len(uniques))
    
    # main df protection
    df = df.copy(deep=True)
    
    all_dummies = []
    for row in df.itertuples():
        i = 1
        uniques_dic = dict(zip(uniques, zeros))
        while i < 6:
            uniques_dic[row[i]] = 1
            i += 1

        all_dummies.append(uniques_dic)
    return pd.DataFrame(all_dummies, columns=uniques)


def main_attributes(attribute):
    '''This function scrapes all of the heroes by their main attribute
       off of dota2 wikisite'''
    # Make a get request to retrieve the page
    html = requests.get(f'https://dota2.gamepedia.com/{attribute}')

    # Pass the page contents to beautiful soup for parsing
    soup = BeautifulSoup(html.content, 'html.parser')

    # search for the characters by attribute
    char_raw = soup.findAll('td', {'style': "white-space:nowrap;"})
    char_list = np.array(
        [item.find('a').attrs['href'].replace('/', '') for item in char_raw])
    return char_list


def heroes_roles(category):
    # Make a get request to retrieve the page
    html = requests.get(f'https://dota2.gamepedia.com/Category:{category}')

    # Pass the page contents to beautiful soup for parsing
    soup = BeautifulSoup(html.content, 'html.parser')

    # search for all the heroes on the list
    page = soup.findAll(class_="mw-category-group")
    try:
        heroes = np.array([hero.find('a').get_text() for hero in page])
    except:
        "Search field was too broad and more than heroes appeared on the list"
    return heroes


def create_hist(df, column, save=None):
    ''' This function creates an histogram using a dataframe and a column name '''
    plt.figure(figsize=(8, 5))

    # Remove the plot frame lines.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(b=None)

    # set labels
    plt.title(f"{column} Heroes in Team Histogram")
    plt.xlabel(f"{column} Per Team", fontsize=16)
    plt.ylabel("Matches Won", fontsize=16)

    # Plot the histogram
    plt.hist(df[column], bins=5, alpha=0.7, density=True)
    plt.show()
    if save:
        plt.savefig(f'{column}_Histogram.png')
    pass


def team_composition(df, attributes):
    for attribute in attributes:
        sns.catplot(kind='count', data=df, x=attribute)
        plt.title('Composition')
        plt.xlabel(f"{attribute} Heroes")
        plt.ylabel(f"Game Count")
        plt.grid(b=None)
        try:
            plt.savefig(f"team_comp_{attribute}.png")
        except:
            plt.savefig(f"team_comp_bad_name.png")
        pass

# FUNCTIONS USED IN THE MODELING PROCCES


def print_metrics(labels, predictions, print_score=None):
    ''' This function receives model predictions along with the actual labels
        and returns the precision score, recall, accuracy and F1'''

    recall = round(recall_score(labels, predictions)*100, 2)
    acc = round(accuracy_score(labels, predictions)*100, 2)
    f1 =
    if print_score:
        print(f"Recall Score: {recall}")
        print(f"Accuracy Score: {acc}")

    return 


def plot_confusion_matrix(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    plt.matshow(matrix,  cmap=plt.cm.RdYlBu, aspect=1.75, alpha=0.5)

    # Add title and Axis Labels
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Add appropriate Axis Scales
    class_names = ['Lose', 'Win']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.grid(b=None)

    # Add Labels to Each Cell
    thresh = matrix.max() / 2.  # Used for text coloring below

    # iterate through the confusion matrix and append the labels
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j], horizontalalignment="center",
                 color="black")

    # Add a Side Bar Legend Showing Colors
    plt.colorbar()
    return


def plot_feature_importance(model, x_train, n=10):
    """ This function recievies a model and plots the 'n' most important features"""
    # extract and sort the feature importance
    features = model.feature_importances_
    feature_names = x_train.columns.values

    # combine the features importance and column names into a matrix and sort them
    feature_matrix = np.array([features, feature_names])
    feature_matrix = feature_matrix.transpose()
    feature_matrix = feature_matrix[feature_matrix[:,0].argsort()]

    # divide the column names and feature importance
    sorted_feat = feature_matrix[:, 0]
    sorted_columns = feature_matrix[:, 1]

    # plot the features
    plt.figure(figsize=(14, 10))
    try:
        plt.barh(sorted_columns[-n:], sorted_feat[-n:], align='center')

    except:
        # if n features is greater than the amount that actually exists
        n = len(sorted_feat)
        plt.barh(sorted_columns[-n:], sorted_feat[-n:], align='center')

    plt.yticks(sorted_columns[-n:], sorted_columns[-n:])
    plt.title('Feature Importances', fontsize=18)
    plt.xlabel('Feature Importance', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    return


def find_optimal_depth(x_train, x_test, y_train, y_test):
    # declare variables
    max_depths = np.linspace(1, 15, 15, endpoint=True)
    train_results = []
    test_results = []
    # iterate over the different depths
    for depth in max_depths:
        trees = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        trees.fit(x_train, y_train)

        # Add auc score to train list
        train_pred = trees.predict(x_train)
        fpr, tpr, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(fpr, tpr)
        train_results.append(roc_auc)

        # Add auc score to test list
        test_pred = trees.predict(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, test_pred)
        roc_auc = auc(fpr, tpr)
        test_results.append(roc_auc)

    plt.figure(figsize=(8, 5))
    plt.plot(max_depths, train_results, 'b', label='Train AUC')
    plt.plot(max_depths, test_results, 'r', label='Test AUC')
    plt.ylabel('AUC score', fontsize=16)
    plt.xlabel('Tree depth', fontsize=16)
    plt.legend()
    plt.show()
    pass


def plot_roc_curve(model, x_test, y_test):
    ''' This function accepts the model, testing set, testing labels, and outputs
        a Receiver Operating Characteristic curve plot'''
    # extract the target probability
    predict_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, predict_proba)
    
    # plot the roc curve
    plt.figure(figsize=(8,5))
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC Curve')
   
    # plot a line through the origin of axis
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    
    # add graph labels
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve', fontsize=118)
    plt.legend(loc="lower right")
    return round(roc_auc_score(y_test, predict_proba), 2)