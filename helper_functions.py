from matplotlib import rcParams
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

plt.style.use('ggplot')
rcParams.update({'figure.autolayout': True})

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
        [item.find('a').attrs['title'].replace('/', '') for item in char_raw])
    return char_list


def hero_roles(category):
    ''' This function scrapes the DOTA2 hero list off of DOTA2 wiki
        given a catagory the user provides '''
    # Make a get request to retrieve the page
    html = requests.get(f'https://dota2.gamepedia.com/Category:{category}')

    # Pass the page contents to beautiful soup for parsing
    soup = BeautifulSoup(html.content, 'html.parser')

    # search for all the heroes on the list
    page = soup.findAll(class_="mw-category-group")
    character_list = []
    for item in page:
        try:
            more_heroes = item.findAll('a')
            for hero in more_heroes:
                character_list.append(hero.get_text())
        except:
            heroes = item.find('a').get_text()
            character_list.append(heroes)
    return np.array(character_list)


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
    ''' This function returns a plot that counts how many characters 
        with the the same attribute the team consists of '''
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
    f1 = round(f1_score(labels, predictions)*100, 2)
    if print_score:
        print(f"Recall: {recall}")
        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")

    return


def multiple_knn(df, labels, ks=[5]):

    x_train, x_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.2)
    best_acc = 0
    best_k = 0
    scores = []

    for k in ks:
        knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=k)
        knn.fit(x_train, y_train.values.ravel())
        test_predict = knn.predict(x_test)
        acc = accuracy_score(y_test, test_predict)
        scores.append(acc)

        # save the the highest accuracy and the how many neighbors
        if best_acc < acc:
            best_acc = acc
            best_k = k
    return best_acc, best_k


def plot_confusion_matrix(y_test, y_pred):
    ''' This function receives model predictions and the 
    actual labels and returns a formatted confusion matrix '''
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = 10, 10
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.spines.top'] = True

    matrix = confusion_matrix(y_test, y_pred)
    plt.matshow(matrix,  cmap=plt.cm.Blues, aspect=1.2, alpha=0.5)
    
    # fixesd the issue where the newest matplotlib version
    # acts strangely
    plt.ylim([-0.5,1.5])
    
    # Add title and Axis Labels
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    
    # Add appropriate Axis Scales
    class_names = ['Lose', 'Win']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add Labels to Each Cell
    thresh = matrix.max() / 2.  # Used for text coloring below
    
    # iterate through the confusion matrix and append the labels
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j], horizontalalignment="center",
                 color="black")

    # Add a Side Bar Legend Showing Colors
    plt.colorbar()
    plt.show()
    return


def plot_feature_importance(model, x_train, n=12):
    """ This function recievies a model and plots the 'n' most important features"""
    # extract and sort the feature importance
    features = model.feature_importances_
    feature_names = x_train.columns.values

    # combine the features importance and column names into a matrix and sort them
    feature_matrix = np.array([features, feature_names])
    feature_matrix = feature_matrix.transpose()
    feature_matrix = feature_matrix[feature_matrix[:, 0].argsort()]

    # divide the column names and feature importance
    sorted_feat = feature_matrix[:, 0]
    sorted_columns = feature_matrix[:, 1]

    # plot the features
    plt.figure(figsize=(14, 10))
    if n > len(sorted_feat):
        plt.barh(sorted_columns, sorted_feat, align='center')
    else:
        plt.barh(sorted_columns[-n:], sorted_feat[-n:], align='center')

    # add label and titles
    plt.yticks(sorted_columns[-n:], sorted_columns[-n:])
    plt.title('Feature Importances', fontsize=18)
    plt.xlabel('Feature Importance', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    return


def find_optimal_depth(x_train, x_test, y_train, y_test):
    """
    find_optimal_depth(x_train, x_test, y_train, y_test)
    Params:
        x_train: list. Training observations
        x_test: list. Testing observations
        y_train: list. Training labels 
        y_test: list. Testing labels
    Returns:
        Returns a plot that shows the training and test score AUC at different tree depths between 1-15
    """
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
    return
