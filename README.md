# In Zeus We Trust!

## Introduction

In this project, I'm tried to predict the outcome of a Defense of the Ancients (DOTA) match based characters chosen heroes. DOTA is an online multiplayer video game, each game consists of 10 players divided between two teams (Sentinal and Scourge), to eliminate the opposing's team base. In the game, there are about 100 heroes to choose from, each with its unique abilities and each hero can only be picked by one player in a game.

DOTA is a highly competitive and complicated game with a big esports scene. Each character can play different roles in different games and the outcome of the match is highly dependet on the players controlling the characters. Since no other info was given about the game or players, I was force to assume all of the players are of similar skill level.

## Technical Description

The dataset was released by [HackerRank](https://www.hackerrank.com/challenges/dota2prediction/problem/ "HackerRank") as a challenge.
The dataset contains observations from 15000 matches in text file. Each row consists of 10 characters names (first 5 Sentinal, then 5 Scourge) and the outcome of the game (1 for Sentinal victory, 2 for Scourge) separated by commas.

The majority of the code was written in Jupyter Notebooks using Python. Additionally, I used [VSCode](https://code.visualstudio.com/ "VSCode") to assist in creating the helper functions module and Readme. The project is publicly accessible can be found on my [GitHub](https://github.com/DaggerForce/Dota_Victory_Classification "GitHub").

For data manipulation and EDA, I utilized [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html/ "Pandas") and [Numpy](https://www.numpy.org/ "Numpy"). For data analysis, I utilized [Scipy](https://www.numpy.org/ "Scipy"), [Sklearn](https://scikit-learn.org/stable/ "Sklearn"), and [XGBoost](https://xgboost.readthedocs.io/en/latest// "XGboost"). For the visualizations, I utilized [Matplotlib](https://matplotlib.org/ "Matplotlib") and [Seaborn](https://seaborn.pydata.org/introduction.html/ "Seaborn")

## Table of Contents

* Exploring the Dataset
  * [Preparing Data](#Preproccesing)
  * [Feature Engineering](#feature-engineering)
* Models
  * [KNN](#KNN)
  * [Decision Tree](#decision-Tree)
  * [Random Forest](#random-forest)
  * [AdaBoost](#adaboost)
  * [XGBoost](#xgboost)
  * [Logistic Regression](#logistic-regression)
  * [Support Vector Machine](#svm)
* [Recommendation](#take-home-message)

### Preprocessing

I began by looking for fragmented observations in the dataset and then started to manipulate it for modeling purposes. As previously mentioned, each row consisted of both the winning team, the losing team and the outcome. Because each observation contained both a winning team and a losing team, I wasn't worried about class imbalance. However, I was curious if either team won significantly more times than the other. 
<img src=Images/classes_hist.png alt="Histogram featuring how many times each team won" width="350"/>

So I  started by splitting the original data-frame into 4 smaller ones, the winning teams from team Sentinal, the winning teams from team Scourge, and the losing teams. I then attributed the outcome '1' for winning, '0' for losing and continued by creating a dummy variable for each character.

### Feature Engineering

Having 

1. Population Size - We divided our countries into three catagories; Small, Medium, and Large.
2. Lifestyle - We created an interaction between alcohol consumption and BMI
3. Economy - The interaction between the population and the GDP.
4. Death ratio - The ratio between adult and infant mortality.

## Modeling




We proceeded by searching for multicollinearity between the selected predictors by creating a correlation matrix. We defined multicollinearity cut-off at 0.8 and omitted alcohol consumption and GDP from the initial model.

![HeatMap](Images/heatmap.png)

We then proceeded to remove possible outliers by looking at their scatter plots and removed the observations we deemed as unusual. After removing the outliers, 1635 observations remained in the dataset.

<img src=Images/paired_before_lifestyle.png alt="Scatter Before removing outliers" width="350"/>


<img src=Images/paired_afte_lifestyle.png alt="Scatter Before removing outliers" width="350"/>

## Results

The first model we ran to predict life expectancy used the features; BMI, HIV, thinness 1-19, GDP, mortality ratio, lifestyle, education, infant mortality rate, economy, and population size. With R squared equal to 0.804, our initial model explains 80%~ of variation in life expectancy.

<img src=Images/init_summary.png alt="Initial model summary" width="450"/>

We ran the model again after scaling the data, and also removing predictors that were deemed insignificant (P-value > 0.05).

<img src=Images/scaled_model_summary.png alt="Scaled model summary" width="450"/>

To test the model, we looked at the distribution of residuals for homoscedasticity. However, the residuals show a relationship. The heteroscedasticity is likely to have caused due to one or more of the predictors' distribution is skewed.

<img src=Images/scaled_residuals.png alt="Residuals scatter plot and historgram" width="450"/>

We conducted a train, test split test using 80% of our data to predict the other 20%. The model's mean absolute error is 3.022

<img src=Images/model_final.png alt="Train test split model" width="450"/>

### Train Test Split
Additionally, we tested the model with all the features we previously excluded (BMI, alcohol, GDP, and population size). Expectedly, the model mean absolute error is slightly smaller (2.995). However, the deviation is still in years meaning the model has to be refined.

<img src=Images/model_all.png alt="Train test split model" width="450"/>

## Take Home Message

Our suggestions for countries looking to increase their life expectancy is to focus their resources mainly towards increasing HIV awareness. Additionally, we recommend increasing promoting education and to invest more in hospital maternity wards.
