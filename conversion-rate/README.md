# Project 03 - 2: Conversion rate

**Certificate - Machine Learning Engineer**

**Bloc 3 - Machine Learning**

## Introduction

In this project, we will participate to a machine learning competition like the ones that are organized by https://www.kaggle.com/. We will be able to work with jupyter notebooks as usual, but in the end we'll have to submit our model's predictions  so our model's performances will be evaluated in an independent way. The scores achieved by the different teams will be stored into a leaderboard.

The data scientists who created the newsletter would like to understand better the behaviour of the users visiting their website. They would like to know if it's possible to build a model that predicts if a given user will subscribe to the newsletter, by using just a few information about the user. They would like to analyze the parameters of the model to highlight features that are important to explain the behaviour of the users, and maybe discover a new lever for action to improve the newsletter's conversion rate.

They designed this competition aiming at building a model that allows to predict the conversions (i.e. when a user will subscribe to the newsletter). To do so, they open-sourced a dataset containing some data about the traffic on their website. To assess the rankings of the different competing teams, they decided to use the f1-score.

This project is part of the professional certification Machine Learning Engineer (Concepteur Développeur en Science des données) and represents bloc 3 of the certification titled: "Machine Learning".

## Project Structure

The project consists of several components:

- `conversion-rate`: This directory contains the notebook, the data files, and other related files.
    
    - `conversion-rate.ipynb`: This is the main Jupyter notebook that contains all the data analysis and machine learning models.
    
    - `conversion_data_train.csv`: This is the dataset used for training.

    - `conversion_data_test.csv`: This is the dataset used for evaluation.

    - `conversion_data_test_predictions_NIZAR-best_model.csv`: These are the predictions of the best model.

    - `README.md`: This file provides an overview of the project.

## Scope of the Project 

- In machine learning challenges, the dataset is always separated into to files :
    - data_train.csv contains labelled data, which means there are both X (explanatory variables) and Y (the target to be predicted). We will use this file to train our model as usual : make the train/test split, preprocessings, assess performances, try different models, fine-tune hyperparameters etc...
    - data_test.csv contains "new" examples that have not be used to train the model, in the same format as in data_train.csv but it is unlabeled, which means the target Y has been removed from the file. Once we've trained a model, we will use data_test.csv to make some predictions that we will send to the organizing team. They will then be able to assess the performances of our model in an independent way, by preventing cheating 🤸
- Our model's predictions will be compared to the true labels and released in a leaderboard where the scores of all the teams are stored.
- All the participants are informed about the metric that will be used to assess the scores. we have to make sure we're using the same metric to evaluate our train/test performances.