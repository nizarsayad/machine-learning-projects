import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.combine import SMOTETomek

class ModelTrainer:
    """
    A class to automate the machine learning model training and evaluation process. 

    Attributes
    ----------
    df : DataFrame
        The input data for model training.
    target_variable : str
        The target variable to be predicted.
    models : list
        The machine learning models to be trained.
    metrics : list
        The metrics used to evaluate model performance.
    experiment : str, optional
        Name of the experiment.
    feature_selection : bool, optional
        If True, performs feature selection.
    feature_selector : callable, optional
        The method used for feature selection.
    resample : bool, optional
        If True, performs data resampling.
    sampler : callable, optional
        The method used for data resampling.
    drop_features : list, optional
        A list of features to drop from the data.
    sampling_strategy : float, optional
        The sampling strategy to be used in the resampling method.
    test_size : float, optional
        The test set size as a percentage of the total data.
    random_state : int, optional
        The seed for the random number generator.
    results : DataFrame
        A DataFrame to store the evaluation results.
    trained_models : list
        A list to store the trained models.
    gs_params : dict
        A dictionary to store the parameters of the grid search.
    gs_results : DataFrame
        A DataFrame to store the results of the grid search.
    best_model : list
        A list to store the best model found during grid search.

    Methods
    -------
    split_features_target():
        Splits the data into features and target variable.
    separate_features():
        Separates the features into numeric and categorical features.
    split_data():
        Splits the data into training and test sets.
    preprocess():
        Preprocesses the data.
    train_evaluate():
        Trains the models and evaluates their performance.
    grid_search(model, param_grid):
        Performs grid search to find the optimal hyperparameters of the model.
    process():
        Performs the entire data processing, model training, and evaluation process.
    """
    def __init__(self, df, target_variable, models, metrics, experiment='Baseline', feature_selection=False, feature_selector=SelectKBest, resample=False, sampler=SMOTETomek, drop_features=None, sampling_strategy=0.75, test_size=0.2, random_state=42):
        self.df = df.drop(columns=drop_features, errors='ignore') if drop_features else df
        self.target_variable = target_variable
        self.models = models if isinstance(models, list) else [models]
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.experiment = experiment
        self.feature_selection = feature_selection
        self.feature_selector = feature_selector(f_classif, k=5)
        self.resample = resample
        self.sampler = sampler(sampling_strategy=sampling_strategy)
        self.test_size = test_size
        self.random_state = random_state
        self.results = pd.DataFrame(columns=['model', 'experiment'] + [f'{metric.__name__}_train' for metric in self.metrics] + [f'{metric.__name__}_test' for metric in self.metrics])
        self.trained_models = []
        self.gs_params = {}
        self.gs_results = pd.DataFrame(columns=['model', 'experiment'] + [f'{metric.__name__}_train' for metric in self.metrics] + [f'{metric.__name__}_test' for metric in self.metrics])
        self.best_model = []
    
    def split_features_target(self):
        self.X = self.df.drop(self.target_variable, axis=1)
        self.y = self.df[self.target_variable]

    def separate_features(self):
        self.numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.X.select_dtypes(exclude=[np.number]).columns.tolist()

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y)

    def preprocess(self):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first')
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)])
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        if self.resample:
            self.X_train, self.y_train = self.sampler.fit_resample(self.X_train, self.y_train)
        if self.feature_selection:
            self.X_train = self.feature_selector.fit_transform(self.X_train, self.y_train)
            self.X_test = self.feature_selector.transform(self.X_test)


    def train_evaluate(self):
        for model in self.models:
            print(f'Training {model.__class__.__name__}...')
            model_name = model.__class__.__name__
            model.fit(self.X_train, self.y_train)
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            results_row = {'model': model_name, 'experiment': self.experiment}
            for metric in self.metrics:
                results_row[f'{metric.__name__}_train'] = metric(self.y_train, y_train_pred)
                results_row[f'{metric.__name__}_test'] = metric(self.y_test, y_test_pred)
            self.results.loc[len(self.results)] = results_row
            self.trained_models.append(model)
            print(f'Training completed...')
        self.results.sort_values('f1_score_test', ascending=False, inplace=True)

    def grid_search(self,model, param_grid):
        # Assume the best model is the first one in the sorted results dataframe
        #best_model_name = self.results.iloc[0]['model']
        # Find the corresponding model object in the trained_models list
        #for model in self.trained_models:
        #    if model.__class__.__name__ == best_model_name:
        #        best_model = model
        self.best_model = model
        best_model_name = model.__class__.__name__
        # Apply grid search on the best model
        print(f'Applying Grid Search on {best_model_name}...')
        grid_search = GridSearchCV(self.best_model, param_grid, cv=5, scoring='f1', n_jobs=6, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        print(f'Best parameters: {grid_search.best_params_}')
        self.gs_params[best_model_name] = grid_search.best_params_
        # Update trained model and evaluate again
        self.best_model = grid_search.best_estimator_
        y_train_pred = self.best_model.predict(self.X_train)
        y_test_pred = self.best_model.predict(self.X_test)
        results_row = {'model': best_model_name, 'experiment': self.experiment + ' + Grid Search'}
        for metric in self.metrics:
            results_row[f'{metric.__name__}_train'] = metric(self.y_train, y_train_pred)
            results_row[f'{metric.__name__}_test'] = metric(self.y_test, y_test_pred)
        self.gs_results.loc[len(self.gs_results)] = results_row
        print(f'Updated results with Grid Search on {best_model_name}...')

    def process(self):
        print("Processing data...")
        print("Splitting features and target...")
        self.split_features_target()
        print("Separating features into numeric and categorical...")
        self.separate_features()
        print("Splitting data into train and test sets...")
        self.split_data()
        print("Building preprocessor...")
        self.preprocess()
        
