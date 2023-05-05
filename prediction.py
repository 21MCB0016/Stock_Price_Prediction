from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


class Predictor:
    def __init__(self, db_uri, db_name, db_collection, tickers):
        self.db_uri = db_uri
        self.db_name = db_name
        self.db_collection = db_collection
        self.load_data()
        self.tickers = tickers
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.pred()

    def load_data(self):
        # load data from MongoDB into a pandas dataframe
        client = MongoClient(self.db_uri)
        db = client[self.db_name]
        collection = db[self.db_collection]
        cursor = collection.find()
        df = pd.DataFrame(list(cursor))
        df = df.drop(['_id'], axis=1)
        df = df.set_index('last_updated')
        return df

    ''' def clean_data(self):
        # remove rows with null values
        # data = self.data.dropna(inplace=True)
        df = self.load_data()
        df = df.fillna(method='ffill', inplace=True)
        return df'''

    def split_data(self):
        """
        Split the data into training and testing sets
        """
        df = self.load_data()
        # Split the data into X and y
        X = df[['close', 'low', 'high', 'open', 'volume']]
        X = X.fillna(0)
        y = df['adjclose']
        y = y.fillna(0)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test

    def pred(self):
        df = self.load_data()
        # Training Model
        StockPredictor = LinearRegression()

        # Define hyperparameter grid
        param_grid = {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'n_jobs': [None, -1, 1, 2, 4],
            'positive': [True, False]
        }

        # Perform GridSearchCV
        reg_gs = GridSearchCV(StockPredictor, param_grid=param_grid, cv=5, error_score='raise')
        reg_gs.fit(self.X_train, self.y_train)
        # Fit the linear regression model
        model = sm.OLS(self.y_train, self.X_train).fit()

        # Calculate the AIC
        aic = model.aic

        # Print the best hyperparameters
        print("Best hyperparameters:", reg_gs.best_params_)
        print('Akaike information criterion value: ',aic)
        fig, ax = plt.subplots(figsize=(10, 6))
        for bank in self.tickers:
            # Train the model with the best hyperparameters
            StockPredictor = LinearRegression(**reg_gs.best_params_)
            StockPredictor.fit(self.X_train, self.y_train)

            # Predict the prices for test data
            y_pred = StockPredictor.predict(self.X_test)
            date = self.X_test.index

            # Create a dataframe with the predicted and actual values
            results = pd.DataFrame({'last_updated': date, 'Actual': self.y_test, 'Predicted': y_pred})
            # Create a line plot with actual values in blue and predicted values in orange
            sns.lineplot(x='last_updated', y='Actual', data=results, color='blue')
            sns.lineplot(x='last_updated', y='Predicted', data=results, color='orange')

            # Set the plot title and axis labels
            plt.title(f'{bank} Linear Regression Results')
            plt.xlabel('Date')
            plt.ylabel('Price')

            # Add a legend to distinguish between actual and predicted values
            plt.legend(['Actual', 'Predicted'])

            # Show the plot
            plt.show()
