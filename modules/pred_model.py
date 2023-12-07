import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class Model:
    '''
    This class contains the Linear Regression, Random Forest, and XGBoost models.

    Attributes:
        lr_model (LinearRegression): Linear Regression model.
        rf_model (RandomForestRegressor): Random Forest model.
        xgb_model (XGBRegressor): XGBoost model.

    Examples:

        >>> from pred_model import Model
        >>> model = Model()
        >>> lr_model, rf_model, xgb_model = model.train(df_x_train_norm, df_y_train_norm)
        >>> y_pred_lr, y_pred_rf, y_pred_xgb = model.predict(df_x_test_norm)
        >>> mse = model.calculate_mse(df_y_test_norm, (y_pred_lr + y_pred_rf + y_pred_xgb) / 3)
    '''
    def __init__(self):
        """
        Initialize the Model class with Linear Regression, Random Forest, and XGBoost models.

        """
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb_model = XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            n_estimators=100,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8
        )

    def train(self, df_x_train_norm, df_y_train_norm):
        """
        Train Linear Regression, Random Forest, and XGBoost models.

        Args:
            df_x_train_norm (pd.DataFrame): Normalized training features.
            df_y_train_norm (pd.DataFrame): Normalized training target variable.

        Returns:
            Tuple[LinearRegression, RandomForestRegressor, XGBRegressor]: Trained models.
        """
        # Training Linear Regression model
        self.lr_model.fit(df_x_train_norm, df_y_train_norm.ravel())

        # Training Random Forest model
        self.rf_model.fit(df_x_train_norm, df_y_train_norm.ravel())

        # Training XGBoost model
        self.xgb_model.fit(df_x_train_norm, df_y_train_norm.ravel())

        return self.lr_model, self.rf_model, self.xgb_model

    def predict(self, df_x_test_norm):
        """
        Make predictions using the trained models.

        Args:
            df_x_test_norm (pd.DataFrame): Normalized test features.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predictions for Linear Regression, Random Forest, and XGBoost models.
        """
        # Predicting with Linear Regression model
        y_pred_lr = self.lr_model.predict(df_x_test_norm)

        # Predicting with Random Forest model
        y_pred_rf = self.rf_model.predict(df_x_test_norm)

        # Predicting with XGBoost model
        y_pred_xgb = self.xgb_model.predict(df_x_test_norm)

        return y_pred_lr, y_pred_rf, y_pred_xgb

    def calculate_mse(self, df_y_test_norm, y_pred_lr, y_pred_rf, y_pred_xgb):
        """
        Calculate the Mean Squared Error.

        Args:
            df_y_test_norm (pd.DataFrame): Normalized test target variable.
            y_pred_lr (np.ndarray): Predictions for Linear Regression model.
            y_pred_rf (np.ndarray): Predictions for Random Forest model.
            y_pred_xgb (np.ndarray): Predictions for XGBoost model.

        Returns:
            float: Mean Squared Error.
        """
        # Calculating mean squared error
        mse = {}
        mse['lr'] = mean_squared_error(df_y_test_norm, y_pred_lr)
        mse['rf'] = mean_squared_error(df_y_test_norm, y_pred_rf)
        mse['xgb'] = mean_squared_error(df_y_test_norm, y_pred_xgb)
        return mse


class Visualizer:
    '''
    This class contains the visualizations for the Mean Squared Error comparison, feature importance for Random Forest,
    feature importance for XGBoost, and feature importance comparison between Random Forest and XGBoost.

    Attributes:
        ft_ls (list): List of feature names.

    Examples:

        >>> from pred_model import Visualizer
        >>> ft_ls = ['orbital period', 'orbital semi-major axis', 'orbital eccentricity',
                        'radial velocity', 'transit depth', 'system rotational vel.']
        >>> visualizer = Visualizer(ft_ls)
        >>> visualizer.visualize_mse_comparison(mse_lr, mse_rf, mse_xgb)
        >>> visualizer.visualize_feature_importance_rf(feature_names_rf)
        >>> visualizer.visualize_feature_importance_xgb(feature_names_xgb)
        >>> visualizer.visualize_model_comparison(feature_names_rf, feature_names_xgb)
    '''

    def __init__(self, ft_ls, theme = 'dark'):
        """
        Initialize the Visualizer class with a list of feature names.

        Args:
            ft_ls (list): List of feature names.
        """
        self.ft_ls = ft_ls
        if theme == 'dark':
            plt.style.use('dark_background')

    def visualize_mse_comparison(self, mse_lr, mse_rf, mse_xgb):
        """
        Visualize Mean Squared Error comparison among different models.

        Args:
            mse_lr (float): Mean Squared Error for Linear Regression.
            mse_rf (float): Mean Squared Error for Random Forest.
            mse_xgb (float): Mean Squared Error for XGBoost.

        Returns:
            plt: Plot of Mean Squared Error comparison among different models.
        """
        models = ['Linear Regression', 'Random Forest', 'XGBoost']
        x_values = np.arange(len(models))

        plt.bar(x_values - 0.2, [mse_lr, 0, 0], width=0.2, color='blue', label='Linear Regression')
        plt.bar(x_values, [0, mse_rf, 0], width=0.2, color='orange', label='Random Forest')
        plt.bar(x_values + 0.2, [0, 0, mse_xgb], width=0.2, color='green', label='XGBoost')
        plt.xticks(x_values, models)
        plt.ylabel('Mean Squared Error (MSE)')
        plt.yscale('log')
        plt.title("Comparison of Mean Squared Error (MSE) among all 3 models")
        plt.tight_layout()
        return plt

    def visualize_feature_importance_rf(self, feature_names_rf):
        """
        Visualize feature importance for Random Forest.

        Args:
            feature_names_rf (np.ndarray): Feature importances for Random Forest.

        Returns:
            plt: Plot of feature importance for Random Forest.
        """
        plt.bar(range(len(feature_names_rf)), feature_names_rf)
        plt.xticks(range(len(feature_names_rf)), self.ft_ls, rotation=-90)
        plt.ylabel('Feature Importance')
        plt.title('Random Forest Regressor - Feature Importance')
        plt.tight_layout()
        return plt

    def visualize_feature_importance_xgb(self, feature_names_xgb):
        """
        Visualize feature importance for XGBoost.

        Args:
            feature_names_xgb (np.ndarray): Feature importances for XGBoost.

        Returns:
            plt: Plot of feature importance for XGBoost.
        """
        plt.bar(range(len(feature_names_xgb)), feature_names_xgb)
        plt.xticks(range(len(feature_names_xgb)), self.ft_ls, rotation=-90)
        plt.ylabel('Feature Importance')
        plt.title('XG Boost - Feature Importance')
        plt.tight_layout()
        return plt

    def visualize_model_comparison(self, feature_names_rf, feature_names_xgb):
        """
        Visualize feature importance comparison between Random Forest and XGBoost.

        Args:
            feature_names_rf (np.ndarray): Feature importances for Random Forest.
            feature_names_xgb (np.ndarray): Feature importances for XGBoost.

        Returns:
            plt: Plot of feature importance comparison between Random Forest and XGBoost.
        """
        n = len(self.ft_ls)
        r = np.arange(n)
        width = 0.25

        bar1 = plt.bar(r, feature_names_rf, color='w', width=width)
        bar2 = plt.bar(r + width, feature_names_xgb, color='r', width=width)
        plt.ylabel('Feature Importance')
        plt.title('Random Forest vs XGBoost - Model Comparison')
        plt.xticks(r + width, self.ft_ls, rotation=-90)
        plt.legend((bar1, bar2), ('Random Forest', 'XGBoost'))
        plt.tight_layout()
        return plt

if __name__ == '__main__':
    # Loading data
    df = pd.read_csv("data/NASA_planetary_data.csv", skiprows = 168)

    # Dropping rows with null values in the 'pl_orbper' column
    df.dropna(subset=['pl_dens','pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'st_radv', 'pl_trandep', 'st_vsin'], axis=0, how='any', inplace=True)

    # Selecting independent features and target variable
    df_x = df[['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'st_radv', 'pl_trandep', 'st_vsin']]
    df_y = df['pl_dens']

    # Splitting the data into training and testing sets
    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.1, random_state=0)

    # Handling null values and cleaning data
    # ...

    # Scaling the features
    scaler = MinMaxScaler()
    df_x_train_norm = scaler.fit_transform(df_x_train)
    df_x_test_norm = scaler.fit_transform(df_x_test)

    df_y_train_norm = scaler.fit_transform(df_y_train.values.reshape(-1, 1))
    df_y_test_norm = scaler.fit_transform(df_y_test.values.reshape(-1, 1))

    # Instantiating the Model class and training the models
    model = Model()
    lr_model, rf_model, xgb_model = model.train(df_x_train_norm, df_y_train_norm)

    # Making predictions
    y_pred_lr, y_pred_rf, y_pred_xgb = model.predict(df_x_test_norm)

    # Calculating Mean Squared Error
    mse = model.calculate_mse(df_y_test_norm, y_pred_lr , y_pred_rf, y_pred_xgb)

    print(mse['lr'], mse['rf'], mse['xgb'])

    # Instantiating the Visualizer class
    ft_ls = ['orbital period', 'orbital semi-major axis', 'orbital eccentricity',
            'radial velocity', 'transit depth', 'system rotational vel.']
    visualizer = Visualizer(ft_ls)

    # Example data for illustration purposes
    feature_names_rf = rf_model.feature_importances_  # getting feature importance according to random forest
    feature_names_xgb = xgb_model.feature_importances_  # feature importance according to XGBoost

    # Visualize comparisons
    visualizer.visualize_mse_comparison(mse['lr'], mse['rf'], mse['xgb'])
    visualizer.visualize_feature_importance_rf(feature_names_rf)
    visualizer.visualize_feature_importance_xgb(feature_names_xgb)
    visualizer.visualize_model_comparison(feature_names_rf, feature_names_xgb)