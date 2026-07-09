import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(101)
tf.random.set_seed(101)

class MyAnn:
    """A custom Artificial Neural Network (ANN) class for regression tasks.

    This class implements a neural network using TensorFlow/Keras for regression problems,
    with support for feature engineering, model training, evaluation, and visualization.
    It includes methods for fitting the model, making predictions, saving/loading the model,
    and generating plots for model evaluation.

    Attributes:
        data_set (pd.DataFrame): The input dataset.
        target (str): The target column name for regression.
        mode (str): Mode for early stopping ('auto', 'min', 'max').
        verbose (int): Verbosity level for training (0, 1, or 2).
        hidden_layer_sizes (tuple): Architecture of hidden layers (e.g., (128, 64, -0.25, 32)).
        activation (str): Activation function for hidden layers.
        loss (str): Loss function for training.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        monitor (str): Metric to monitor for early stopping.
        patience (int): Patience for early stopping.
        optimizer (str): Optimizer for training.
        classes_ (list): Class labels (empty for regression tasks).
        loss_ (float): Last training loss after fitting.
        best_loss (float): Minimum training loss during fitting.
        features_ (list): List of feature column names after preprocessing.
        n_layers (int): Number of layers in the model.
        n_outputs_ (int): Number of outputs (1 for regression).
        out_activation_ (str): Activation function of the output layer.
    """
    def __init__(self,
                 data_set: str,
                 target: str,
                 mode: str,
                 verbose: int,
                 hidden_layer_sizes: tuple = (100,),
                 activation: str = 'relu',
                 loss: str = 'mse',
                 batch_size: int = 32,
                 epochs: int = 1,
                 monitor: str = 'val_loss',
                 patience: int = 1,
                 optimizer: str = 'adam'):
        self.target = target
        self.hidden_layer_sizes = hidden_layer_sizes
        if not isinstance(self.hidden_layer_sizes, tuple):
            raise ValueError("hidden_layer_sizes must be a tuple")
        for size in self.hidden_layer_sizes: 
            if size > 0 and not isinstance(size, int):
                raise ValueError("Neuron counts must be positive integers")
            elif size < 0 and (size < -1 or size >= 0):
                raise ValueError("Dropout rates must be between -1 and 0")
        self.activation = activation 
        if self.activation not in ['relu', 'sigmoid', 'softmax', 'tanh']:
            raise ValueError(f"Invalid activation function: {self.activation}. Must be one of ['relu', 'sigmoid', 'softmax', 'tanh']")
        self.loss = loss
        if self.loss not in ['mse', 'binary_crossentropy', 'categorical_crossentropy']:
            raise ValueError(f"Invalid loss function: {self.loss}. Must be one of ['mse','binary_crossentropy','categorical_crossentropy']")
        self.optimizer = optimizer
        if self.optimizer not in ['adam', 'rmsprop', 'sgd']:
            raise ValueError(f"Invalid optimizer: {self.optimizer}. Must be one of ['adam', 'rmsprop', 'sgd']")
        self.batch_size = batch_size
        if self.batch_size < 0:
            raise ValueError("batch_size should be a positive integer")
        self.epochs = epochs
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        self.monitor = monitor
        if self.monitor not in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
            raise ValueError(f"Invalid monitor: {self.monitor}. Must be one of ['loss', 'val_loss', 'accuracy', 'val_accuracy']")
        self.patience = patience if patience is not None else self.epochs
        if self.patience <= 0:
            raise ValueError("patience must be a positive integer")
        self.mode = mode
        if self.mode not in ['auto', 'min', 'max']:
            raise ValueError("mode must be one of ['auto', 'min', 'max']")
        self.verbose = verbose
        if self.verbose not in [0, 1, 2]:
            raise ValueError("verbose must be 0, 1, or 2")

        self.classes_ = []
        self.loss_ = None
        self.best_loss = None
        self.features_ = None
        self.n_layers = None
        self.n_outputs_ = 1
        self.out_activation_ = None

        if isinstance(data_set, str):
            self.data_set = pd.read_csv(data_set)
        elif isinstance(data_set, pd.DataFrame):
            self.data_set = data_set
        else:
            raise ValueError("data_set must be a filepath (str) or a pandas DataFrame")
        self.preprocess_data()

    def preprocess_data(self):
        """Preprocess the dataset by adding engineered features and scaling."""
        self.data_set = self.data_set.dropna()
        self.data_set['distance_to_coast'] = np.abs(self.data_set['longitude'] - (-122))
        self.data_set['rooms_per_household'] = self.data_set['total_rooms'] / self.data_set['households']
        self.data_set['bedrooms_per_room'] = self.data_set['total_bedrooms'] / self.data_set['total_rooms']
        self.data_set['population_per_household'] = self.data_set['population'] / self.data_set['households']
        self.data_set['lat_lon_interaction'] = self.data_set['latitude'] * self.data_set['longitude']

        categorical_cols = self.data_set.select_dtypes(include=['object']).columns
        self.X = pd.get_dummies(self.data_set.drop(self.target, axis=1), columns=categorical_cols, drop_first=True)
        self.X_columns = list(self.X.columns)
        self.X = self.X.values
        self.y = self.data_set[self.target].values

        self.features_ = self.X_columns

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=101)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self.y_scaler = StandardScaler()
        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test_scaled = self.y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()

    def fit(self):
        """Fit the neural network model to the training data.

        This method builds a sequential neural network using the specified architecture,
        compiles it with the chosen optimizer and loss function, and trains it on the
        preprocessed training data. It includes early stopping and learning rate scheduling
        to prevent overfittin.

        Raises:
            ValueError: If the model has not been properly initialized.
        """
        tf.random.set_seed(101)
        model = Sequential()
        self.model = model
        model.add(Input(shape=(self.X_train_scaled.shape[1],)))
        model.add(Dense(self.hidden_layer_sizes[0], activation=self.activation))
        hidden_layer_count = 1
        for size in self.hidden_layer_sizes[1:]:
            if size > 0:
                model.add(Dense(size, activation = self.activation))
                hidden_layer_count += 1
            else:
                model.add(Dropout(abs(size)))
        model.add(Dense(1))

        self.n_layers = 1 + hidden_layer_count + 1
        self.out_activation_ = 'linear'

        optimizers = {
            'adam': tf.keras.optimizers.Adam(learning_rate=0.0005),
            'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=0.005),
            'sgd': tf.keras.optimizers.SGD()
        }

        losses = {
            'mse': MeanSquaredError(),
            'binary_crossentropy': BinaryCrossentropy(),
            'categorical_crossentropy': CategoricalCrossentropy(),
        }

        model.compile(
            optimizer=optimizers[self.optimizer],
            loss =losses[self.loss],
            metrics=['mae'],
        )
        self.early_stop = EarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode
        )
        self.lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.monitor,
            factor=0.5,
            patience=self.patience,
            min_lr=0.0001
        )
        self.history = model.fit(
            x=self.X_train_scaled,
            y=self.y_train_scaled,
            epochs=self.epochs, 
            batch_size=self.batch_size,
            validation_data=(self.X_test_scaled, self.y_test_scaled),
            callbacks=[self.early_stop, self.lr_schedule],
            verbose=self.verbose
        )

        self.loss_ = float(self.history.history['loss'][-1])
        self.best_loss = float(min(self.history.history['loss']))
    
    def model_loss(self):
        """
        Method to return Pandas DataFrame for all loss, 
        val_loss , accuracy and val_accuracy values."""
        if not hasattr(self, 'history'):
            raise ValueError("Model has not been trained yet. Call fit() first.")
        losses = pd.DataFrame(self.history.history)
        return losses

    def plot_model_loss(self):
        """Method Show the plotting of model’s loss and val_loss 
            against number of epochs"""
        if not hasattr(self, 'history'): 
            raise ValueError("Model has not been trained yet. Call fit() first.")
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

    def model_predict(self, row):
        """Take in a row of data set and return a predict.  
        Ps. Check needed processing for the input to 
        get the right prediction.    
        """
        if not hasattr(self, 'model'):
            raise ValueError('Model has not been trained yet. Call fit() first.')
        row = pd.DataFrame([row])
        required_cols = ['longitude', 'latitude', 'total_rooms', 'total_bedrooms', 'population', 'households']
        missing_cols = [col for col in required_cols if col not in row.columns]
        if missing_cols:
            raise ValueError(f"Input row is missing required columns: {missing_cols}")
        row['distance_to_coast'] = np.abs(row['longitude'] - (-122))
        row['rooms_per_household'] = row['total_rooms'] / row['households']
        row['bedrooms_per_room'] = row['total_bedrooms'] / row['total_rooms']
        row['population_per_household'] = row['population'] / row['households']
        row['lat_lon_interaction'] = row['latitude'] * row['longitude']

        categorical_cols = row.select_dtypes(include=['object']).columns
        row_X = pd.get_dummies(row, columns=categorical_cols, drop_first=True)
        row_X = row_X.reindex(columns=self.X_columns, fill_value=0)
        
        row_scaled = self.scaler.transform(row_X.values)
        row_scaled = row_scaled.reshape(1, -1)
        row_prediction = self.model.predict(row_scaled)
        row_prediction = self.y_scaler.inverse_transform(row_prediction.reshape(-1, 1)).flatten()

        return row_prediction[0]
    
    def save_original(self, path):
        """Save Keras model ( NOT MyANN model)  in .h5 format"""
        if not hasattr(self, 'model'):
            raise ValueError('Model has not been trained yet. call fit() first.')
        self.model.save(path)

    def load_original(self, path):
        """load saved Keras model ( NOT ANN model) from .h5 format"""
        try:
            self.model = tf.keras.models.load_model(path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        except Exception as e:
            raise ValueError(f"Failed to load model from {path}: {str(e)}")

    def plot_predictions_scatter(self):
        """Show scatter plot of real values in target columns 
        against predict value"""
        if not hasattr(self, 'model'):
            raise ValueError('Model has not been trained yet. Call fit() first.')
        try:
            pred = self.model.predict(self.X_test_scaled)
            pred = pred.reshape(-1, 1)
            y_pred = self.y_scaler.inverse_transform(pred)
            y_pred = y_pred.flatten()

            plt.figure(figsize=(15, 10))
            plt.scatter(x=self.y_test, y=y_pred, label='Predictions')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title('True vs Predicted Values')
            plt.show()
        except Exception as e:
            raise ValueError(f"Failed to load prediceted value from {pred}: {str(e)}")

    def rmse(self):
        """Return sqrt of the mean squared error"""
        if not hasattr(self, 'model'):
            raise ValueError('Model has not been trained yet. Call fit() first')
        try:
            pred = self.model.predict(self.X_test_scaled)
            pred = pred.reshape(-1, 1)
            y_pred = self.y_scaler.inverse_transform(pred)
            y_pred = y_pred.flatten()
            rmse = mean_squared_error(self.y_test, y_pred) ** (1/2)
            return rmse
        except Exception as e:
            raise ValueError(f"Failed to load prediceted value from {pred}: {str(e)}")

    def mae(self):
        """Return MAE"""
        if not hasattr(self, 'model'):
            raise ValueError('Model has not been trained yet. Call fit() first')
        try:
            pred = self.model.predict(self.X_test_scaled)
            pred = pred.reshape(-1, 1)
            y_pred = self.y_scaler.inverse_transform(pred)
            y_pred = y_pred.flatten()
            mae = mean_absolute_error(self.y_test, y_pred)
            return mae
        except Exception as e:
            raise ValueError(f"Failed to load prediceted value from {pred}: {str(e)}")

    def plot_residual_error(self):
        """Shows displot for residual error. where residual errors
        are the difference between real label and predicted label  values"""
        if not hasattr(self, 'model'):
            raise ValueError('Model has not been trained yet. Call fit() first')
        try:
            pred = self.model.predict(self.X_test_scaled)
            pred = pred.reshape(-1, 1)
            y_pred = self.y_scaler.inverse_transform(pred)
            y_pred = y_pred.flatten()
            re = self.y_test - y_pred
            sns.histplot(re, kde=True)
            plt.xlabel('Residual Error (True - Predicted)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Residual Errors')
            plt.show()
        except Exception as e:
            raise ValueError(f"Failed to load prediceted value from {pred}: {str(e)}")

    @classmethod
    def run(cls):
        """Run a full training and evaluation pipeline on the California Housing dataset."""
        my_ann = MyAnn('housing.csv', 'median_house_value', 'min', 2, hidden_layer_sizes=(128, 64, -0.25, 32, 16, 8), epochs=100, patience=5, optimizer='adam')
        my_ann.fit()
        my_ann.save_original('housing_model.h5')
        my_ann.load_original('housing_model.h5')
        row = pd.read_csv('housing.csv').iloc[1]
        prediction = my_ann.model_predict(row)
        my_ann.plot_predictions_scatter()
        rmse = my_ann.rmse()
        mae = my_ann.mae()
        my_ann.plot_residual_error()
        pml = my_ann.plot_model_loss()

        print("Classes:", my_ann.classes_)
        print("Last Loss:", my_ann.loss_)
        print("Best Loss:", my_ann.best_loss)
        print("Features:", my_ann.features_)
        print("Number of Layers:", my_ann.n_layers)
        print("Number of Outputs:", my_ann.n_outputs_)
        print("Output Activation:", my_ann.out_activation_)

        print("Prediction:", prediction)
        print('plot model losses:', pml)
        print("RMSE:", rmse)
        print('MAE:', mae)
        print('Residual error plot displayed')

if __name__ == "__main__":
    MyAnn.run()