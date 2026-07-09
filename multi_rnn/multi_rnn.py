import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import uuid
import random

data = pd.read_csv('Sales_Monthly.csv', index_col='DATE', parse_dates=True)

train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]


# Set random seeds for reproducibility
def set_seeds(seed=101):
    # Set seeds for various libraries
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


set_seeds(101)

class MultiRNN:
    def __init__(self,  train: pd.DataFrame = train, 
                        test: pd.DataFrame = test,
                        length: int = 10, 
                        LSTM_units: int | list= 50,
                        activation: str | list = 'tanh', 
                        optimizer: str | list = 'adam',
                        batch_size: int = 1, 
                        epochs: int | list = 25,
                        random_seed: int = 101):
        """
        Unitialize MultiRNN with hyperparameters and validate inputs,
        calls generate_model_list_per_column_in_data_set to build models.
        """
        set_seeds(random_seed)
        # Validate inputs
        self.train = self.ready_for_processing(train)
        self.test = self.ready_for_processing(test)
        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer")
        self.length = length
        self.batch_size = batch_size if isinstance(batch_size, int) and batch_size > 0 else 1
        
        # Handle single value or list for LSTM_units, activation, optimizer, epochs
        self.LSTM_units = self._validate_param(LSTM_units, int, train.columns, "LSTM_units")
        self.activation = self._validate_param(activation, str, train.columns, "activation", 
                                             valid_values=['tanh', 'sigmoid', 'softmax', 'relu'])
        self.optimizer = self._validate_param(optimizer, str, train.columns, "optimizer", 
                                            valid_values=['adam', 'rmsprop', 'sgd'])
        self.epochs = self._validate_param(epochs, int, train.columns, "epochs")
        
        # Dictionary to store models, losses, and names
        self.model_dict = {}
        
        # Generate models for each column
        self.generate_model_list_per_column_in_data_set(
            train, test, 
            length, self.LSTM_units,
            self.activation, self.optimizer, 
            self.batch_size, self.epochs
        )

    def _validate_param(self, param, param_type, columns, param_name, valid_values=None):
        """
        Validate if param is a single value or list matching the number of columns.
        """
        if isinstance(param, list):
            if len(param) != len(columns):
                raise ValueError(f"{param_name} list length must match number of columns")
            for p in param:
                if not isinstance(p, param_type) or (valid_values and p not in valid_values):
                    raise ValueError(f"Invalid {param_name}: {p}")
            return dict(zip(columns, param))
        else:
            if not isinstance(param, param_type) or (valid_values and param not in valid_values):
                raise ValueError(f"Invalid {param_name}: {param}")
            return dict(zip(columns, [param] * len(columns)))

    def ready_for_processing(self, data_set: pd.DataFrame) -> pd.DataFrame:
        """
        Check if dataset is ready: no NaN, no object dtypes, valid index.
        """
        if not isinstance(data_set, pd.DataFrame):
            raise ValueError("Dataset must be a pandas DataFrame")
        if data_set.isna().any().any():
            raise ValueError("Dataset contains NaN values")
        if data_set.index.inferred_type not in ['datetime64', 'int64', 'float64']:
            raise ValueError("Dataset index must be datetime or numeric")
        if any(data_set.dtypes == 'object'):
            raise ValueError("Dataset contains object data types")
        return data_set

    def generate_data_set_per_column_with_original_index(self, data_set: pd.DataFrame) -> dict:
        """
        Generate single-column datasets with original index for each feature.
        """
        datasets = {}
        for column in data_set.columns:
            datasets[column] = pd.DataFrame(data_set[column], index=data_set.index)
        return datasets

    def build_model_per_column(self, train: pd.DataFrame, test: pd.DataFrame, 
                              length: int, LSTM_units: int, activation: str, 
                              optimizer: str, batch_size: int, epochs: int):
        """
        Build and train an LSTM model for a single-column dataset.
        Includes scaling, time series generation, and early stopping.
        Returns model, losses DataFrame, and model name.
        """
        # Prepare scaler
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
        test_scaled = scaler.transform(test.values.reshape(-1, 1))

        # Generate time series sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_scaled, length)
        X_test, y_test = create_sequences(test_scaled, length)

        # Define LSTM model
        # Using LSTM layer for temporal dependencies, followed by Dense for output
        model = Sequential([
            LSTM(units=LSTM_units, activation=activation, input_shape=(length, 1), 
                 return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mse')

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Train model / internal forward pass and backpropagation
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                           validation_data=(X_test, y_test), callbacks=[early_stopping], 
                           verbose=0)

        # Create losses DataFrame
        losses = pd.DataFrame({
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        })

        # Generate predictions for evaluation
        predictions_scaled = model.predict(X_test, verbose=0) 
        predictions = scaler.inverse_transform(predictions_scaled)
        test_indices = test.index[length:len(predictions) + length]
        predictions_df = pd.DataFrame(predictions, index=test_indices, columns=['predicted'])

        # Generate unique model name
        model_name = f"model_{uuid.uuid4().hex}.h5"
        
        # Save model 
        model.save(model_name)

        return model, losses, model_name, predictions_df

    def generate_model_list_per_column_in_data_set(self, train: pd.DataFrame, 
                                                  test: pd.DataFrame, length: int, 
                                                  LSTM_units: dict, activation: dict, 
                                                  optimizer: dict, batch_size: int, 
                                                  epochs: dict):
        """
        Create a dictionary of models for each column by generating datasets and building models.
        Stores results in self.model_dict.
        """
        # Generate single-column datasets
        train_datasets = self.generate_data_set_per_column_with_original_index(train)
        test_datasets = self.generate_data_set_per_column_with_original_index(test)

        # Build a model for each feature
        # Iterates over columns to create independent LSTM models
        for column in train.columns:
            train_col = train_datasets[column]
            test_col = test_datasets[column]
            
            model, losses, model_name, predictions_df = self.build_model_per_column(
                train_col, test_col, length, 
                LSTM_units[column], activation[column], optimizer[column], 
                batch_size, epochs[column]
            )
            
            self.model_dict[column] = {
                'model': model,
                'losses': losses,
                'model_name': model_name,
                'predictions': predictions_df
            }

        return self.model_dict

    def predict(self, data_row: pd.DataFrame) -> pd.DataFrame:
        """
        Predict values for all features using trained models.
        Input is a DataFrame with same columns as training data.
        """
        if not isinstance(data_row, pd.DataFrame) or sorted(data_row.columns) != sorted(self.model_dict.keys()):
            raise ValueError("Input DataFrame must have same columns as training data")

        predictions = {}
        scaler = MinMaxScaler()
        
        for column in data_row.columns:
            model = self.model_dict[column]['model']
            data = data_row[column].values.reshape(-1, 1)
            scaled_data = scaler.fit_transform(data)
            
            # Prepare sequence for prediction
            if len(scaled_data) >= self.length:
                X = scaled_data[-self.length:].reshape(1, self.length, 1)
                pred_scaled = model.predict(X, verbose=0) # internal forward pass
                pred = scaler.inverse_transform(pred_scaled)
                predictions[column] = pred.flatten()[0]
            else:
                predictions[column] = np.nan  

        return pd.DataFrame([predictions], index=data_row.index)

    def plot_predict_against_test_dataset_per_column(self, column: str, 
                                                    figure_width: int, 
                                                    figure_hight: int, 
                                                    save_plot_name: str = 'plot_test_vs_predict_'):
        """
        Plot predicted vs actual test values for a specific column.
        """
        if column not in self.model_dict:
            raise ValueError(f"Column {column} not found in trained models")
        
        test_data = self.test[column]
        predictions = self.model_dict[column]['predictions']
        
        plt.figure(figsize=(figure_width, figure_hight))
        plt.plot(test_data.index, test_data, label='Actual')
        plt.plot(predictions.index, predictions['predicted'], label='Predicted')
        plt.title(f'Predictions vs Actual for {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_plot_name}{column}.png")
        plt.close()

    def plot_loss_val_loss_per_column(self, column: str, figure_width: int, 
                                      figure_hight: int, 
                                      save_plot_name: str = 'Plot_Loss_val_loss_'):
        """
        Plot loss and validation loss for a specific column.
        """
        if column not in self.model_dict:
            raise ValueError(f"Column {column} not found in trained models")
        
        losses = self.model_dict[column]['losses']
        
        plt.figure(figsize=(figure_width, figure_hight))
        plt.plot(losses['loss'], label='Training Loss')
        plt.plot(losses['val_loss'], label='Validation Loss')
        plt.title(f'Loss vs Validation Loss for {column}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_plot_name}{column}.png")
        plt.close()


    def run_all(self):
        """
        Run all models and plot results for each column.
        """
        for column in self.model_dict.keys():
            self.plot_predict_against_test_dataset_per_column(column, 10, 6)
            self.plot_loss_val_loss_per_column(column, 10, 6)
            prediction_input = self.test.iloc[-self.length:]
            predictions = self.predict(prediction_input)
            print(f"Predictions for {column}: {predictions}")


if __name__ == "__main__":
    set_seeds(101)
    multi_rnn = MultiRNN()
    multi_rnn.run_all()
    print("All models trained and plots generated.")
    #print(data.isna().any())
    #print(data.dtypes)