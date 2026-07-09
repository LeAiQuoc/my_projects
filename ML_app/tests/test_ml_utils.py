import unittest
from unittest.mock import patch, Mock, ANY
import pandas as pd
import numpy as np
from project1 import ml_utils


class TestMLUtils(unittest.TestCase):

    # Test for load_csv function
    @patch('project1.ml_utils.pd.read_csv')
    def test_load_csv(self, mock_read_csv):
        # Mocking successful read
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        df = ml_utils.load_csv('fake_path.csv')
        self.assertTrue('col1' in df.columns)
        self.assertTrue('col2' in df.columns)

        # Mock file not found error
        mock_read_csv.side_effect = FileNotFoundError('File not found')

        # Testing exception
        with self.assertRaises(FileNotFoundError):
            ml_utils.load_csv('non_existent_path.csv')

    @patch('project1.ml_utils.train_test_split')
    @patch('project1.ml_utils.StandardScaler')
    def test_preprocess_data(self, mock_scaler, mock_train_test_split):
        # Create fake data
        fake_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'target': [0, 1, 0]})
        
        # Mock standard scaler behavior
        mock_scaler_instance = mock_scaler.return_value
        mock_scaler_instance.fit_transform.return_value = np.array([[-1, -1], [0, 0], [1, 1]])
        mock_scaler_instance.transform.return_value = np.array([[-1, -1], [0, 0], [1, 1]])

        # Mock train_test_split to return fake data
        mock_train_test_split.return_value = (
            fake_data.drop('target', axis=1), 
            fake_data.drop('target', axis=1), 
            fake_data['target'], 
            fake_data['target']
        )

        # Call preprocess_data method
        X_train_scaled, X_test_scaled, y_train, y_test = ml_utils.preprocess_data(fake_data, 'target', 'classifier')

        # Check that fit_transform and transform were called
        mock_scaler_instance.fit_transform.assert_called_once_with(ANY)
        mock_scaler_instance.transform.assert_called_once_with(ANY)

        # Assert that the returned values are the mocked ones
        self.assertEqual(X_train_scaled.tolist(), [[-1, -1], [0, 0], [1, 1]])
        self.assertEqual(X_test_scaled.tolist(), [[-1, -1], [0, 0], [1, 1]])

    @patch('project1.ml_utils.classification_report')
    @patch('project1.ml_utils.accuracy_score')
    @patch('project1.ml_utils.confusion_matrix')
    @patch('project1.ml_utils.GridSearchCV')
    @patch('project1.ml_utils.LogisticRegression')
    @patch('project1.ml_utils.KNeighborsClassifier')
    @patch('project1.ml_utils.SVC')
    def test_run_classifiers(self, mock_svc, mock_knn, mock_lr, mock_grid_search, mock_confusion_matrix, mock_accuracy_score, mock_classification_report):
        # Fake data
        fake_X_train = pd.DataFrame({'col1': [1,2,3]})
        fake_y_train = pd.Series([0,1,0])
        fake_X_test = pd.DataFrame({'col1': [4,5,6]})
        fake_y_test = pd.Series([0,1,0])

        # Mock Logistic Regression
        mock_lr_instance = mock_lr.return_value
        mock_lr_instance.fit.return_value = mock_lr_instance

        # Mock KNN
        mock_knn_instance = mock_knn.return_value
        mock_knn_instance.fit.return_value = mock_knn_instance

        # Mock SVC
        mock_svc_instance = mock_svc.return_value
        mock_svc_instance.fit.return_value = mock_svc_instance

        # Mock GridSearchCV for each classifier
        mock_grid_instance = mock_grid_search.return_value
        mock_grid_instance.fit.return_value = mock_grid_instance
        mock_grid_instance.predict.return_value = [0,1,0]
        mock_grid_instance.best_params_ = {'param': 'value'}
        mock_grid_instance.best_estimator_ = mock_lr_instance

        # Mock metrics
        mock_accuracy_score.return_value = 0.85
        mock_classification_report.return_value = {'macro avg': {'f1-score': 0.85}}
        mock_confusion_matrix.return_value = [[1,0],[0,2]]

        # Call run_classifiers
        results = ml_utils.run_classifiers(fake_X_train, fake_y_train, fake_X_test, fake_y_test)

        # Assert calls
        mock_lr.assert_called()
        mock_knn.assert_called()
        mock_svc.assert_called()
        mock_grid_search.assert_called()
        mock_accuracy_score.assert_called()
        mock_classification_report.assert_called()
        mock_confusion_matrix.assert_called()

        # Assert results structure
        self.assertIn('LoR', results)
        self.assertIn('KNN', results)
        self.assertIn('SVC', results)
        self.assertIn('best_params', results['LoR'])
        self.assertIn('accuracy', results['LoR'])
        self.assertIn('classification_report', results['LoR'])
        self.assertIn('confusion_matrix', results['LoR'])
        self.assertEqual(results['LoR']['accuracy'], 0.85)

    @patch('project1.ml_utils.mean_absolute_error')
    @patch('project1.ml_utils.mean_squared_error')
    @patch('project1.ml_utils.r2_score')
    @patch('project1.ml_utils.GridSearchCV')
    @patch('project1.ml_utils.LinearRegression')
    @patch('project1.ml_utils.Lasso')
    @patch('project1.ml_utils.Ridge')
    @patch('project1.ml_utils.ElasticNet')
    @patch('project1.ml_utils.SVR')
    def test_run_regressors(self, mock_svr, mock_elastic, mock_ridge, mock_lasso, mock_lr, mock_grid_search, mock_r2, mock_mse, mock_mae):
        # Fake data
        fake_X_train = pd.DataFrame({'col1': [1,2,3]})
        fake_y_train = pd.Series([0.1, 0.2, 0.3])
        fake_X_test = pd.DataFrame({'col1': [4,5,6]})
        fake_y_test = pd.Series([0.4, 0.5, 0.6])

        # Mock models
        mock_lr_instance = mock_lr.return_value
        mock_lr_instance.fit.return_value = mock_lr_instance

        mock_lasso_instance = mock_lasso.return_value
        mock_lasso_instance.fit.return_value = mock_lasso_instance

        mock_ridge_instance = mock_ridge.return_value
        mock_ridge_instance.fit.return_value = mock_ridge_instance

        mock_elastic_instance = mock_elastic.return_value
        mock_elastic_instance.fit.return_value = mock_elastic_instance

        mock_svr_instance = mock_svr.return_value
        mock_svr_instance.fit.return_value = mock_svr_instance

        # Mock GridSearchCV
        mock_grid = mock_grid_search.return_value
        mock_grid.fit.return_value = mock_grid
        mock_grid.predict.return_value = [0.4, 0.5, 0.6]
        mock_grid.best_params_ = {'param': 'value'}
        mock_grid.best_estimator_ = mock_lr_instance

        # Mock metrics
        mock_mae.return_value = 0.1
        mock_mse.return_value = 0.04
        mock_r2.return_value = 0.95

        # Call run_regressors
        results = ml_utils.run_regressors(fake_X_train, fake_y_train, fake_X_test, fake_y_test)

        # Assert calls
        mock_lr.assert_called()
        mock_lasso.assert_called()
        mock_ridge.assert_called()
        mock_elastic.assert_called()
        mock_svr.assert_called()
        mock_grid_search.assert_called()
        mock_mae.assert_called()
        mock_mse.assert_called()
        mock_r2.assert_called()

        # Assert results structure
        self.assertIn('LiR', results)
        self.assertIn('Lasso', results)
        self.assertIn('Ridge', results)
        self.assertIn('ElasticNet', results)
        self.assertIn('SVR', results)
        self.assertIn('best_params', results['LiR'])
        self.assertIn('MAE', results['LiR'])
        self.assertIn('RMSE', results['LiR'])
        self.assertIn('R2', results['LiR'])
        self.assertEqual(results['LiR']['MAE'], 0.1)
        self.assertEqual(results['LiR']['RMSE'], np.sqrt(0.04))
        self.assertEqual(results['LiR']['R2'], 0.95)

if __name__ == "__main__":
    unittest.main()