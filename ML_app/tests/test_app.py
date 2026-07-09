import unittest
import tkinter as tk
from project1.app import MLApp
from unittest.mock import patch
import pandas as pd
import os
from project1 import app

class TestMLApp(unittest.TestCase):

    def setUp(self):
        # Set up Tkinter root for testing
        self.root = tk.Tk()
        self.app = MLApp(self.root)
        self.root.update_idletasks()
        self.root.update()

    def tearDown(self):
        self.root.destroy()
    
    def test_init(self):
        # Window properties
        self.assertEqual(self.app.root.title(), "ML Model Finder")
        geometry = self.app.root.geometry()
        self.assertTrue(geometry.startswith("800x600"))

        # ml_type_var
        self.assertIsInstance(self.app.ml_type_var, tk.StringVar)
        self.assertEqual(self.app.ml_type_var.get(), "classifier")

        # Labels
        label_texts = {
            "Select ML Type:",
            "Select CSV File:",
            "Enter Target Column:",
            "Available Columns: None"
        }
        found_label_texts = {
            widget.cget('text') for widget in self.app.root.winfo_children()
            if isinstance(widget, tk.Label)
        }
        for text in label_texts:
            self.assertIn(text, found_label_texts)

        # Radio buttons
        radio_texts_values = {"Classifier": "classifier", "Regressor": "regressor"}
        found_radio_texts = {
            widget.cget("text") for widget in self.app.root.winfo_children()
            if isinstance(widget, tk.Radiobutton)
        }
        found_radio_values = {
            widget.cget("value") for widget in self.app.root.winfo_children()
            if isinstance(widget, tk.Radiobutton)
        }
        for text, value in radio_texts_values.items():
            self.assertIn(text, found_radio_texts)
            self.assertIn(value, found_radio_values)

        # File button + label
        self.assertEqual(self.app.file_button.cget('text'), "Browse")
        self.assertEqual(self.app.file_label.cget('text'), "No file selected")

        # Target entry
        self.assertIsInstance(self.app.target_entry, tk.Entry)

        # Columns label
        self.assertEqual(self.app.columns_label.cget('text'), "Available Columns: None")

        # Proceed button
        self.assertEqual(self.app.proceed_button.cget('text'), "Proceed")
        self.assertEqual(self.app.proceed_button.cget('state'), "disabled")

        # Results frame/text
        self.assertEqual(self.app.results_text.cget('height'), 20)
        self.assertEqual(self.app.results_text.cget('width'), 80)

    @patch('project1.app.filedialog.askopenfilename', autospec=True)
    @patch('project1.app.ml_utils.load_csv', autospec=True)  
    @patch('tkinter.messagebox.showerror')
    def test_load_file(self, mock_showerror, mock_load_csv, mock_askopenfilename):
        # Success case
        mock_askopenfilename.return_value = 'fake_path.csv'
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_load_csv.return_value = df

        self.app.load_file()

        self.assertEqual(self.app.file_label.cget('text'), os.path.basename('fake_path.csv'))
        self.assertEqual(self.app.columns_label.cget('text'), "Available Columns: col1, col2")
        self.assertEqual(self.app.proceed_button.cget('state'), 'normal')

        # No file selected
        mock_askopenfilename.return_value = ''
        self.app.proceed_button.config(state='disabled')
        self.app.load_file()
        self.assertEqual(self.app.proceed_button.cget('state'), 'disabled')

        # Error case
        mock_askopenfilename.return_value = 'fake_path.csv'
        mock_load_csv.side_effect = Exception('File read error')
        self.app.load_file()
        mock_showerror.assert_called_with("Error", "Failed to load file: File read error")

    @patch('project1.ml_utils.preprocess_data')
    def test_process_data_success(self, mock_preprocess_data):
        # Setup fake data and widgets
        fake_data = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6], 'target': [0,1,0]})
        self.app.data = fake_data
        self.app.ml_type_var.set("classifier")
        self.app.target_entry.insert(0, "col1")

        # Mock preprocess_data return value
        mock_preprocess_data.return_value = (None, None, None, None)

        with patch.object(self.app, 'show_results') as mock_show_results:
            self.app.process_data()
            mock_show_results.assert_called_once_with("classifier")

        # Assert preprocess_data was called correctly
        args, kwargs = mock_preprocess_data.call_args
        self.assertTrue(args[0].equals(fake_data))


    @patch('project1.ml_utils.preprocess_data', side_effect=Exception('Some error'))
    @patch('tkinter.messagebox.showerror')
    def test_process_data_failure(self, mock_show_error, mock_preprocess_data):
        # Setup fake data and widgets
        fake_data = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6], 'target': [0,1,0]})
        self.app.data = fake_data
        self.app.ml_type_var.set("classifier")
        self.app.target_entry.insert(0, "col1")

        # Call the method (will raise exception via mock)
        self.app.process_data()

        # Assert messagebox.showerror 
        mock_show_error.assert_called_with("Error", "Some error")

    @patch('project1.app.ml_utils.run_classifiers')
    @patch('project1.app.ml_utils.run_regressors')
    def test_show_results(self, mock_run_regressors, mock_run_classifiers):
        # Fake results for testing
        fake_classifier_results = {
            'LoR': {'accuracy': 0.9, 'best_params': {'C': 1.0}},
            'KNN': {'accuracy': 0.8, 'best_params': {'n_neighbors': 3}},
            'SVC': {'accuracy': 0.85, 'best_params': {'C': 1.0, 'kernel': 'linear'}}
        }

        fake_regressor_results = {
            'Lasso': {'MSE': 1.0, 'best_params': {'C': 1.0}, 'R2': 0.9},  # Add Lasso here
            'LiR': {'MSE': 1.0, 'best_params': {'C': 1.0}, 'R2': 0.9}, 
            'SVR': {'MSE': 1.0, 'best_params': {'C': 1.0}, 'R2': 0.9},  
            'ElasticNet': {'MSE': 1.0, 'best_params': {'C': 1.0}, 'R2': 0.9},
            'Ridge' : {'MSE': 1.0, 'best_params': {'C': 1.0}, 'R2': 0.9} 
        }


        # Setup global variables for testing
        app.X_train_scaled = [[1, 2], [3, 4]]  
        app.X_test_scaled = [[5, 6], [7, 8]]   
        app.y_train = [0, 1]                   
        app.y_test = [1, 0]

        # Mock the external function calls
        mock_run_classifiers.return_value = fake_classifier_results
        mock_run_regressors.return_value = fake_regressor_results

        # Test classifier results display
        self.app.show_results("classifier")

        # Force the app to update UI elements
        self.app.root.update()  

        # Ensure that the results frame is properly packed and visible
        self.assertTrue(self.app.results_frame.winfo_ismapped())
        self.assertGreater(self.app.results_frame.winfo_height(), 0)
        self.assertGreater(self.app.results_frame.winfo_width(), 0)

        # Retrieve and assert content in results_text widget
        content = self.app.results_text.get(1.0, tk.END)
        
        # Assert title and content for classifier results
        self.assertIn("Classifier Results:", content)
        self.assertIn("LoR", content)  
        self.assertIn("accuracy", content)  
        self.assertIn("KNN", content)  
        self.assertIn("SVC", content)  
        self.assertIn("best_params", content)

        # Assert that the accuracy value for 'LoR' is within the expected range
        accuracy = fake_classifier_results['LoR']['accuracy']
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

        # Assert the classifier function call
        mock_run_classifiers.assert_called_once_with(app.X_train_scaled, app.X_test_scaled, app.y_train, app.y_test)

        # Test regressor results display
        self.app.show_results("regressor")

        # Force the app to update UI elements again (after showing regressor results)
        self.app.root.update()

        # Ensure the frame is packed after calling show_results
        self.assertTrue(self.app.results_frame.winfo_ismapped())
        content = self.app.results_text.get(1.0, tk.END)

        # Assert title and content for regressor results
        self.assertIn("Lasso", content)
        self.assertIn("Ridge", content)
        self.assertIn("best_params", content)

        # Assert that the MSE value for 'LiR' is of type float and greater than 0
        mse = fake_regressor_results['LiR']['MSE']
        self.assertIsInstance(mse, float)
        self.assertGreater(mse, 0)

        # Assert the regressor function call
        mock_run_regressors.assert_called_once_with(app.X_train_scaled, app.X_test_scaled, app.y_train, app.y_test)


if __name__ == "__main__":
    unittest.main()
