# project1/app.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from joblib import dump
from project1 import ml_utils


classifier_results = {}
regressor_results = {}
X_train_scaled = X_test_scaled = y_train = y_test = None

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Finder")
        self.root.geometry("800x600")

        self.ml_type_var = tk.StringVar(value="classifier")
        tk.Label(root, text="Select ML Type:").pack(pady=5)
        tk.Radiobutton(root, text="Classifier", variable=self.ml_type_var, value="classifier").pack()
        tk.Radiobutton(root, text="Regressor", variable=self.ml_type_var, value="regressor").pack()

        tk.Label(root, text="Select CSV File:").pack(pady=5)
        self.file_button = tk.Button(root, text="Browse", command=self.load_file)
        self.file_button.pack()
        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.pack()

        tk.Label(root, text="Enter Target Column:").pack(pady=5)
        self.target_entry = tk.Entry(root)
        self.target_entry.pack()
        self.columns_label = tk.Label(root, text="Available Columns: None")
        self.columns_label.pack(pady=5)

        self.proceed_button = tk.Button(root, text="Proceed", command=self.process_data, state="disabled")
        self.proceed_button.pack(pady=10)

        self.results_frame = tk.Frame(root)
        self.results_text = tk.Text(self.results_frame, height=20, width=80)
        self.results_text.pack(pady=5)

        self.data = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = ml_utils.load_csv(file_path)
                self.file_label.config(text=os.path.basename(file_path))
                self.columns_label.config(text=f"Available Columns: {', '.join(self.data.columns)}")
                self.proceed_button.config(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def process_data(self):
        global X_train_scaled, X_test_scaled, y_train, y_test
        target_col = self.target_entry.get().strip()
        ml_type = self.ml_type_var.get()

        try:
            X_train_scaled, X_test_scaled, y_train, y_test = ml_utils.preprocess_data(self.data, target_col, ml_type)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.show_results(ml_type)

    def show_results(self, ml_type):
        for widget in self.root.winfo_children():
            widget.pack_forget()
        self.results_frame.pack(fill="both", expand=True)
        self.results_text.delete(1.0, tk.END)

        global classifier_results, regressor_results
        if ml_type == "classifier":
            classifier_results = ml_utils.run_classifiers(X_train_scaled, X_test_scaled, y_train, y_test)
            results = classifier_results
        else:
            regressor_results = ml_utils.run_regressors(X_train_scaled, X_test_scaled, y_train, y_test)
            results = regressor_results

        self.results_text.insert(tk.END, f"{ml_type.capitalize()} Results:\n\n")
        for name, result in results.items():
            self.results_text.insert(tk.END, f"{name}: {result}\n\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
