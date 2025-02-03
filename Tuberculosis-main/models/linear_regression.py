'''
To run this script
python linear_reression.py --split random

The split data are generated using the script splitting.py

Train data (used for train validation and test)
15,619 /rds/general/user/qg622/home/Y2/Antibacterial/data/Mtb_published_regression_AC_Cleaned.csv

External Test data (not used yet)
1194 /rds/general/user/qg622/home/Y2/Antibacterial/data/RCB_Mtb_inhibition_2007-2019.csv
259 /rds/general/user/qg622/home/Y2/Antibacterial/data/2017_Mtb_testset.csv

'''

import os
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import json

class DataLoader:
    def __init__(self, data_dir, split_type):
        self.data_dir = data_dir
        self.split_type = split_type

    def load_data(self):
        train_data = pd.read_csv(os.path.join(self.data_dir, f'Mtb_{self.split_type}_train.csv'))
        val_data = pd.read_csv(os.path.join(self.data_dir, f'Mtb_{self.split_type}_val.csv'))
        test_data = pd.read_csv(os.path.join(self.data_dir, f'Mtb_{self.split_type}_test.csv'))
        return train_data, val_data, test_data

class FingerprintGenerator:
    @staticmethod
    def smiles_to_fingerprint(smiles, bit_length=256):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=bit_length)
            return np.array(fingerprint)
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            return np.zeros(bit_length)

class ModelTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def grid_search(self, param_grid):
        best_params = None
        best_val_mse = float('inf')
        best_model = None

        print("Starting grid search...")
        for i, param in enumerate(param_grid, 1):
            print(f"\nIteration {i}/{len(param_grid)}: Testing parameters {param}")
            
            # Set model parameters and train
            self.model.set_params(**param)
            print(f"Training model with parameters: {param}")
            self.model.fit(self.X_train, self.y_train)
            
            # Predict on validation set
            val_pred = self.model.predict(self.X_val)
            val_mse = mean_squared_error(self.y_val, val_pred)
            print(f"Validation MSE: {val_mse}")
            
            # Update best parameters if current model is better
            if val_mse < best_val_mse:
                print(f"New best model found! Validation MSE improved from {best_val_mse} to {val_mse}")
                best_params = param
                best_val_mse = val_mse
                best_model = self.model
            else:
                print(f"No improvement. Current best validation MSE: {best_val_mse}")

        print("\nGrid search completed.")
        print(f"Best parameters: {best_params}")
        print(f"Best validation MSE: {best_val_mse}")
        
        return best_model, best_params, best_val_mse

class Evaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        return test_mse, test_r2, test_pred

def main():
    parser = argparse.ArgumentParser(description="Linear regression on antibacterial data")
    parser.add_argument('--split', type=str, required=True, choices=['random', 'scaffold', 'butina', 'umap'], help="Data split type")
    parser.add_argument('--X_column', type=str, default='SMILES', choices=['random', 'scaffold', 'butina', 'umap'], help="Data split type")
    parser.add_argument('--y_column', type=str, default='value', choices=['random', 'scaffold', 'butina', 'umap'], help="Data split type")
    args = parser.parse_args()

    work_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(work_dir, 'data', 'data_after_split')
    split_results_dir = os.path.join(work_dir, 'results', 'lr', f'split_{args.split}')
    os.makedirs(split_results_dir, exist_ok=True)

    data_loader = DataLoader(data_dir, args.split)
    train_data, val_data, test_data = data_loader.load_data()

    for dataset, name in zip([train_data, val_data, test_data], ['Train', 'Validation', 'Test']):
        if args.X_column not in dataset.columns or args.y_column not in dataset.columns:
            raise ValueError(f"Columns '{args.X_column}' or '{args.y_column}' not found in {name} dataset")

    fingerprint_generator = FingerprintGenerator()
    train_X = np.array([fingerprint_generator.smiles_to_fingerprint(smiles) for smiles in train_data[args.X_column]])
    val_X = np.array([fingerprint_generator.smiles_to_fingerprint(smiles) for smiles in val_data[args.X_column]])
    test_X = np.array([fingerprint_generator.smiles_to_fingerprint(smiles) for smiles in test_data[args.X_column]])

    train_y = train_data[args.y_column]
    val_y = val_data[args.y_column]
    test_y = test_data[args.y_column]

    model = LinearRegression()
    trainer = ModelTrainer(model, train_X, train_y, val_X, val_y)
    param_grid = [{'fit_intercept': True}, {'fit_intercept': False}]
    best_model, best_params, best_val_mse = trainer.grid_search(param_grid)
    
    # Save best parameters
    with open(os.path.join(split_results_dir, 'best_params.json'), "w") as f:
        json.dump(best_params, f)
            
    print(f"Best Fit Intercept: {best_params}, Best Validation MSE: {best_val_mse}")

    evaluator = Evaluator()
    test_mse, test_r2, test_pred = evaluator.evaluate_model(best_model, test_X, test_y)
    print(f"Test Mean Squared Error: {test_mse}")
    print(f"Test R^2 Score: {test_r2}")

    test_results = pd.DataFrame({
        'SMILES': test_data[args.X_column],
        'y_true': test_y,
        'y_pred': test_pred
    })
    test_results_path = os.path.join(split_results_dir, 'test_predictions.csv')
    test_results.to_csv(test_results_path, index=False)
    print(f"Test predictions saved to: {test_results_path}")

if __name__ == "__main__":
    main()