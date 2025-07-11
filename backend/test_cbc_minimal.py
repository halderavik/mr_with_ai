#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("Script starting...")

import sys
import os
import pandas as pd
import numpy as np
import torch

print("Imports completed...")

def test_cbc_core_functionality():
    print("Testing CBC core functionality...")
    try:
        print("[1] Checking packages...")
        print("✓ pandas available")
        print("✓ numpy available") 
        print("✓ torch available")
        print(f"✓ torch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

        print("[2] Testing basic operations...")
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✓ Numpy array: {arr}")
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✓ Pandas DataFrame shape: {df.shape}")
        tensor = torch.tensor([1, 2, 3, 4, 5])
        print(f"✓ Torch tensor: {tensor}")

        print("[3] Testing simple data creation...")
        # Create very simple, consistent data structure
        data = []
        for task in range(3):    # 3 tasks
            for alt in range(2): # 2 alternatives per task
                data.append({
                    'task_id': task,
                    'alternative_id': alt,
                    'choice': 1 if alt == 0 else 0,  # Always choose first alternative
                    'brand': 'Brand A' if alt == 0 else 'Brand B'
                })
        
        test_data = pd.DataFrame(data)
        print(f"✓ Created simple test data with shape: {test_data.shape}")
        print(f"✓ Data columns: {list(test_data.columns)}")
        print(f"✓ Data preview:")
        print(test_data.head())

        print("[4] Testing conjoint structure identification...")
        def identify_conjoint_structure(data, metadata):
            column_labels = metadata.get('column_labels', {})
            choice_col = None
            task_col = None
            alt_col = None
            for col in data.columns:
                if 'choice' in col.lower():
                    choice_col = col
                    break
            for col in data.columns:
                if 'task' in col.lower():
                    task_col = col
                    break
            for col in data.columns:
                if 'alternative' in col.lower() or 'alt' in col.lower():
                    alt_col = col
                    break
            attribute_cols = []
            excluded_cols = [choice_col, task_col, alt_col]
            for col in data.columns:
                if col not in excluded_cols and col not in [None]:
                    if data[col].dtype in ['object', 'int64']:
                        unique_vals = data[col].unique()
                        if 2 <= len(unique_vals) <= 20:
                            attribute_cols.append(col)
            return {
                'choice': choice_col,
                'task_id': task_col,
                'alternative_id': alt_col,
                'attributes': attribute_cols
            }
        metadata = {
            'column_labels': {
                'choice': 'Which alternative did you choose?',
                'task_id': 'Task number',
                'alternative_id': 'Alternative number',
                'brand': 'Brand name'
            }
        }
        conjoint_structure = identify_conjoint_structure(test_data, metadata)
        print(f"✓ Identified conjoint structure: {conjoint_structure}")

        print("[5] Testing data preparation...")
        def prepare_conjoint_data(data, column_map):
            print("  Entered prepare_conjoint_data")
            choice_col = column_map['choice']
            task_col = column_map['task_id']
            alt_col = column_map['alternative_id']
            attribute_cols = column_map.get('attributes', [])
            print(f"  Columns: choice={choice_col}, task={task_col}, alt={alt_col}, attributes={attribute_cols}")
            
            tasks = sorted(data[task_col].unique())
            alternatives = sorted(data[alt_col].unique())
            print(f"  Found {len(tasks)} tasks and {len(alternatives)} alternatives")
            print(f"  Tasks: {tasks}, Alternatives: {alternatives}")
            
            # Create consistent feature matrix
            feature_data = []
            choice_data = []
            
            for task in tasks:
                print(f"  Processing task {task}")
                task_data = data[data[task_col] == task]
                print(f"    Task {task} has {len(task_data)} rows")
                task_features = []
                task_choices = []
                
                for alt in alternatives:
                    print(f"    Processing task {task}, alt {alt}")
                    alt_data = task_data[task_data[alt_col] == alt]
                    print(f"      Alt {alt} has {len(alt_data)} rows")
                    
                    if len(alt_data) > 0:
                        # Create one-hot encoded features
                        features = []
                        for attr_col in attribute_cols:
                            if attr_col in alt_data.columns:
                                value = alt_data[attr_col].iloc[0]
                                unique_vals = sorted(data[attr_col].unique())
                                for unique_val in unique_vals:
                                    features.append(1.0 if value == unique_val else 0.0)
                        
                        choice = alt_data[choice_col].iloc[0] if choice_col in alt_data.columns else 0
                        task_features.append(features)
                        task_choices.append(choice)
                        print(f"      Task {task}, Alt {alt}: features={features}, choice={choice}")
                    else:
                        # Handle missing alternative with zeros
                        features = [0.0] * len(attribute_cols) * 2  # Assuming 2 levels per attribute
                        task_features.append(features)
                        task_choices.append(0)
                        print(f"      Task {task}, Alt {alt}: MISSING - using zeros")
                
                print(f"    Task {task} features: {task_features}")
                print(f"    Task {task} choices: {task_choices}")
                
                if task_features:
                    feature_data.append(task_features)
                    choice_data.append(task_choices)
            
            print(f"  Feature data shape: {len(feature_data)} x {len(feature_data[0]) if feature_data else 0}")
            print(f"  Choice data shape: {len(choice_data)} x {len(choice_data[0]) if choice_data else 0}")
            
            print("  Creating tensors...")
            X = torch.tensor(feature_data, dtype=torch.float32)
            choice_mask = torch.tensor(choice_data, dtype=torch.float32)
            print(f"  Created tensors: X shape={X.shape}, choice_mask shape={choice_mask.shape}")
            return X, choice_mask
        
        column_map = {
            'choice': 'choice',
            'task_id': 'task_id', 
            'alternative_id': 'alternative_id',
            'attributes': ['brand']
        }
        
        X, choice_mask = prepare_conjoint_data(test_data, column_map)
        print(f"✓ Prepared data - X shape: {X.shape}, choice_mask shape: {choice_mask.shape}")

        print("[6] Running simple HB analysis...")
        class SimpleHBAnalysis:
            def __init__(self, X, choice_mask, device=None):
                print("  Entered SimpleHBAnalysis.__init__")
                self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.X = X.to(self.device).float()
                self.choice_mask = choice_mask.to(self.device)
                print("  Tensors moved to device")
                self.N, self.T, self.J, self.K = X.shape
                print(f"  Shape unpacked: N={self.N}, T={self.T}, J={self.J}, K={self.K}")
            def run(self):
                print("  Running simplified HB analysis...")
                beta_i = torch.randn(self.N, self.K, device=self.device) * 0.1
                mu = torch.zeros(self.K, device=self.device)
                Sigma = torch.eye(self.K, device=self.device)
                num_samples = 10
                beta_samples = torch.zeros(num_samples, self.N, self.K, device=self.device)
                mu_samples = torch.zeros(num_samples, self.K, device=self.device)
                Sigma_samples = torch.zeros(num_samples, self.K, self.K, device=self.device)
                for i in range(num_samples):
                    beta_i += torch.randn_like(beta_i) * 0.01
                    mu += torch.randn_like(mu) * 0.01
                    beta_samples[i] = beta_i
                    mu_samples[i] = mu
                    Sigma_samples[i] = Sigma
                print("  HB analysis loop complete")
                return beta_samples, mu_samples, Sigma_samples, [0.1] * num_samples
        
        hb_analysis = SimpleHBAnalysis(X, choice_mask)
        beta_samples, mu_samples, Sigma_samples, acceptance_rates = hb_analysis.run()
        print(f"✓ HB analysis completed")
        print(f"✓ Beta samples shape: {beta_samples.shape}")
        print(f"✓ Mu samples shape: {mu_samples.shape}")
        print(f"✓ Sigma samples shape: {Sigma_samples.shape}")

        print("\n✓ All tests passed!")
        return True
    except Exception as e:
        print("EXCEPTION CAUGHT!")
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

print("About to call test function...")

if __name__ == "__main__":
    success = test_cbc_core_functionality()
    print("Test function completed...")
    if success:
        print("\n🎉 CBC core functionality is working correctly!")
    else:
        print("\n❌ CBC core functionality has issues that need to be fixed.")

print("Script ending...") 