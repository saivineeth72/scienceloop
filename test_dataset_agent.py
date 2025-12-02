#!/usr/bin/env python3
"""
Test script to call DatasetAgent API directly
"""

import json
import requests
import sys

# Simulation plan from the user
simulation_plan = {
  "constants_required": [
    {
      "description": "Length of peptide sequences (fixed for each classification problem)",
      "name": "M",
      "value_or_range": "9 for HIV-protease, 9 for T-cell epitopes, variable for HLA binding (typically 8-11)"
    },
    {
      "description": "Range of lag values for correlation factors",
      "name": "k_range",
      "value_or_range": "[1, 19] as indicated by Pp_20+k notation"
    },
    {
      "description": "Total number of available physicochemical properties for amino acids",
      "name": "total_physicochemical_properties",
      "value_or_range": "531 properties available in AAindex database (as commonly used in peptide encoding)"
    },
    {
      "description": "Number of cross-validation folds for robust accuracy estimation",
      "name": "cv_folds",
      "value_or_range": "5 or 10"
    },
    {
      "description": "Kernel type for SVM classifier",
      "name": "svm_kernel",
      "value_or_range": "rbf (radial basis function) as commonly used in peptide classification"
    },
    {
      "description": "Regularization parameter for SVM",
      "name": "svm_C",
      "value_or_range": "[0.1, 1, 10, 100] - to be optimized via grid search"
    },
    {
      "description": "Kernel coefficient for SVM",
      "name": "svm_gamma",
      "value_or_range": "[0.001, 0.01, 0.1, 1] - to be optimized via grid search"
    }
  ],
  "expected_outcomes": "The simulation should reveal: (1) An initial steep increase in classification accuracy as num_properties increases from 1 to approximately 20-50 properties, demonstrating that multiple physicochemical properties capture complementary information about peptide function. (2) A plateau region where accuracy reaches maximum performance (optimal threshold), typically between 50-150 properties depending on the classification problem and dataset size. (3) A potential slight decrease or stagnation in test accuracy beyond the optimal threshold (e.g., >200 properties) while training accuracy continues to increase, indicating overfitting due to feature redundancy and curse of dimensionality. (4) Increased variance in accuracy for very high num_properties values due to random noise in less informative properties. (5) Consistency of the optimal threshold pattern across all three classification problems (HIV-protease, T-cell epitopes, HLA binding), though the exact optimal value may differ. (6) Higher correlation among properties as num_properties increases, confirming feature redundancy. (7) Diminishing returns in accuracy improvement per additional property beyond the optimal threshold, validating the hypothesis that additional properties do not significantly enhance performance. (8) The property selection method (especially correlation-based selection) should achieve optimal performance with fewer properties compared to random selection. These outcomes will quantitatively validate the hypothesis and identify the practical optimal range for physicochemical property selection in peptide encoding.",
  "procedure_steps": [
    "Step 1: Initialize simulation environment - Load or generate synthetic peptide datasets for three classification problems (HIV-protease, T-cell epitopes, HLA binding) with binary labels (positive/negative). Load physicochemical property database (AAindex or equivalent) containing normalized values val(p, A) for each property p and amino acid A.",
    "Step 2: Implement correlation factor computation - For each peptide sequence of length M and each physicochemical property p, compute correlation factors Pp_20+k for k in [1, 19] using the formula: Pp_20+k = (1/(M-k)) * sum_{a=1}^{M-k} [val(p, A_a) - val(p, A_{a+k})]. Store results in a feature matrix where each row represents a peptide and columns represent correlation factors.",
    "Step 3: Vary number of properties systematically - For each value in num_properties range [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 531]: (a) Select the specified number of physicochemical properties using the chosen selection method (random, top variance, correlation with labels, or sequential), (b) Compute correlation factors for selected properties only, (c) Concatenate all correlation factors to form the final encoding vector for each peptide (dimension = num_properties × 19).",
    "Step 4: Train and evaluate SVM classifier - For each encoding configuration: (a) Split dataset into training and testing sets or use k-fold cross-validation, (b) Optimize SVM hyperparameters (C, gamma) using grid search on training data, (c) Train SVM classifier with optimized parameters, (d) Evaluate classification accuracy on test set, (e) Record accuracy, precision, recall, F1-score, and training time.",
    "Step 5: Repeat for multiple property selection strategies - For each num_properties value, repeat Steps 3-4 using different property selection methods (random with 10 repetitions for statistical robustness, top variance, top correlation with labels). Compute mean and standard deviation of accuracy across repetitions.",
    "Step 6: Test across all classification problems - Repeat Steps 2-5 for each of the three classification problems (HIV-protease, T-cell epitopes, HLA binding) to verify consistency of findings across different peptide classification tasks.",
    "Step 7: Analyze feature redundancy and overfitting - For each num_properties value: (a) Compute correlation matrix among selected properties to quantify redundancy, (b) Calculate training accuracy vs. test accuracy gap to detect overfitting, (c) Perform feature importance analysis using SVM weights or permutation importance.",
    "Step 8: Generate comprehensive visualizations - Create plots: (a) Classification accuracy vs. number of properties (main hypothesis test), (b) Training vs. test accuracy curves to identify overfitting threshold, (c) Accuracy comparison across three classification problems, (d) Heatmap of property correlation matrix at different num_properties values, (e) Computational time vs. number of properties, (f) Precision-recall curves for optimal num_properties value.",
    "Step 9: Statistical analysis - Perform statistical tests (paired t-tests or Wilcoxon signed-rank tests) to determine: (a) At which num_properties value accuracy improvement becomes statistically insignificant, (b) Whether there is a significant performance drop after the optimal threshold due to overfitting, (c) Confidence intervals for optimal num_properties range.",
    "Step 10: Validate optimal threshold - Identify the optimal number of properties where: (a) Test accuracy is maximized, (b) Training-test accuracy gap is minimal (low overfitting), (c) Computational cost is reasonable. Validate this threshold by testing on held-out validation sets or through nested cross-validation."
  ],
  "simulation_equations": [
    "Pp_20+k = (1/(M-k)) * sum_{a=1}^{M-k} [val(p, A_a) - val(p, A_{a+k})]",
    "Feature_Matrix = [Pp_20+1, Pp_20+2, ..., Pp_20+19] for each property p in selected_properties",
    "Encoded_Peptide = concatenate([Feature_Matrix_p1, Feature_Matrix_p2, ..., Feature_Matrix_pN])",
    "Classification_Accuracy = (True_Positives + True_Negatives) / Total_Samples",
    "Cross_Validation_Score = mean(accuracy_scores_across_folds)"
  ],
  "variables_to_vary": [
    {
      "description": "Number of physicochemical properties (p) used in encoding",
      "name": "num_properties",
      "range": "[1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 531]",
      "units": "count"
    },
    {
      "description": "Method for selecting which properties to use",
      "name": "property_selection_method",
      "range": "['random', 'top_variance', 'top_correlation_with_label', 'sequential']",
      "units": "categorical"
    },
    {
      "description": "Which peptide classification problem to test",
      "name": "classification_problem",
      "range": "['HIV-protease', 'T-cell epitopes', 'HLA binding']",
      "units": "categorical"
    }
  ]
}

def test_dataset_agent_api():
    """Test the DatasetAgent API endpoint"""
    api_url = "http://localhost:5001/api/generate-datasets"
    
    print("=" * 80)
    print("TESTING DATASET AGENT API")
    print("=" * 80)
    print(f"\nAPI URL: {api_url}")
    print(f"\nSimulation Plan Keys: {list(simulation_plan.keys())}")
    print(f"Constants Required: {len(simulation_plan['constants_required'])}")
    print(f"Variables to Vary: {len(simulation_plan['variables_to_vary'])}")
    print(f"Procedure Steps: {len(simulation_plan['procedure_steps'])}")
    print(f"Simulation Equations: {len(simulation_plan['simulation_equations'])}")
    
    print("\n" + "=" * 80)
    print("SENDING REQUEST...")
    print("=" * 80)
    
    try:
        response = requests.post(
            api_url,
            json={
                "simulation_plan": simulation_plan
            },
            headers={
                "Content-Type": "application/json"
            },
            timeout=120  # 2 minute timeout
        )
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 80)
            print("SUCCESS - API RESPONSE:")
            print("=" * 80)
            print(json.dumps(result, indent=2))
            
            if 'result' in result:
                dataset_result = result['result']
                print("\n" + "=" * 80)
                print("DATASET RESULT SUMMARY:")
                print("=" * 80)
                print(f"Dataset Type: {dataset_result.get('dataset_type', 'N/A')}")
                print(f"Number of Datasets: {len(dataset_result.get('datasets', {}))}")
                print(f"Reasoning: {dataset_result.get('reasoning', 'N/A')}")
                
                if dataset_result.get('datasets'):
                    print("\nDatasets Generated:")
                    for name, path in dataset_result['datasets'].items():
                        import os
                        exists = os.path.exists(path)
                        size = os.path.getsize(path) if exists else 0
                        status = "✓ EXISTS" if exists else "✗ MISSING"
                        print(f"  - {name}: {path}")
                        print(f"    Status: {status} ({size} bytes)")
                else:
                    print("\nNo datasets generated!")
                    print("\nNOTE: Check Flask server logs for [DatasetAgent] DEBUG messages")
                    print("      to see why datasets weren't generated.")
        else:
            print("\n" + "=" * 80)
            print("ERROR - API RESPONSE:")
            print("=" * 80)
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2))
            except:
                print(f"Response Text: {response.text}")
                
    except requests.exceptions.Timeout:
        print("\nERROR: Request timed out after 120 seconds")
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API. Make sure the Flask server is running on port 5001")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_agent_api()

