# Experiment 1.1 v0
## Main objective:
 - Run the complete pipeline based on the Efficient-CapsNet model for the first time
## Model test performance metrics:
 - AUC 0.61±0.07
 - Precision 0.36±0.15
 - Specificity 0.85±0.05
 - Recall 0.38±0.11
## Main issues:
 - Model overfitting
 - Training loss curve converging at a relatively high loss value
# Experiment 1.1 v1
## Objective:
 - Repeat experiment 1.1 v0 reducing the learning rate from 5e-4 to 5e-5 to address model overfitting issue
## Model test performance metrics:
 - AUC 0.63±0.06
 - Precision 0.37±0.15
 - Specificity 0.85±0.06
 - Recall 0.39±0.11
## Main issues:
 - Model overfitting
 - Training loss curve converging at a relatively high loss value
# Experiment 1.1 v2
## Objective:
 - Repeat experiment 1.1 v1 reducing the learning rate from 5e-5 to 5e-6 to address model overfitting issue
## Model test performance metrics:
 - AUC 0.65±0.05
 - Precision 0.37±0.13
 - Specificity 0.84±0.06
 - Recall 0.4±0.14
## Main issues:
 - Model underfitting
# Experiment 1.1 v3
## Objective:
 - Repeat experiment 1.1 v2, increasing the learning rate from 5e-6 to 1e-5 to address model underfitting issue
## Model test performance metrics:
 - AUC 0.68±0.05
 - Precision 0.38±0.13
 - Specificity 0.85±0.06
 - Recall 0.41±0.14
## Main issues:
 - Model underfitting
# Experiment 1.1 v4
## Objective:
 - Repeat experiment 1.1 v3, increasing the batch size from 16 to 32 to address model underfitting issue
## Model test performance metrics:
 - AUC 0.74±0.08
 - Precision 0.39±0.12
 - Specificity 0.85±0.05
 - Recall 0.46±0.19
## Main issues:
 - Model underfitting
# Experiment 1.1 v5
## Objective:
 - Repeat experiment 1.1 v4, increasing the learning rate from 1e-5 to 5e-5 to address model underfitting issue
## Model test performance metrics:
 - AUC 0.73±0.08
 - Precision 0.4±0.15
 - Specificity 0.85±0.06
 - Recall 0.46±0.16
## Main issues:
 - Model overfitting
# Experiment 1.1 v6
## Objective:
 - Repeat experiment 1.1 v5, decreasing the learning rate from 5e-5 to 2.5e-5 to address model overfitting issue
## Model test performance metrics:
 - AUC 0.76±0.08
 - Precision 0.42±0.14
 - Specificity 0.86±0.06
 - Recall 0.49±0.17
## Main issues:
 - Model underfitting
# Experiment 1.1 v7
## Objective:
 - Repeat experiment 1.1 v6, increasing the batch size from 32 to 64 to address model underfitting issue
## Model test performance metrics:
 - AUC 0.79±0.1
 - Precision 0.41±0.12
 - Specificity 0.85±0.05
 - Recall 0.51±0.21
## Main issues:
 - Model underfitting
# Experiment 1.1 v8
## Objective:
 - Repeat experiment 1.1 v7, increasing the number of epochs from 100 to 200 to address model underfitting issue
## Model test performance metrics:
 - AUC 0.79±0.1
 - Precision 0.41±0.13
 - Specificity 0.85±0.06
 - Recall 0.51±0.22
## Main issues:
 - Model underfitting
## Pivotal findings from experiment 1.1, version 1-8:
- Apparently, any attempt to address model underfitting by changing the learning rate, batch size and number of epochs will cause model overfitting
 - All training loss curves converged at relatively high loss values
 - Therefore, something different needs to be changed to result in a well-adjusted model
# Experiment 1.1 v9
## Objective:
 - Repeat experiment 1.1 v8, adding data augmentation
## Model test performance metrics:
 - AUC 0.77±0.1
 - Precision 0.39±0.12
 - Specificity 0.84±0.09
 - Recall 0.48±0.27
## Main issues:
 - Model underfitting
# Experiment 1.1 v10
## Objective:
 - Repeat experiment 1.1 v7, increasing the number of low-level capsules from 16 to 32 to address model underfitting issue
## Model test performance metrics:
 - AUC 0.79±0.1
 - Precision 0.41±0.11
 - Specificity 0.85±0.06
 - Recall 0.51±0.21
## Main issues:
 - Model underfitting
# Experiment 1.1 v11
## Objective:
 - Repeat experiment 1.1 v7, increasing the high-level capsule dimension from 16 to 32 to address model underfitting issue
## Model test performance metrics:
 - AUC 0.78±0.1
 - Precision 0.41±0.13
 - Specificity 0.85±0.06
 - Recall 0.51±0.22
## Main issues:
 - Model underfitting
---
# Experiment 2.1 
This experiment aims to propose the **Efficient-X-Caps**, which is the Efficient-CapsNet model [1] but with classification and loss functions 
of the X-Caps model [2], so that the model can classify not only the average nodule malignancy score, but also the visual attributes  of nodules. 
The baseline model test performance metrics are:

| Subtlety | Sphericity | Margin | Lobulation | Spiculation | Texture | Malignancy | 
|:--------:|:----------:|:------:|:----------:|:-----------:|:-------:|:----------:|
|  90.39   |   85.44    | 84.14  |   70.69    |    75.23    |  93.10  |   86.39    |

### References:
 - [1]: Efficient-CapsNet: Capsule Network with Self-Attention Routing
 - [2]: Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses

## Experiment 2.1 v1-64 (Grid search for hyperparameter optimization)
### Objective:
 - Run the complete pipeline based on the Efficient-X-CapsNet model for the first time
### Best version: 31
### Model test performance metrics:

|                  | Subtlety | Internal structure | Calcification | Sphericity | Margin | Lobulation | Spiculation | Texture | Malignancy | 
|:----------------:|:--------:|:------------------:|:-------------:|:----------:|:------:|:----------:|:-----------:|:-------:|:----------:|
|     Baseline     |  90.39   |         -          |       -       |   85.44    | 84.14  |   70.69    |    75.23    |  93.10  |   86.39    |
| Efficient-X-Caps |  91 ± 1  |      100 ± 1       |    93 ± 1     |   90 ± 2   | 88 ± 2 |   89 ± 1   |   88 ± 1    | 90 ± 2  |   83 ± 2   |

### Main issues:
 - Model overfitting
 - Malignancy accuracy bellow baseline

## Experiment 2.1 v65-72 (Grid search for hyperparameter optimization)
### Objective:
 - Run the complete pipeline based on the Efficient-X-CapsNet model for the first time
### Model test performance metrics:
Best version: 71

|                  | Subtlety | Internal structure | Calcification | Sphericity | Margin | Lobulation | Spiculation | Texture | Malignancy | 
|:----------------:|:--------:|:------------------:|:-------------:|:----------:|:------:|:----------:|:-----------:|:-------:|:----------:|
|     Baseline     |  90.39   |         -          |       -       |   85.44    | 84.14  |   70.69    |    75.23    |  93.10  |   86.39    |
| Efficient-X-Caps |  91 ± 1  |      100 ± 1       |    94 ± 2     |   90 ± 2   | 87 ± 2 |   89 ± 0   |   87 ± 1    | 90 ± 2  |   85 ± 2   |

### Main issues:
 - Model overfitting
 - Malignancy accuracy bellow baseline

## Experiment 3.0 v1-9 (Grid search for hyperparameter optimization)
### Objective:
 - Run the complete pipeline based on the Efficient-Proto-CapsNet model for the first time
### Model test performance metrics:
Best version: 71

|                      | Subtlety  | Internal structure | Calcification | Sphericity | Margin | Lobulation | Spiculation | Texture | Malignancy | 
|:--------------------:|:---------:|:------------------:|:-------------:|:----------:|:------:|:----------:|:-----------:|:-------:|:----------:|
|       Baseline       |   90.39   |         -          |       -       |   85.44    | 84.14  |   70.69    |    75.23    |  93.10  |   86.39    |
|   Efficient-X-Caps   |  91 ± 1   |      100 ± 1       |    94 ± 2     |   90 ± 2   | 87 ± 2 |   89 ± 0   |   87 ± 1    | 90 ± 2  |   85 ± 2   |
| Efficient-Proto-Caps | 90 ± 0.02 |      100 ± 1       |    96 ± 2     |   91 ± 2   | 87 ± 2 |   91 ± 0   |   90 ± 1    | 90 ± 2  |   80 ± 2   |

### Main issues:
 - Model overfitting
 - Malignancy accuracy bellow baseline