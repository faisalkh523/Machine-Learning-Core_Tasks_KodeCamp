# Medical Imaging ML & Deep Learning
Applied machine learning and deep learning projects spanning medical diagnostic classification, image recognition, and core algorithm implementation completed independently as part of the KodeCamp Machine Learning Core Certification, alongside my B.Eng in Computer Engineering.

The centerpiece is a neural network for breast cancer diagnosis, alongside CNN-based image classification (MNIST, CIFAR-10) and first-principles implementations of the core ML/DL building blocks, gradient descent, cost functions, and activation functions, built from scratch rather than called from a library.

Each notebook is self-contained, documented step-by-step, and runnable directly in Google Colab.

## Highlights
- 98.25% validation accuracy classifying breast cancer tumors (benign vs. malignant) with a 3-layer neural network
- 97.30% test accuracy on MNIST digit classification using a LeNet-5 CNN
- 73.88% test accuracy on CIFAR-10 image classification with a custom 3-block convolutional network
- Core algorithms: gradient descent, cost functions, and activation functions implemented from first principles in NumPy, not just called from a library

## Notebooks
| Notebook | Task | Result |
|----------|------|--------|
| `Breast_Cancer_Dataset_Classifier_Task.ipynb` | Binary classification of breast masses (benign/malignant) from the Wisconsin Diagnostic dataset | 98.25% validation accuracy |
| `CNN_training_MNIST.ipynb` | Handwritten digit classification (0–9) on the MNIST dataset | 97.30% test accuracy |
| `CIFAR_10_Image_Classification_Task_6.ipynb` | Multi-class image classification across 10 object categories (CIFAR-10) | 73.88% test accuracy |
| `Logistic_Regression_Task.ipynb` | Multivariable logistic regression for binary classification | Converging cross-entropy loss with decision-boundary visualization |
| `Multivariable_Linear_Regression_Task.ipynb` | Income prediction from age and years of experience | Converging MSE loss curve |
| `Linear_Regression_with_one_variable_Task.ipynb` | Single-variable linear regression | Converging cost function | 
| `Feature_Engineering_Task_.ipynb` | Data cleaning and feature engineering on a real-world "dirty" dataset | Cleaned, model-ready dataset |
| `Activation_Functions.ipynb` | Implementing activation functions and their gradients from scratch | Comparative loss analysis across activation functions |

## Tools/Libraries
`Python` | `PyTorch` | `NumPy` | `Scikit-learn` | `Pandas` | `Matplotlib` | `Kagglehub` 

## Skills Demonstrated
- Classical machine learning (linear/logistic regression) built from first principles, not just library calls
- Deep learning model design and training in PyTorch, including CNNs and fully connected networks
- Applied medical classification — the breast cancer task is direct, hands-on experience using deep learning for a real diagnostic problem
- Data preprocessing: feature scaling, missing-value handling, stratified splitting
- Model evaluation: loss/accuracy tracking, confusion matrices, overfitting analysis via train/validation curves
- Regularization techniques: dropout, weight decay (L2), and early stopping
