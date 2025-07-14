**DNA Promoter Sequence Classification**

This project focuses on classifying DNA sequences as promoter or non-promoter using various machine learning algorithms. A promoter is a DNA segment essential in initiating the transcription of a gene. We use a dataset from the UCI Machine Learning Repository for this classification task.

** Dataset**

Source: UCI ML Repository
Samples: 106 DNA sequences
Features:
Class: + for promoter, - for non-promoter
id: Sequence identifier
Sequence: 57 nucleotide sequence (A, T, G, C)

** Objective**

To classify a DNA sequence into promoter (+) or non-promoter (-) using:

K-Nearest Neighbors
Gaussian Process
Decision Tree
Random Forest
Neural Network
AdaBoost
Naive Bayes
SVM (Linear, RBF, Sigmoid kernels)

** Technologies Used**

Python 3
Pandas, NumPy, Seaborn, Matplotlib
Scikit-learn

**Data Processing**

Load raw data and extract nucleotide sequences.
Clean sequences (remove tab characters).
Split sequences into individual nucleotides.
Append the target class label.
Convert the dataset into a structured pandas DataFrame.
One-hot encode nucleotides to make the dataset machine learning ready.
**
Exploratory Data Analysis**

Distribution of classes (promoter vs. non-promoter).
Count plots of nucleotide frequencies.
Tabular view of nucleotide distributions across samples.

**Model Training**

Feature Matrix (X): Nucleotide positions as features (after one-hot encoding).
Target Vector (y): Promoter or Non-Promoter class.
Split: 75% training, 25% testing
Evaluation Metrics:
Accuracy Score
Classification Report (Precision, Recall, F1-Score)
10-fold Cross-validation

**Results
**
Each algorithm is trained and evaluated using cross-validation on the training set and tested on the hold-out set. Below is an example result:

Nearest Neighbors: 0.850000 (0.078927)
Gaussian Process:  0.830000 (0.092411)
Decision Tree:     0.800000 (0.120000)
...
SVM Sigmoid:       0.720000 (0.101234)


** How to Run**

1. Clone the repo or download the script.
2. Install dependencies:
pip install pandas numpy seaborn matplotlib scikit-learn

3. Run the script:
python dna_classifier.py

**Conclusion **
The dataset is relatively small (106 samples), so results may not generalize well.
Further improvements can include deep learning models or augmenting the dataset with synthetic sequences.
