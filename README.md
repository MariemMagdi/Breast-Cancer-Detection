## Breast Cancer Detection Using Naive Bayes Classification

### Project Overview
Breast cancer is the most common cancer among women worldwide, constituting 25% of all cancer cases. Early detection is crucial for effective treatment and management. This project utilizes a Naive Bayes Classification algorithm to predict whether breast cancer is benign or malignant based on various diagnostic measures. 

### Dataset
- **Name**: Breast Cancer Dataset
- **Source**: Kaggle
- **Purpose**: To predict the type of breast cancer (benign or malignant) using the given features.
- **Description**: Features include characteristics of the cell nuclei present in the digitized image of a fine needle aspirate (FNA) of a breast mass.
- **Size**: 569 samples, 32 features

### Installation

To run this project, install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Usage

1. Clone the repository and navigate to the project directory.
2. Load the dataset using the following Python code:
```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/mouradmagdy/stats_dataset/main/breast-cancer.csv')
```
3. Run the Jupyter Notebook to follow the data analysis and model training steps.

### Data Preprocessing

Data cleaning steps included:
- Checking for and handling missing values
- Removing duplicate entries
- Encoding categorical variables
- Removing outliers
- Normalizing and standardizing numerical features

### Feature Analysis

- **Feature Selection**: Irrelevant features like 'id' were dropped.
- **Outlier Removal**: Outliers were identified and removed to ensure model accuracy.
- **Standardization**: Features were standardized to have zero mean and unit variance.

### Model Training

The dataset was split into training and validation sets, with 80% of the data used for training and the remaining 20% for validation. The Naive Bayes model was trained on the training set.

### Model Evaluation

The model's performance was evaluated on the validation set. Various metrics such as accuracy, precision, recall, and F1-score were calculated to assess the effectiveness of the model in predicting the type of breast cancer.

### Results

The results are documented in detail in the Jupyter Notebook. Plots and statistical tests are used to interpret the model's performance and the influence of different features on the prediction.


