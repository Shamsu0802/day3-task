## Step 1: Dataset Inspection

- Loaded the cleaned customer churn dataset into Google Colab.
- Verified the dataset contains 420 rows and 15 columns.
- Checked the column names and data types.
- Confirmed there are no missing values in any column.
- Identified `has_churned` as the target variable.
- Observed both numerical and categorical features, which will require preprocessing before model training.

## Step 2: Feature Selection and Train-Test Split
- Selected has_churned as the target variable for prediction.
- Removed customer_id because it is a unique identifier and does not contribute to predicting customer churn.
- Removed signup_date because it is a raw date field and the dataset already includes tenure_months and tenure_years, which capture the customer's duration with the company.
- Separated the remaining columns as input features (X) and the target column as the output (y).
- Split the dataset into 80% training and 20% testing sets.
- Used stratified sampling to preserve the original class distribution of the target variable in both training and testing datasets.
- Set random_state=42 to ensure the train-test split is reproducible across different runs.

## Step 3: Data Preprocessing Pipeline
- Identified numerical and categorical features from the input dataset.
- Applied StandardScaler to numerical features to normalize their values for models that are sensitive to feature scales.
- Applied OneHotEncoder to categorical features to convert categorical values into numerical format.
- Combined both preprocessing steps using a ColumnTransformer.
- Configured the preprocessing pipeline to be fitted only on the training data during model training, preventing data leakage and ensuring a fair evaluation.

## Step 4: Model Training
- Trained two machine learning models to predict customer churn:
- Logistic Regression
- Random Forest Classifier
- Integrated the preprocessing pipeline with each model using Scikit-learn's Pipeline.
- Ensured preprocessing was fitted only on the training data to prevent data leakage.
- Used random_state=42 to make model training reproducible.

## Step 5: Model Evaluation and Overfitting Analysis
- Evaluated Logistic Regression and Random Forest models using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- Logistic Regression achieved the best overall performance with an accuracy of 76.19% and the highest ROC-AUC score.
- Compared training and testing accuracy for both models to identify overfitting.
- Logistic Regression achieved 76.19% accuracy on both the training and testing datasets, indicating good generalization and no significant overfitting.
- Random Forest achieved 100% training accuracy but only 75% testing accuracy, indicating overfitting.
- To reduce overfitting in the Random Forest model, techniques such as limiting tree depth, increasing minimum samples for splits/leaves, using cross-validation, and  collecting more data can be considered.


## Overfitting Analysis

To identify overfitting, the training and testing accuracy of both models were compared.

### Logistic Regression

- Training Accuracy: **76.19%**
- Testing Accuracy: **76.19%**

The training and testing accuracies are identical, indicating that the Logistic Regression model generalizes well to unseen data and does not exhibit significant overfitting.

### Random Forest (Before Tuning)

The initial Random Forest model was trained using the default parameters:

- max_depth = None (no depth limit)
- min_samples_split = 2
- min_samples_leaf = 1
- n_estimators = 100

Results:

- Training Accuracy: **100.00%**
- Testing Accuracy: **75.00%**

Since the training accuracy was perfect while the testing accuracy was considerably lower, the model was overfitting the training data. This occurred because the trees were allowed to grow without any restriction, causing the model to memorize the training data rather than learn generalized patterns.

### Random Forest (After Tuning)

To reduce overfitting, the following hyperparameters were modified:

- max_depth = 5
- min_samples_split = 10
- min_samples_leaf = 5
- n_estimators = 100

These changes reduced the complexity of the trees by limiting their depth and requiring more samples before creating new splits and leaf nodes. This helps the model generalize better to unseen data and reduces the gap between training and testing performance.

After retraining the model with these parameters, the new training and testing accuracies were:

- Training Accuracy: **84.52%**
- Testing Accuracy: **77.38%**

The tuned model showed improved generalization compared to the original Random Forest model.

## Cross-Validation

To further evaluate the robustness of the selected model, 5-fold Stratified Cross-Validation was performed on the Logistic Regression model. Stratified sampling was used to preserve the class distribution in each fold.

The cross-validation results were:

- Fold 1 Accuracy: **71.43%**
- Fold 2 Accuracy: **79.76%**
- Fold 3 Accuracy: **69.05%**
- Fold 4 Accuracy: **72.62%**
- Fold 5 Accuracy: **80.95%**

The average cross-validation accuracy was **74.76%**, with a standard deviation of **4.73%**.

The average cross-validation accuracy is reasonably close to the test accuracy obtained from the train-test split (**76.19%**). This indicates that the Logistic Regression model performs consistently across different subsets of the dataset and generalizes well to unseen data. The relatively low standard deviation also suggests that the model's performance is stable and not heavily dependent on a particular train-test split.
