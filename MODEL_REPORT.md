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


