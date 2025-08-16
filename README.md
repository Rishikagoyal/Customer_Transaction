# Customer_Transaction

## Summary
This project aimed to develop a machine learning model that predicts
 whether a customer will make a transaction in the future. The dataset
 provided was anonymized, consisting of 200 numeric features (var_0 to
 var_199), along with ID_code and a binary target column where 1
 indicates a future transaction. Traditional Exploratory Data Analysis
 (EDA) was skipped as per the project instructions, due to the
 anonymized nature of the features.
 We applied data preprocessing, handled class imbalance, scaled the
 features, and trained multiple classification models including Logistic
 Regression, Random Forest, and XGBoost. After comparing
 performance using accuracy, confusion matrix, and AUC-ROC,
 XGBoost emerged as the most accurate and robust model. The final
 model is suitable for deployment in customer behavior prediction
 pipelines to help financial institutions optimize outreach and marketing
 strategies.

## Tools Used

1. Python:Core language for implementation and analysis
2. Pandas & NumPy:Data preprocessing and numerical operations
3.  Matplotlib & Seaborn: Basic visualization for understanding class
 imbalance.
4.  Scikit-learn: Model training, evaluation, cross-validation, and
 preprocessing
5.  XGBoost: High-performance gradient boosting algorithm used for
 classification

## Challenges Faced

1. Missing Values:
 Some features had missing or constant values which could affect model
 training. These were handled using removal of constant features and
 ensuring clean data through .dropna() or imputation strategies
2.  Imbalanced Target Variable
 The dataset was significantly imbalanced with far fewer transaction
positive (target=1) samples. This was addressed using class weights in
 models and ROC-AUC as a key performance metric instead of accuracy.
3. High Dimensionality
 With 200 features, many were likely irrelevant or redundant. We used
 model-based feature importance (from Random Forest and XGBoost) to
 focus on top-performing features.
4. No Feature Names
 Due to anonymization, we could not apply business logic or traditional
 visual EDA. Instead, we focused on correlation matrices and distribution
 visualizations for numeric insight.
5. Overfitting in Complex Models
 Random Forest showed signs of overfitting. We implemented cross
validation and parameter tuning to improve generalization.
6.  Model Selection and Evaluation
 It was essential to compare models based on metrics like ROC-AUC
 rather than raw accuracy due to class imbalance. XGBoost gave the best
 results in terms of generalization and score.

## Conclusion
 The Customer Transaction Prediction project successfully produced a
 high-performing classification model utilizing anonymised customer
 data. The team managed issues like overfitting, high dimensionality,
 and class imbalance despite having little feature context. XGBoost was
 chosen as the final model after a number of models were evaluated
 because of its excellent AUC-ROC score and high accuracy. This
 approach can be used to enhance campaign ROI and facilitate
 predictive customer targeting in banking and e-commerce systems.
