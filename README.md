# Credit-Card-Fraud-Detection(DATASET: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud )
Credit Card Fraud Detection (Imbalanced Dataset Handling)

📌 Objective:
 -> Detect fraudulent transactions in an extremely imbalanced dataset (Fraud: 0.17%).

⚙️ Preprocessing:
 -> Scaled Amount and Time using RobustScaler to reduce outlier impact.
 
 -> Dropped original Amount and Time columns after scaling.

🧪 Stratified Split:
 -> Used StratifiedKFold (5-fold) to maintain fraud/non-fraud class ratios in each split.

 -> Converted train/test splits to NumPy arrays and verified label distribution.

🔁 Balancing the Dataset:

  -> Undersampled the majority class to match the 492 fraud samples:
 -> python
   - fraud_df = df[df['Class'] == 1]
   - non_fraud_df = df[df['Class'] == 0][:492]
   - new_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

✅ Classification:

 -> Applied multiple classifiers on the balanced dataset:

  - Logistic Regression: 95% accuracy
  
  - KNN: 94%
  
  - SVM (SVC): 94%
  
  - Decision Tree: 90%

📊 Outcome:

 -> Successfully handled extreme imbalance using undersampling and achieved high accuracy on multiple classifiers.

OUTPUT:

![Image](https://github.com/user-attachments/assets/7afd2153-6b50-4bd2-8db1-9f02d3d2ebc6)

