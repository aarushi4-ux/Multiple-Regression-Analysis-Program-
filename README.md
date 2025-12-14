# Credit Card Customer Transaction Amount Prediction

## Project Overview
This project applies **Linear Regression** to the *Credit Card Customers* dataset from Kaggle to predict a customer’s **Total Transaction Amount (`Total_Trans_Amt`)** based on demographic information, account tenure, credit usage, and transaction behavior. The goal of the assignment is to understand how different customer attributes influence annual spending and to evaluate the effectiveness of a linear regression model for this task.

## Dataset
- **Source:** Kaggle – Credit Card Customers Dataset  
- **File Used:** `BankChurners.csv`  
- **Description:** The dataset contains customer demographics, credit card usage, transaction statistics, and banking relationship details.

## Target Variable
- **`Total_Trans_Amt`**: Total amount spent by a customer using their credit card over the last 12 months.

## Feature Variables
The following features were selected as predictors:
- `Customer_Age`
- `Dependent_count`
- `Months_on_book`
- `Total_Relationship_Count`
- `Months_Inactive_12_mon`
- `Contacts_Count_12_mon`
- `Credit_Limit`
- `Total_Revolving_Bal`
- `Avg_Open_To_Buy`
- `Total_Amt_Chng_Q4_Q1`
- `Total_Trans_Ct`
- `Total_Ct_Chng_Q4_Q1`
- `Avg_Utilization_Ratio`

These variables represent customer demographics, engagement level, credit utilization, and transaction behavior.

## Tools and Libraries Used
- **Python**
- **Pandas** – data loading and manipulation  
- **Matplotlib & Seaborn** – data visualization  
- **Scikit-learn** – model training and evaluation  

## Methodology
1. Loaded and explored the dataset using Pandas.
2. Selected relevant predictor variables and defined the target variable.
3. Split the data into training (70%) and testing (30%) sets.
4. Trained a **Linear Regression** model on the training data.
5. Generated predictions on the test set.
6. Evaluated model performance using MAE, MSE, and RMSE.
7. Analyzed model coefficients to interpret feature importance.

## Model Evaluation Metrics
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Mean Squared Error (MSE):** Penalizes larger prediction errors.
- **Root Mean Squared Error (RMSE):** Indicates prediction error in the same units as the target variable.

These metrics help assess how accurately the model predicts customer transaction amounts.

## Key Findings
- Transaction-related variables such as **Total Transaction Count** and **Change in Transaction Amount (Q4 vs Q1)** are the strongest predictors of total spending.
- Credit utilization and inactivity negatively impact transaction amounts.
- Demographic variables have relatively smaller influence compared to behavioral features.
- The linear regression model provides interpretable results but shows limitations in handling non-linear spending patterns.

## Conclusion
This assignment demonstrates how linear regression can be used to model and interpret customer spending behavior. While the model captures meaningful relationships between features and total transaction amount, future improvements could include non-linear models or feature engineering to enhance prediction accuracy.

## How to Run the Project
1. Install required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
