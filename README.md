# Home_credit_default_risk-prediction

[HCDR DataSet] (https://www.kaggle.com/competitions/home-credit-default-risk/data)

# Introduction

This project aims to predict the likelihood of an applicant defaulting on a loan using various factors such as previous loan history, monthly income, previous loan applications, and other income sources. The goal is to build a model that accurately predicts how capable each applicant is of repaying a loan, so that loans can be sanctioned only to applicants who are likely to repay.

Home Credit is a Czech Republic-based company that provides loans to people with little or no credit history. The company uses various sources of information to determine an applicant's creditworthiness, including past loan history, income, and other financial information. In this project, we will use a dataset provided by Home Credit to build a model that predicts whether or not an applicant will default on a loan.

#  Modeling

* Combined available datasets from Kaggle
* Implemented feature engineering considering categorical, numerical, and aggregated features
* Experimented with Logistic Regression, decision tree approaches using Xgboost, and further attempted to improve on those models using random forest and SVM
* Implemented Visual EDA-driven feature sampling and baseline model development
* Evaluated model strength using various classification metrics such as MSE, RMSE, ROC curve, confusion matrix, and F1 score

The first step in building a model is to combine the available datasets from Kaggle into a single one. This is done by merging the datasets on a common key, such as the applicant ID.

In feature engineering, I considered categorical, numerical, and aggregated features. Categorical features are those that take on a limited number of discrete values, such as the applicant's gender or education level. Numerical features are those that take on a continuous range of values, such as the applicant's age or income. Aggregated features are those that are derived from multiple other features, such as the average income across all previous loan applications.

After feature engineering, I experimented with different models to see which one performs the best. Initially I implemented Logistic Regression, a simple yet powerful model that is commonly used for binary classification tasks. Then built decision tree approaches using Xgboost, and further attempt to improve on those models using random forest and SVM.

In order to evaluate the strength of the model, I used various classification metrics such as MSE, RMSE, ROC curve, confusion matrix, and F1 score.

# Feature Engineering

* Included new features based on domain knowledge and aggregated features using min and max as aggregators
* Merged all datasets into a single one
* Identified and removed unimportant and irrelevant features in the dataset
* Added new features such as days_employed_pct and credit_income_pct from domain knowledge and prev_amt_application_avg and prev_amt_application_range from aggregators

In the feature engineering step, I included some new features based on domain knowledge. For example, I added new features such as days_employed_pct and credit_income_pct which are derived from the applicant's employment and credit history. I also added aggregated features using min and max as aggregators. For example, I added prev_amt_application_avg and prev_amt_application_range, which are derived from the applicant's previous loan applications.

I also removed unimportant and irrelevant features in our dataset. For example, features having more than 50% missing values have been dropped.

## Neural Network

* Built a multi-layer neural network model in Pytorch and used Tensorboard to visualize real-time training results
* Monitored error generalization with an early stopping technique
* Evaluated model performance by monitoring loss functions such as CXE and Hinge Loss
* Built 2 models, the first model contains one linear layer with the Relu function for probability prediction, and the second model contains one linear layer, one hidden layer with the Relu function, and the sigmoid function for probability prediction
* Using Tensorboard I visualized the CXE loss for training data for each epoch.

In addition to traditional machine learning models, I also built a multi-layer neural network model in Pytorch. Neural networks are a powerful technique for modeling complex data and can often achieve better results than traditional models.

I used Tensorboard to visualize real-time training results, which allowed us to monitor the performance of the model during training. I also used an early stopping technique to monitor error generalization and prevent overfitting.

I evaluated the model performance by monitoring loss functions such as CXE and Hinge Loss. I built 2 models, the first model contains one linear layer with the Relu function for probability prediction, and the second model contains one linear layer, one hidden layer with the Relu function, and the sigmoid function for probability prediction.

Using Tensorboard I visualized the CXE loss for training data for each epoch, which allowed us to evaluate the performance of the model over time.

Results

XGBOOST achieved a Test AUC score of 0.7537
Multi-layer neural network model achieved AUC scores of 0.588 for train data and 0.5172 for test data
Single-layer model achieved an AUC score of 0.7558 for test data
The results of our models can be found in the table above. As we can see, XGBOOST achieved a Test AUC score of 0.7537, which is a good score. However, the multi-layer neural network model achieved AUC scores of 0.588 for train data and 0.5172 for test data, which is not as good as XGBOOST. The single-layer model achieved an AUC score of 0.7558 for test data, which is better than the multi-layer model but still not as good as XGBOOST.

Conclusion

In this project, I built a model to predict the likelihood of an applicant defaulting on a loan. I used various factors such as previous loan history, monthly income, previous loan applications, and other income sources to build our model. I used Logistic Regression, decision tree approaches using Xgboost, Random Forest and SVM to experiment and improve on the models. Additionally, wIe also built a multi-layer neural network model in Pytorch. Our results show that XGBOOST performed the best, achieving a Test AUC score of 0.7537. However, the neural network model performed poorly with AUC scores of 0.588 for train data and 0.5172 for test data.

