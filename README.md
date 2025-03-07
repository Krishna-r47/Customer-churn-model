### Problem statement:

To accurately classify customer churn for each client by leveraging tree-based models such as random forest and Extreme Gradient Boosting (XGB).



### EDA and Data Pre-processing steps taken:

✓ Our dataset has 100,000 rows and one target variable in the form of a binary data type (0’s and 1’s).

✓ The following screenshot sums up all the available values in each column.

![image](https://github.com/user-attachments/assets/fde3816a-f634-4c10-9cbb-a7a851a65034)

![image](https://github.com/user-attachments/assets/ecacb1e3-8800-454c-ad0b-7cd7512d6eba)

![image](https://github.com/user-attachments/assets/fd390d2a-e56e-4974-8ee5-5ace1494a739)

✓ Columns that had more 30% missing values were dropped from the dataset.

✓ The distribution of churn is checked with the help of countplotto make sure classification results for target variable are balanced.

✓ Although it is not recommended to use KNN for large datasets, I wanted to personally test it out to see how KNN performs on large datasets. Missing numerical values were filled with KNN and mode was used to fill out all the categorical columns.

✓ Outliers were removed using the Interquartile range limits with a slightly more lenient range. (3 times IQR instead of the usual 1.25).

✓ Because the dataset is big, I tried different imputation methods to get the maximum out of leveraging machine learning techniques. Different versions of the dataset with different imputations were taken to try and get the best accuracy.

1. In the first version (df1), all the numerical columns are subjected to PCA (principal component analysis). Since the dataset has high dimensionality (99 independent variables), considering PCA for reducing feature complexity was a viable choice. Features that retained 95% variance in data were converted to principal components (Decomposed columns). The end result was reduction of number of variables from 100 to 44 with 26 principal components.

2. In the second version (final_df), all the numerical columns are checked for multi-collinearity. Models such as XGB deal with multi-collinearity pretty well but this step was carried to check for accuracies across different iterations of pre-processed data. Columns with VIF > 5 (Variance inflation factor –checks how much a predictor variable is correlated with all other predictor variables) are eliminated from the dataset. VIF is only calculated for top 30 highly correlated features.

3. A third version that is unaffected by above two steps is also fed into the models. The top 30 highly correlated features are only selected in this case for prediction.



### Model Training and Evaluation:

→ A train-test split of 80-20 is allotted and stratify is passed as a parameter to make sure classes are balanced in test and training dataset.

→ To reduce computational cost and time, all the necessary attributes for our tree model are passed into param_dist. By this method, randomized search CV uses the combinations only passed into param_dist. These parameters are a common starting point but try to cover most bases (different combinations) with respect to achieving high accuracy.

→ The same steps are carried for our XGB model.

→ Classification reports are generated for each variation of our dataset.

→ The model that had only the top 30 highly correlated variables ( No VIF elimination) had the highest accuracy with 62% among the Random forest models. This was the same case with the XGB model.

→ Confusion matrices are plotted for the best models to understand proportions of True negatives to True positives along with other combinations.

![image](https://github.com/user-attachments/assets/35bcc2ae-8711-4d96-8e7f-ce7ffd774a05)



### Parameters passed into param_dist:

n_estimators → Decides the total number of decision trees that will be used to train the model.

Learning rate → Controlling the contribution of each tree to final outcome. Learning rate will be same for all trees.

Max_depth → Increasethe numberof levelsor nodesin ourtreesto fishout for more complexinsights in data.

Min_child_weight → Can be used to limit how many data points are in a child node before the final split. An input of 3 means that nodes will be split until each child node contains at least 3 data points.

Subsample → Each tree will be assigned random subsets of the data. For eg, each tree will be trained on a random 70% of the dataset, and the remaining 30% of the data will not be used for that particular tree.

Colsample_bytree → Similar to subsample, a certain percentage of the features can be chosen randomly for training each tree. This can help the model to generalize well with unseen data.

Alpha, lambda and gamma → Used for performing L1 Regularization, L2 Regularization and loss reduction. These regularization parameters are used to dynamically change coefficient weights to prevent overfitting whereas loss reduction is used to make sure that a split takes place only if minimal error (or loss) is achieved.



### Challenges faced:

✓ Model was consistently overfitting with data. Finding a fix for overfitting proved to be a tedious task.

✓ An highly accurate test score was difficult to achieve. Hyperparameter tuning did not turn out to be efficient.

✓ Different pre-processing steps had to be taken for the dataset to figure out model with best accuracy.



### Key library methods in use →

✓ Scikit-learn: NearestNeighbors, PCA, RandomizedSearchCV, RandomForestClassifier

✓ StatsModels: variance_inflation_factor,add_constant

✓ xgboost: XGBClassifier

