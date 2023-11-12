# Classification
The purpose of this project is to implement and compare the performance of three classification models:
SVM, KNN, and Decision Tree.
For implementation, a dataset has been considered with the goal of predicting individuals' income based on their features.
This dataset is located in the "Data" folder. The "Adult_TrainDataset" dataset, consisting of 32,561 samples, has been chosen for training the models.
The data in this dataset is categorized into two classes, with the "Income" column containing labels,
and the remaining columns representing features that should be utilized for implementation.

In this project, we can use pre-built models from the sklearn library for implementation.
Visualize the training datasets using at least two graphical models and analyze them.
Due to the presence of missing values (Null) in this dataset, you should replace the Null values with appropriate values or delete the corresponding columns.

Since some columns contain Categorical values, if you choose to use them for model training,
we must encode the values of these columns using existing encoding methods and convert them to numerical values.
Note that there are various methods for encoding Categorical data, including One-Hot Encoding and Label Encoding.
Compare these two methods and explain which method may lead to better results. Discuss your choice and proceed with encoding the data using the selected method.

If necessary, we can apply various preprocessing techniques such as normalization, standardization, etc., to this dataset before training the models.

Finally, after training the models, evaluate their performance using the "Adult_TestDataset" dataset.
Use the confusion matrix and evaluation metrics such as recall, precision, accuracy, and F1-score to analyze the obtained results.
Keep in mind that for simplicity, you can use sklearn's ready-made functions to obtain the confusion matrix and evaluation metrics.

In the training of each model, if hyperparameter tuning is performed using sklearn's library,
report the reasons for these choices and analyze your results based on the obtained outcomes.
