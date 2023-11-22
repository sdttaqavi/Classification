# Classification
The purpose of this project is to implement and compare the performance of three classification models:
SVM, KNN, and Decision Tree.
For implementation, a dataset has been considered with the goal of predicting individuals' income based on their features.
This dataset is located in the "Data" folder. The "Adult_TrainDataset" dataset, consisting of 32,561 samples, has been chosen for training the models.
The data in this dataset is categorized into two classes, with the "Income" column containing labels,
and the remaining columns representing features that should be utilized for implementation.

In this project, we can use pre-built models from the sklearn library for implementation.
Visualize the training datasets using at least two graphical models and analyze them.
Due to the presence of missing values (Null) in this dataset, we should replace the Null values with appropriate values or delete the corresponding columns.

Finally, after training the models, evaluate their performance using the "Adult_TestDataset" dataset.
We Use the confusion matrix and evaluation metrics such as recall, precision, accuracy, and F1-score to analyze the obtained results,
by using sklearn's ready-made functions to obtain the confusion matrix and evaluation metrics.
