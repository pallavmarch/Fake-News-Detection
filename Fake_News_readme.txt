Dataset: https://www.kaggle.com/datasets/jainpooja/fake-news-detection

1. we imported the fake and true news csv files and checked them
2. Inserting a new column called 'type' where we have 0 for fake news dataset, and 1 for real news dataset
3. Appending both the fake and true news datasets together into a totalnews dataset
4. Spliting the totalnews dataset into training and testing datasests
5. Dropping the title, subject and date columns from the totalnews dataset
6. Assining the 'text' column to the features variable and 'type' column to the targets variable
7. train_test_split function from scikit-learn is used to split the data into training and testing sets. We have a test size of 20% and a random state of 18.
8. Cleaned the totalnews dataset by importing the regular expression. We removed the urls, non word characters and extra spaces from the texts present
