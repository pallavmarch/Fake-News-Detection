**Introduction**

Fake news detection is crucial in today's information age, where misinformation can spread rapidly. This project leverages machine learning to automatically classify news articles as real or fake.

**Code Structure**

The code is organized into well-defined functions for data checking (datacheck), text preprocessing (text_pre), model training and testing (mod, pred), and a combined testing function (testing).

**Data Preparation**

Loads separate datasets for real and fake news (True.csv and Fake.csv).
Performs data checks to assess shape, unique values, missing values, and data types.
Creates a new class column to label real (1) and fake (0) news.
Merges the datasets into a combined DataFrame df_combine.
Randomizes the rows in df_combine.
Preprocesses text by removing special characters, URLs, HTML tags, punctuation, newline characters, and alphanumeric digits using the text_pre function.
Splits data into training and testing sets (80%/20%) using train_test_split with stratification and a random state for reproducibility.
Applies TF-IDF vectorization (TfidfVectorizer) to convert text into numerical features.
Machine Learning Models

Logistic Regression: A classic model for binary classification, trained using LogisticRegression.
Decision Tree: A tree-based model, trained with DecisionTreeClassifier with parameters for random state and maximum depth.
Random Forest: An ensemble of decision trees, trained with RandomForestClassifier with parameters for complexity control, number of estimators, minimum samples for splitting, and random state.
Neural Network: A simple neural network with one hidden layer, trained with MLPClassifier with parameters for hidden layer size, activation function, and regularization.
Evaluation and User Input

Model performance is evaluated using the classification_report function.
The testing function takes user input for news articles, preprocesses them, and classifies them using all four models, presenting the results in a tabular format.
How to Use

Clone or download the repository.
Install required libraries (pandas, numpy, matplotlib, seaborn, tabulate, sklearn.model_selection, sklearn.feature_extraction.text, sklearn.metrics, sklearn.linear_model, sklearn.tree, sklearn.ensemble, sklearn.neural_network, and warnings).
Make sure you have the True.csv and Fake.csv datasets in the appropriate location (/content/drive/MyDrive/Python/).
Run the Fake_News.ipynb script using a Jupyter Notebook environment.
Follow the prompts to enter news articles for classification.
Disclaimer

Machine learning models can be biased based on the training data. It's essential to use reliable and diverse datasets.
This is a basic implementation for educational purposes. Consider more advanced techniques and rigorous evaluation for real-world deployments
