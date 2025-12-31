## Sentiment Analysis with IMDb Reviews
## Project Description

This project performs sentiment analysis on movie reviews using the IMDb dataset.
It classifies reviews as positive or negative using a Logistic Regression model with TF-IDF features.

The system can also predict the sentiment of custom reviews provided by the user.

## Technologies Used

1.Python
2.NLTK
3.pandas
4.scikit-learn
5.Regular Expressions (re)
6.Dataset Used

NLTK’s built-in IMDb Movie Reviews dataset

## How the Project Works

1.Load the IMDb movie reviews dataset.

2.Clean the text by removing special characters and converting to lowercase.

3.Convert sentiment labels to numbers (positive → 1, negative → 0).

4.Split the dataset into training and test sets.

5.Convert text data to numeric features using TF-IDF vectorization.

6.Train a Logistic Regression model on the training set.

7.Evaluate the model using accuracy and classification report.

8.Predict sentiment of new reviews using the trained model.

## Installation Steps

Install Python

Install required libraries using:

pip install nltk pandas scikit-learn


Download NLTK movie reviews dataset (only first time):

import nltk
nltk.download('movie_reviews')

## How to Run the Project

Open the project folder in VS Code or any IDE.

Run the Python file:

python sentiment_analysis_imdb.py


The script will output:

Model accuracy

Classification report

Predicted sentiment for example or custom reviews

## Output

Accuracy score of the model

Detailed classification report

Predicted sentiment for any custom review

Example Output:

Review: The movie was fantastic with great acting
Predicted Sentiment: Positive 😊

## Applications

Movie review analysis

Social media sentiment analysis

Customer feedback analysis

Real-time sentiment monitoring

## Conclusion

This project demonstrates how Logistic Regression with TF-IDF features can be used for sentiment analysis on text data.
It provides an easy way to classify reviews as positive or negative and can be extended to other text datasets.