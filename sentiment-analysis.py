#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import json
import pandas as pd
import boto3
import pickle

from io import StringIO, BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import flask

app = flask.Flask(__name__)
app.config['DEBUG'] = True

sentiment_vectorizer_model_file = 'sentiment_vectorizer_model.pkl'
sentiment_classifier_model_file = 'sentiment_classifier_model.pkl'

@app.route('/', methods=['GET'])
def home():
    return get_home()

@app.route('/train', methods=['GET'])
def train():
    result = read_rankings_and_train()
    return json.dumps(result)

@app.route('/sentiment/<sentence>')
def predict_sentence(sentence):
    result = predict(sentence)
    return json.dumps(result)

def get_home():
    return '<h1>Sentiment Analysis</h1><p>Simple sentiment analysis</p>'

# read the ranking files from s3 and train it
def read_rankings_and_train():
    vectorizer = CountVectorizer()
    classifier = LogisticRegression()

    result = {}
    total_score = 0

    try:
        # connect to s3
        s3 = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')

        # files that're uploaded to s3
        filepath_dict = {'yelp': 'sentiment-analysis/yelp_labelled.txt',
                 'amazon': 'sentiment-analysis/amazon_cells_labelled.txt',
                 'imdb': 'sentiment-analysis/imdb_labelled.txt'}

        # concatinated (from the above files) data frame list 
        df_list = []

        # iterate the filepath, read the data frame and concatinate to df_list
        for source, filepath in filepath_dict.items():
            obj = s3.get_object(Bucket='ml-data.s3.us-east-1.amazonaws.com', Key=filepath)
            df = pd.read_csv(BytesIO(obj['Body'].read()), names=['sentence', 'label'], sep='\t')
            df['source'] = source  # Add another column filled with the source name
            df_list.append(df)
            df = pd.concat(df_list)
        
        # prepare the model
        for source in df['source'].unique():
            df_source = df[df['source'] == source]
            sentences = df_source['sentence'].values
            y = df_source['label'].values

            sentences_train, sentences_test, y_train, y_test = train_test_split(
                sentences, y, test_size=0.25, random_state=1000)

            vectorizer.fit(sentences_train)
            X_train = vectorizer.transform(sentences_train)
            X_test  = vectorizer.transform(sentences_test)

            classifier.fit(X_train, y_train)
            score = classifier.score(X_test, y_test)
            print('Accuracy for {} data: {:.4f}'.format(source, score))
            total_score += score

        if(total_score):
            total_score /= 3

        save_model(vectorizer, sentiment_vectorizer_model_file)
        save_model(classifier, sentiment_classifier_model_file)

        result = {
            'training_avg_score': total_score
        }
    except Exception as e:
        print('Exception in read_rankings_and_train: {0}'.format(e))

    return result

def save_model(model, filename):
    result = {}

    # Save to file in the current working directory
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
            result = {'save-model': 'success'}

    except Exception as e:
        print('Exception in save_model: {0}'.format(e))
        result = {'save-model': 'failed'}

    return result

def load_model(filename):
    
    # Load from file
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        print('Exception in load_model: {0}'.format(e))
        raise

    return model

def predict(sentence):
    result = {}

    try:
        loaded_vectorizer = load_model(sentiment_vectorizer_model_file)
        loaded_classifier = load_model(sentiment_classifier_model_file)
    except Exception as e:
        print('Error in load model')
        return {
            'Error': 'Error in load model'
        }

    try:
        sentences = []
        sentences.append(sentence)
        review_transformed = loaded_vectorizer.transform(sentences)
        review_result = loaded_classifier.predict(review_transformed)

        review = 'Positive review' if review_result[0] == 1 else 'Negative Review'
        print(review)
        result = {
            'review': review
        }
    except Exception as e:
        print('Exception in predict: {0}'.format(e))
        result = {
            'review': 'Failure'
        }

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)