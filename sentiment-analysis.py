#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import json
import pandas as pd
import boto3
from io import StringIO, BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True

vectorizer = CountVectorizer()
classifier = LogisticRegression()
total_score = -1

@app.route('/', methods=['GET'])
def home():
    return get_home()

@app.route('/train', methods=['GET'])
def train():
    result = read_rankings_and_train()
    return json.dumps(result)

@app.route("/sentiment/<sentence>")
def predict_sentence(sentence):
    result = predict(sentence)
    return json.dumps(result)

def get_home():
    return "<h1>Sentiment Analysis</h1><p>Simple sentiment analysis</p>"

# read the ranking files from s3 and train it
def read_rankings_and_train():
    result = {}
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

        result = {
            'training_avg_score': total_score
        }
    except Exception as e:
        print(e)

    return result

def predict(sentence):
    # training is not done
    if(total_score < 0):
        read_rankings_and_train()

    sentences = []
    sentences.append(sentence)
    review_transformed = vectorizer.transform(sentences)
    review_result = classifier.predict(review_transformed)

    review = 'Positive review' if review_result[0] == 1 else 'Negative Review'
    print(review)

    result = {
        "review": review
    }
    return result

app.run()