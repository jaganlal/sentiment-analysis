FROM python:3.7
COPY ./sentiment-analysis.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./sentiment_classifier_model.pkl /deploy/
COPY ./sentiment_vectorizer_model.pkl /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "sentiment-analysis.py"]