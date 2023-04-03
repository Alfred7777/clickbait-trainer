import logging
import click
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

@click.command()
@click.option('--vec-method', required=True, type=click.Choice(['tfidf', 'word2vec']))
@click.option('--model', required=True, type=click.Choice(['bayes', 'logreg', 'svm', 'decisiontree', 'randomforest']))
@click.option('--train', required=False, type=click.Path(exists=True), default="train/train.tsv", help="Path to train data file in TSV format.")
@click.option('--dev', required=False, type=click.Path(exists=True), default="dev/dev.tsv", help="Path to dev data file in TSV format.")
@click.option('--test', required=False, type=click.Path(exists=True), default="test/test.tsv", help="Path to test data file in TSV format.")
@click.option('--w2v-model', required=False, type=click.Path(exists=True), default="w2v_model/w2v.bin", help="Path to word2vec model.")
@click.option('--output', required=False, default="model/model.joblib", help="Output path for created model.")
def main(vec_method: str, model: str, train: str, dev: str, test: str, w2v_model: str, output: str):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    logging.info(f" Loading train data...")
    train = pd.read_table(train, on_bad_lines='skip')
    X_train, y_train = train['content'], train['label']
    logging.info(f" Loading dev data...")
    dev = pd.read_table(dev, on_bad_lines='skip')
    X_dev, y_dev = dev['content'], dev['label']
    logging.info(f" Loading test data...")
    test = pd.read_table(test, on_bad_lines='skip')
    X_test, y_test = test['content'], test['label']

    if vec_method == 'tfidf':
        if model == 'bayes':
            model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        if model == 'logreg':
            model = make_pipeline(TfidfVectorizer(), LogisticRegression())
        if model == 'svm':
            model = make_pipeline(TfidfVectorizer(), SVC())
        if model == 'decisiontree':
            model = make_pipeline(TfidfVectorizer(), DecisionTreeClassifier())
        if model == 'randomforest':
            model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
        model.fit(X_train, y_train)

        y_dev_pred = model.predict(X_dev)
        y_test_pred = model.predict(X_test)

        print('Classification Report - dev set')
        print(classification_report(y_dev, y_dev_pred, target_names=['non-clickbait', 'clickbait']))

        print('Classification Report - test set')
        print(classification_report(y_test, y_test_pred, target_names=['non-clickbait', 'clickbait']))


    if vec_method == 'word2vec':
        logging.info("Wczytywanie modelu Word2Vec...")
        word2vec = KeyedVectors.load(w2v_model).wv
        logging.info("Tokenizacja danych...")
        X_train = [word_tokenize(content, language='polish') for content in X_train]
        X_dev = [word_tokenize(content, language='polish') for content in X_dev]
        X_test = [word_tokenize(content, language='polish') for content in X_test]
        print("Wektoryzacja danych...")
        X_train = [np.mean([word2vec[word] for word in content if word in word2vec.key_to_index] or [np.zeros(300)], axis=0) for content in X_train]
        X_dev = [np.mean([word2vec[word] for word in content if word in word2vec.key_to_index] or [np.zeros(300)], axis=0) for content in X_dev]
        X_test = [np.mean([word2vec[word] for word in content if word in word2vec.key_to_index] or [np.zeros(300)], axis=0) for content in X_test]

        if model == 'bayes':
            model = MultinomialNB()
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        if model == 'logreg':
            model = LogisticRegression()
        if model == 'svm':
            model = SVC()
        if model == 'decisiontree':
            model = DecisionTreeClassifier()
        if model == 'randomforest':
            model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_dev_pred = model.predict(X_dev)
        y_test_pred = model.predict(X_test)

        print('Classification Report - dev set')
        print(classification_report(y_dev, y_dev_pred, target_names=['non-clickbait', 'clickbait']))

        print('Classification Report - test set')
        print(classification_report(y_test, y_test_pred, target_names=['non-clickbait', 'clickbait']))

    dump(model, output)

if __name__ == '__main__': 
    main()
