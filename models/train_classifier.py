import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import pickle


"""
Load cleaned data from SQLite database, tokenizing text values and using machine learning algorithm to train dataset for a best performance classifier model
"""

def load_data(database_filepath):
    """
    Loading data from SQLite database
    
    Arguments:
        database_filepath - str, contains cleaned and transformed message and categories data
        
    Return:
        X - independent variable, text values array
        Y - dependent variable, categories values array
        category_names - categories array
    """
    # create connect 
    engine = create_engine('sqlite:///../data/' + database_filepath)
    # create table name from database filepath
    table_name = database_filepath.replace('.db', '')
    # read data from SQLite database
    # transform related values into 0/1, replace 2 as 1
    df = pd.read_sql_table(table_name, con = engine)
    df.related = df.related.map(lambda x: '1' if x == '2' else x)
    
    # define independent, dependent variables
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = df.columns[4:].values
    
    return X, Y, category_names

def tokenize(text):
    """
    Detect URL like text and replace as urlplaceholder, then normalize, tokenize and lemmatize text values
    
    Arguments:
        text - str, list of str, array of str
        
    Return:
        clean_tokens - normalized, tokenized and lemmatized text values
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # find url text and replace as urlplace text
    detect_url = re.findall(url_regex, text)
    for url in detect_url:
        text = text.replace(url, 'urlplaceholder')
        
    # norlamized and tokenized text
    texts = word_tokenize(re.sub('[^a-zA-Z0-9]', ' ', text.lower()))
    # remove stopwords if exists
    words = [w for w in texts if w not in stopwords.words('english')]
    # tokenize and lemmatizer text
    clean_tokens = [WordNetLemmatizer().lemmatize(word) for word in words]
    
    return clean_tokens


def build_model():
    """
    Using pipeline to automatically execute estimators step by step
    
    Return:
        pipeline - contains different estimators with modified hyper parameters
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer(smooth_idf=False)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, 
                                                         n_estimators=200)))])
    
    return pipeline 


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Using confusion matrix and classification report to determine train model's accuracy with test data and output 
    results on command
    
    Arguments:
        model - trainded model
        X_test - for predict prediction of test data
        Y_test - true values
        categore_names - categorie's label
        
    Return: 
        None
    """
    Y_pred = model.predict(X_test)
    
    true = []
    pred = []

    for i in range(35):
        true.append(Y_test[:,i])
        pred.append(Y_pred[:,i])

    # output each label's confusion matrix and classification report 
    for t, p, col in zip(true, pred, category_names):
        labels = np.unique(t)
        confusion_mat = confusion_matrix(t, p, labels= labels)
        label_accuracy = (t == p).mean()

        print('Labels Name:', col.upper())
        print('Labels:', labels)
        print('Confusion matrix of each label:\n', confusion_mat, '\n')
        print('{} accuracy:'.format(col.upper()), label_accuracy, '\n')
        print('Classification Report: \n', classification_report(t, p), '\n')
    

def save_model(model, model_filepath):
    """
    save and pickle model as a new pickle file
    
    Arguments:
        model - trained and optimized model
        model_filepath - new pickle file path
        
    Return:
        save a new pickle file
    """
    return pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main function for executing pipeline:
    
    procedures:
        1. Load data from database
        2. split independent and target variables
        3. set model variable and train data
        4. output classification report 
        5. save the best performance model as a new pickle file
        
    operation:
        Ex: python train_classifier.py <database_filepath> <model_filepath>
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()