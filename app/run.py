import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objects as pgo
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import pickle


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Disaster_database.db')
df = pd.read_sql_table('overall', engine)

# load model
model = pickle.load(open("../models/disaster_model.pkl", 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visual 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data for visual 2
    label_names = df.columns[5:].values
    label_counts = df[df.columns[5:]].astype('int').sum()
    
    # create visuals
    graphs = [
            # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
            # GRAPH 2 - category graph    
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Labels',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Labels",
                    'tickangle': 60
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["figures-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query, query as dict key with a input value
    query = request.args.get('query', '') 
    
    # use model to predict classification for query, return a result array
    classification_labels = model.predict([query])[0]
    # save category with array values as dict items
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()