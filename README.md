# Disaster_pipeline_project
![**Web app homepage**]()

## Content:
1. [**Installation**](#Installation)
2. [**Motivation**](#Motivation)
3. [**ETL pipeline notebook**](# ETL-pipeline-notebook)
4. [**ML pipeline notebook**](#ML-pipeline-notebook)
5. [**Web application**](#Web-application)
6. [**Operation**](#Operation)
7. [**License**](#License)

----------------------------------------------------------------------------------------------------------------------------
### Installation:
**Python** - There should be no necessary libraries to run the code here beyond the **Anaconda distribution of Python**. 
**NLTK** - Natural Language Toolkit, using for natural language texts processing.
**SQL database** - Either **SQLite or SQLAlchemy** for database storage source.
**Scikit-learn** - Machine learning Algorithm libraries.
**Flask** - Back-end API of Python for building up web application.
**Plotly** - Plotting library.
**Boostrap** - Front-end frame work for build up web application.

### Motivation:
In order to understand and utilize NLP(Natural Language Pipeline) process, also for better understanding of data engineering skills, this project is part of Data Science Nanodegree Program by Udacity in collaboration with **Figure Eight**. 

### ETL pipeline notebook:
1. **Extracting** data from origianl dataset.
2. **Transforming** categories into new features.
3. **Loading** cleaning dataset into database for machine learing. 
4. Refactoring the process which used in notebook and build a [**ETL pipeline python script**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/data/ETL%20pipeline.ipynb) for future using.

### ML pipeline notebook:
1. Loading data for database.
2. Build a function for text **normalization, tokenization, remove stop words and lemmatization**.
3. Using for loop and **pipeline** to find the best performance classifier.
4. Using **GridSearchCV** to tune hyper parameters for better performance.
5. Saving model.
6. Refactoring the process which used in notebook and build a [**ML pipeline python script**] for future using.() 

### Web application:
1. Opening [**run.py**] file.()
2. Loading data and model
3. Setting graph variables and define x, y for each figure.
4. Execute web app, using **python run.py**.
5. Typing any text and summit.
6. At go page will show **result** and highlight **labels**.

### Operation:
1. Git clone repo:
    **git clone **
2. Using Anaconda Prompt and change directory to Python scripts repo:
    **cd folder_repo**
3. Executing ETL pipeline and ML pipeline, inputing Python scripts, file name and file path into command:
    Ex: **python train_classifier.py Disaster_database.db disaster.pkl**
    ![python]()
4. Execution web application:
    **python run.py**
    ![web_app]()
    
### License:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
