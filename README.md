# Disaster_pipeline_project
![**Web app homepage**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/homepage.png)

## Content:
1. [**Installation**](#Installation)
2. [**Motivation**](#Motivation)
3. [**ETL pipeline notebook**](#ETL-pipeline-notebook)
4. [**ML pipeline notebook**](#ML-pipeline-notebook)
5. [**Web application**](#Web-application)
6. [**Operation**](#Operation)
7. [**Discussion - imbalanced dataset**](#Discussion-imbalanced-dataset)
8. [**License**](#License)

----------------------------------------------------------------------------------------------------------------------------
### Installation:
* **Python** - There should be no necessary libraries to run the code here beyond the **Anaconda distribution of Python**. 
* **NLTK** - Natural Language Toolkit, using for natural language texts processing.
* **SQL database** - Either **SQLite or SQLAlchemy** for database storage source.
* **Scikit-learn** - Machine learning Algorithm libraries.
* **Flask** - Back-end API of Python for building up web application.
* **Plotly** - Plotting library.
* **Boostrap** - Front-end frame work for build up web application.

### Motivation:
In order to understand and utilize **NLP(Natural Language Pipeline)** process, also for better understanding of data engineering skills, this project is part of Data Science Nanodegree Program by Udacity in collaboration with **Figure Eight**. 

### [ETL pipeline notebook](https://nbviewer.jupyter.org/github/yayuchen/Disaster_pipeline_project/blob/main/raw_files/ETL%20pipeline.ipynb#1):
1. **Extracting** data from origianl dataset.
2. **Transforming** categories into new features.

   **Before transformed values**
   ![before transform](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/before_trans.png)
   
   **After transformed values**
   ![after](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/after_trans.png)
   
3. **Loading** cleaning dataset into database for machine learing. 
4. Refactoring the process which used in notebook and build a [**ETL pipeline python script**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/data/process_data.py) for future using.

### [ML pipeline notebook](https://nbviewer.jupyter.org/github/yayuchen/Disaster_pipeline_project/blob/main/raw_files/ML_pipeline.ipynb#1):
1. Loading data for database.
2. Build a function for text **normalization, tokenization, remove stop words and lemmatization**.
3. Using for loop and **pipeline** to find the best performance classifier.
4. Using **GridSearchCV** to tune hyper parameters for better performance.
5. Saving model also output classification report.  
6. Refactoring the process which used in notebook and build a [**ML pipeline python script**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/models/train_classifier.py) for future using. 

### Web application:
1. Opening [**run.py**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/app/run.py) file.
2. Loading data and model
3. Setting graph variables and define x, y for each figure.
4. Execute web app, using **python run.py**.
5. Typing any text and summit.
6. At go page will return **result** also highlight **labels**.

   **Classification result**
   ![text](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/weather.png)
   
   **Classification labels**
   ![result](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/result.png)
   ![labels](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/labels.png)
   
### Operation:
1. Git clone repo:

    git clone https://github.com/yayuchen/Disaster_pipeline_project.git
    
2. Using Anaconda Prompt and change directory to Python scripts repo:

    **cd <folder_repo>**
    
3. Executing ETL pipeline and ML pipeline, inputing Python scripts, file name and file path into command:

    **python train_classifier.py Disaster_database.db disaster.pkl**
    ![pipeline](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/operation.png)
    
4. Execution web application:

    **python run.py**
    ![web](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/run_app.png)
    
5. Check web application at local device:

    type **localhost:5000/** to find web application 

### Discussion - [imbalanced dataset](https://nbviewer.jupyter.org/github/yayuchen/Disaster_pipeline_project/blob/main/raw_files/Imbalances_dataset.ipynb)

   **Bar plot of categories values** 
   ![bar plot](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/bar_plot.png)
 
i. **Conclusion**: As above, all of labels in this case are imbalanced, it also showed the metric result at classification report and confusion matrix.

   **Classification report and confusion matrix**
  ![metric report](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/class_metrics.png)

ii. **Impact**: As most of the machine learning algorithms are developed with the assumption of the class is balanced, this case could lead to a poor predictive performance for minority classes.

iii. **Way to deal**: 

        Resampling the train data: there are 2 differnt way of resampling, **under sampling** and **over sampling**.
        1. Under sampling: resampling by reducing the number of majority classes to achieve our goal, it could cause a issue which might loss the potential            information from majority classes.
        2. Over sampling: the opposite way to under resampling by increasing the number of minority classes to balance training data, there is a technique              called **SMOTE(Synthetic Minority Over-Sampling Technique)** to help to deal with this kind of issue.
        
iiii. **Reference**: 

        **For imbalanced training data**
        https://imbalanced-learn.org/stable/introduction.html
        https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
        https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
        
        **For imbalanced multilabel classification**
        https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87
        https://github.com/niteshsukhwani/MLSMOTE/blob/master/mlsmote.py

### License:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
