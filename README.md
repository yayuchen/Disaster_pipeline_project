# Disaster_pipeline_project
![**Web app homepage**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/homepage.png)

## Content:
1. [**Installation**](#Installation)
2. [**Motivation**](#Motivation)
3. [**ETL pipeline notebook**](#ETL-pipeline-notebook)
4. [**ML pipeline notebook**](#ML-pipeline-notebook)
5. [**Web application**](#Web-application)
6. [**Operation**](#Operation)
7. [**Discussion-imbalanced dataset**](#Discussion-imbalanced-dataset)
8. [**Reference**](#Reference)
9. [**License**](#License)

----------------------------------------------------------------------------------------------------------------------------
## Installation:
* **Python** - There should be no necessary libraries to run the code here beyond the **Anaconda distribution of Python**. 
* **NLTK** - Natural Language Toolkit, using for natural language texts processing.
* **SQL database** - Either **SQLite or SQLAlchemy** for database storage source.
* **Scikit-learn** - Machine learning Algorithm libraries.
* **Flask** - Back-end API of Python for building up web application.
* **Plotly** - Plotting library.
* **Boostrap** - Front-end frame work for build up web application.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Motivation:
In order to understand and utilize **NLP(Natural Language Pipeline)** process, also for better understanding of data engineering skills, this project is part of Data Science Nanodegree Program by Udacity in collaboration with **Figure Eight**. 

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## [ETL pipeline notebook](https://nbviewer.jupyter.org/github/yayuchen/Disaster_pipeline_project/blob/main/raw_files/ETL%20pipeline.ipynb#1):


* **Extracting** data from origianl dataset.
* **Transforming** categories into new features.
>
>
   Before transformed values
   ![before transform](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/before_trans.png)
>   
>   
   After transformed values
   ![after](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/after_trans.png)
>   
>   
* **Loading** cleaning dataset into database for machine learing. 
* Refactoring the process which used in notebook and build a [**ETL pipeline python script**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/data/process_data.py) for future using.
>



>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## [ML pipeline notebook](https://nbviewer.jupyter.org/github/yayuchen/Disaster_pipeline_project/blob/main/raw_files/ML_pipeline.ipynb#1):
>
>
* Loading data for database.
* Build a function for text **normalization, tokenization, remove stop words and lemmatization**.
* Using for loop and **pipeline** to find the best performance classifier.
* Using **GridSearchCV** to tune hyper parameters for better performance.

    
    find out pipeline's hyper-parameters
    ![modify](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/modify_hyper.png)
    
    
    use GridSearchCV to tune for better classifier performance
    ![tune](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/tune_by_grid.png)
    
    
* Saving model also output classification report.  
* Refactoring the process which used in notebook and build a [**ML pipeline python script**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/models/train_classifier.py) for future using. 
>



>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Web application:
>
>
* Opening [**run.py**](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/app/run.py) file.
* Loading data and model
* Setting graph variables and define x, y for each figure.
* Execute web app, using **python run.py**.
* Typing any text and summit.
* At go page will return **result** also highlight **labels**.



   Classification result
   ![text](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/weather.png)



   Classification labels
   ![result](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/result.png)
   ![labels](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/labels.png)
>   



>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Operation:
* Git clone repo:

      git clone https://github.com/yayuchen/Disaster_pipeline_project.git
>    
* Using Anaconda Prompt and change directory to data/model/app folder:

      cd <folder_repo>
>   ![cd data](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/cd_data.png)   


* Executing **ETL pipeline**, inputing Python scripts, input datasets filepath and database filepath into command:


    python process_data.py disaster_messages.csv disaster_categories.csv Disaster_database.db
    ![execute data](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/execute_data.png)
     
    successful saved data from pipeline: **creating a new database file in data folder and saved data**
    ![save data](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/add_db.png)
    
    
* Execution **ML pipeline**, inputing Python scripts, database filepath and pickle filepath into command:

    python train_classifier.py Disaster_database.db disaster.pkl
    ![pipeline](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/operation.png) 
    
    successful trained data from pipeline: **creating a new pickle file in model folder also shown classification report in command**
    ![train model](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/save_model.png)


* Execution web application:


    python run.py
    ![web](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/run_app.png)
>  
>  
* Check web application at local server:

      localhost:5000/
>



>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Discussion-[imbalanced dataset](https://nbviewer.jupyter.org/github/yayuchen/Disaster_pipeline_project/blob/main/raw_files/Imbalances_dataset.ipynb)
>
>
   Bar plot of categories values 
   ![bar plot](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/bar_plot.png)
>   
> 
* **Conclusion**: As above, all of labels in this case are imbalanced, it also showed the metric result at classification report and confusion matrix.
>
>
  ![metric report](https://github.com/yayuchen/Disaster_pipeline_project/blob/main/images/class_metrics.png)
>  
>
* **Impact**: As most of the machine learning algorithms are developed with the assumption of the class is balanced, this case could lead to a **poor predictive performance for minority classes**.
>
>
* **Way to deal**: Resampling the training data, there are 2 differnt ways of resampling, **under sampling** and **over sampling**.
>
>   
1. **Under sampling**: Resampling by reducing the number of majority classes to achieve our goal, it could cause a issue which might loss the potential            information from majority classes.
>      
2. **Over sampling**: The opposite way to under resampling by increasing the number of minority classes to balance training data, there is a technique              called **SMOTE(Synthetic Minority Over-Sampling Technique)** to help to deal with this kind of issue.




>        
## Reference: 
>
>
#### For imbalanced training data:

* https://imbalanced-learn.org/stable/introduction.html
* https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
* https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
>       
>        
#### For imbalanced multilabel classification:

* https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87
* https://github.com/niteshsukhwani/MLSMOTE/blob/master/mlsmote.py
>



>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## License:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
