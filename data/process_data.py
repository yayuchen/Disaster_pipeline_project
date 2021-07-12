import sys
import pandas as pd
import re
import sqlite3

""" 
Using process_data.py script to automatically extract raw data, transforming categories data into new features dataframe, concatenating with message dataframe to form a new one and loading new dataframe into database for storage. 
"""

# load message, category csv files and merge
def load_data(messages_filepath, categories_filepath):
    """
    Extracting data from files
    
    Arguments:
        messages_filepath - str, CSV file path which contains messages information
        categories_filepath - str, CSV file path which contains categories information
        
    Reture:
        merged dataframe - contains messages and categories information
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories, on='id')


def clean_data(df):
    """
    Transforming categories columns into a new categories feature dataframe and change column types.
    
    Arguments:
        df - original dataframe which contains original message and categories data
        
    Return:
        new_df - old dataframe concatenate with new features dataframe and remove the old categories column
    """
    # expand categories as new data frame
    new_columns = [re.sub('[^a-zA-Z]', ' ', i).strip() for i in df['categories'][0].split(';')]
    cat_df = df['categories'].str.split(';', expand=True)
    cat_df.columns = new_columns
    
    # remove anything except numerical value
    # change new feature's type
    for column in cat_df:
        cat_df[column] = cat_df[column].apply(lambda x: re.sub('[^0-9]', '', x)).astype('str')
        
    # concatenate old dataframe and new features dataframe
    # remove olf categories column
    new_df = pd.concat([df, cat_df], axis=1)
    new_df = new_df.drop('categories', axis=1).drop_duplicates()
    
    return new_df

def save_data(df, database_filepath):
    """
    Loading cleaned and transformed dataframe into database for storage
    
    Arguments:
        df - cleaned and transformed dataframe which contains message and categories data
        database_filepath - database, automatically create a new database if not exist
        
    Return:
        create a table into SQLite database, automatically replace a new table if exist
    """
    # create a database connect
    conn = sqlite3.connect(database_filepath)
    # replace .db with empty space for new table name
    table_name = database_filepath.replace('.db', '')
    
    return df.to_sql(table_name, con=conn, if_exists='replace', index=False)


def main():
    """
    Main funtion for executing pipeline:
    
    procedures:
        1. Extract data and merge
        2. Transform categories data and concatenate with old dataframe
        3. Load cleaned and transformed dataframe into SQLite database and save
        
    operation: input python scripts, data file and database into command
        Ex: python process_data.py <messages_filepath> <categories_filepath> <database_filepath>
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()