import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
  
    row = categories.iloc[0,:]
    row.describe
    row = row.replace(to_replace ='[-\d]', value = '', regex = True) 
    category_colnames = row

    categories.columns = category_colnames
    for column in categories:    
     # set each value to be the last character of the string
     categories[column] = categories[column].replace(to_replace ='[^\d]', value = '', regex = True)    
     # convert column from string to numeric
     categories[column] = pd.to_numeric(categories[column])
    
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories] , axis =1)
   
    #6- REMOVE DUPLICATES
  
    df.shape
    # check number of duplicates
    df.duplicated().sum()
    df.duplicated().value_counts()
    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    df.duplicated().value_counts()
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)


def main():
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