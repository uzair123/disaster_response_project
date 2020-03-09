import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.externals import joblib
from sklearn.utils import parallel_backend 

nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

def load_data(database_filepath):
 # load data from database
 engine = create_engine('sqlite:///'+database_filepath)
 df = pd.read_sql_table("disaster_response", engine)
 X = df.message.values 
 cat_df = df.loc[:,'related':]
 names  = cat_df.columns
 y      = cat_df.values 
 return X,y,names

def tokenize(text):
   
    # tokenize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)    

    #Stopword removal
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok  = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline_rforest  = Pipeline([ ('vect',  CountVectorizer(tokenizer=tokenize)),
                                   ('tfidf', TfidfTransformer()), 
                                   ('clf',   MultiOutputClassifier(RandomForestClassifier())),])
    
    parameters = {     
        'clf__estimator__n_estimators': [100,200],
        'clf__estimator__min_samples_split': [2,4]
    }

    cv = GridSearchCV(pipeline_rforest, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
  y_pred = model.predict(X_test)
  print("\nBest Parameters:", model.best_params_)
  print("--------------------------------------")
  for ind in range (len(category_names)):
    print('CATEGORY',category_names[ind])
    print(classification_report(Y_test[:,ind], y_pred[:,ind]))
 
   


def save_model(model, model_filepath):
  # Save the model as a pickle in a file 
  joblib.dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        print('Spliting Data for Training and Test :-)')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        with parallel_backend('multiprocessing'):
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