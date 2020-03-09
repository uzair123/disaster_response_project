# Disaster Response Pipeline Project
This project classifies the messages into 36 categories (related to disaster).

ETL piple line transforms the data and saves it in database. 
ML pipleline uses multioutput classificaiton - using random forest classifier. Gridserch is performed with cv=2 for runtime considerations
In app dashboard two additional visualizations are plotted regarding how weather, flood and aid related messages were shared. 

# acknowledgments:
 Following instructions are taken from Udacity disaster response control project
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
