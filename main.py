## Since our best model out of the three models chosen is the binary linear regression model 
## we will load that into our main.py for us to test it with custom data
from joblib import load
from src.data_prep import load_and_prep_data
from src.pre_proc import feature_eng, pre_process
from src.plots import plt_conf_matrix
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime


## Access data and store in dictionary for ease of use
dframes = {"person_df" : "/home/ebi/Blunomy_cs/data/custom/PERSON.csv",
           "accident_df": "/home/ebi/Blunomy_cs/data/custom/ACCIDENT.csv",
           "vehicle_df": "/home/ebi/Blunomy_cs/data/custom/VEHICLE.csv",
           "accident_event_df": "/home/ebi/Blunomy_cs/data/custom/ACCIDENT_EVENT.csv",
           "accident_location_df": "/home/ebi/Blunomy_cs/data/custom/ACCIDENT_LOCATION.csv",
           "atmospheric_cond_df": "/home/ebi/Blunomy_cs/data/custom/ATMOSPHERIC_COND.csv",
           "node_df": "/home/ebi/Blunomy_cs/data/custom/NODE.csv",
           "node_id_complex_int_id_df": "/home/ebi/Blunomy_cs/data/custom/NODE_ID_COMPLEX_INT_ID.csv",
           "road_surface_cond_df": "/home/ebi/Blunomy_cs/data/custom/ROAD_SURFACE_COND.csv"}

## Load & Prep data
loaded_data = load_and_prep_data(dframes)

## Select and Create Features 
main_df = feature_eng(loaded_data)

## One-Hot-Encode, Scale and reduce dimensions of data chosen 
train_test = pre_process(main_df)

## load x_train and y_train_binary into usable variables
print("Loading data needed to fit and train model...")
x_test = train_test['x_test']
y_test_binary = train_test['y_test_binary']

## load saved binary linear regression model
model = load('../models/binary_lr_clf/binary_lr_clf.joblib')

## Predicting results for test data
print("Evaluating Binary Logistic Regression model...")
y_pred = model.predict(x_test)

## Create a classification report for details on precision recall and f1-score
print("Creating a classification report...")
class_rep = classification_report(y_test_binary, y_pred)
print(class_rep)

## Saving classification report 
today_str = datetime.now().strftime('%Y-%m-%d')

report_dir = '/home/ebi/Blunomy_cs/models/custom/reports/'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

filename = f'classification_report_{today_str}.txt'

with open(os.path.join(report_dir, filename), 'w') as f:
    f.write(class_rep)

## Create a confusion matrix using the y_test and predicted test data
plts_dir = '/home/ebi/Blunomy_cs/models/custom/plts/'
if not os.path.exists(plts_dir):
    os.makedirs(plts_dir)

cm = confusion_matrix(y_test_binary, y_pred)

## Plot the graphs to help visualize the accuracy and quality of the model
plt_conf_matrix(cm, plts_dir + 'confusion_matrix.png')