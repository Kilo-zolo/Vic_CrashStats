import sys
sys.path.append('./data_prep.py')
from data_prep import load_and_prep_data
from pre_proc import feature_eng, pre_process
from plots import plt_conf_matrix
from mord import LogisticIT
import os
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
root = os.getenv('root_path')

## Access data and store in dictionary for ease of use
dframes = {"person_df" : root + "data/raw/PERSON.csv",
           "accident_df": root + "data/raw/ACCIDENT.csv",
           "vehicle_df": root + "data/raw/VEHICLE.csv",
           "accident_event_df": root + "data/raw/ACCIDENT_EVENT.csv",
           "accident_location_df": root + "data/raw/ACCIDENT_LOCATION.csv",
           "atmospheric_cond_df": root + "data/raw/ATMOSPHERIC_COND.csv",
           "node_df": root + "data/raw/NODE.csv",
           "node_id_complex_int_id_df": root + "data/raw/NODE_ID_COMPLEX_INT_ID.csv",
           "road_surface_cond_df": root + "data/raw/ROAD_SURFACE_COND.csv"}

## Load & Prep data
loaded_data = load_and_prep_data(dframes)

## Select and Create Features 
main_df = feature_eng(loaded_data)

## One-Hot-Encode, Scale and reduce dimensions of data chosen 
train_test = pre_process(main_df)

## load x_train and y_train_binary into usable variables
print("Loading data needed to fit and train model...")
x_train = train_test['x_train']
y_train_binary = train_test['y_train_binary']
y_train = train_test["y_train"]

## Define Ordinal Regression model
print("Defining Ordinal Regression model...")
olr_model = LogisticIT(max_iter=1000, verbose=3)

# Define the params
# Hyperparams tuned: alpha (regularization parameter)
print("Defining parameters...")
olr_params = {'alpha': [0, 0.5, 1, 2, 5]}
grid_search = {k: v for (k, v) in olr_params.items()}
olr_clf = GridSearchCV(olr_model, olr_params, verbose=2)

# Trainign the model
print("Training Ordinal Regression Classifier model...")
olr_clf.fit(x_train, y_train)

## Save the model to models file
print("Saving model to models file...")

model_dir = root + 'models/olr_clf/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

dump(olr_clf, model_dir + 'olr_clf.joblib')

## Load x_test, y_test into a usable var
print("Loading data required to test and evaluate model")
x_test = train_test['x_test']
y_test = train_test['y_test']


## Predicting results for test data
print("Testing Ordinal Linear Regression model...")
olr_pred = olr_clf.predict(x_test)

## Predicting results for train data
olr_train_pred = olr_clf.predict(x_train)

## Create a classification report for details on precision recall and f1-score
print("Creating a classification report for test data...")
test_class_rep = classification_report(y_test, olr_pred)
print(test_class_rep)

## Saving test classification report 
report_dir = root + 'models/olr_clf/reports/'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

with open(report_dir + 'test_classification_report.txt', 'w') as f:
    f.write(test_class_rep)

## Create a confusion matrix using the y_test and predicted test data
print("Creating test data plots and saving to olr_clf/plts folder...")

plts_dir = root + 'models/olr_clf/plts/'
if not os.path.exists(plts_dir):
    os.makedirs(plts_dir)

test_cm = confusion_matrix(y_test, olr_pred)

print("Creating a classification report for training data...")
train_class_rep = classification_report(y_train, olr_train_pred)
print(train_class_rep)

## Saving train classification report 
report_dir = root + 'models/olr_clf/reports/'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

with open(report_dir + 'train_classification_report.txt', 'w') as f:
    f.write(train_class_rep)

print("Creating plots for training data and saving to olr_clf/plts...")
train_cm = confusion_matrix(y_train, olr_train_pred)

## Plot the graphs to help visualize the accuracy and quality of the model
plt_conf_matrix(test_cm, plts_dir + 'test_confusion_matrix.png')
plt_conf_matrix(train_cm, plts_dir + 'train_confusion_matrix.png')












