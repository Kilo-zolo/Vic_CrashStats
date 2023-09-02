import sys
sys.path.append('/path/to/directory/containing/data_prep')
from data_prep import load_and_prep_data
from pre_proc import feature_eng, pre_process
from plots import plt_roc, plt_conf_matrix, plt_prec_recall
from joblib import dump
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
import os


## Access data and store in dictionary for ease of use
dframes = {"person_df" : "/home/ebi/Blunomy_cs/data/raw/PERSON.csv",
           "accident_df": "/home/ebi/Blunomy_cs/data/raw/ACCIDENT.csv",
           "vehicle_df": "/home/ebi/Blunomy_cs/data/raw/VEHICLE.csv",
           "accident_event_df": "/home/ebi/Blunomy_cs/data/raw/ACCIDENT_EVENT.csv",
           "accident_location_df": "/home/ebi/Blunomy_cs/data/raw/ACCIDENT_LOCATION.csv",
           "atmospheric_cond_df": "/home/ebi/Blunomy_cs/data/raw/ATMOSPHERIC_COND.csv",
           "node_df": "/home/ebi/Blunomy_cs/data/raw/NODE.csv",
           "node_id_complex_int_id_df": "/home/ebi/Blunomy_cs/data/raw/NODE_ID_COMPLEX_INT_ID.csv",
           "road_surface_cond_df": "/home/ebi/Blunomy_cs/data/raw/ROAD_SURFACE_COND.csv"}

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

## Define model; hyperparameters tuned: C (regularization parameter)
print("Training Binary Logistic Regression model...")
binary_lr_model = LogisticRegressionCV(Cs=10, cv=5, class_weight='balanced', random_state=0, max_iter=1000, verbose=3, n_jobs=3)
binary_lr_model.fit(x_train, y_train_binary)

## Save the model to models file
print("Saving model to models file...")

model_dir = '/home/ebi/Blunomy_cs/models/binary_lr_clf/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

dump(binary_lr_model, model_dir + 'binary_lr_clf.joblib')

## Load x_test, y_test_binary into a usable var
print("Loading data required to test and evaluate model")
x_test = train_test['x_test']
y_test_binary = train_test['y_test_binary']


## Predicting results for test data
print("Testing Binary Logistic Regression model...")
binary_lr_pred = binary_lr_model.predict(x_test)

## Predicting results for train data
binary_lr_train_pred = binary_lr_model.predict(x_train)

## Create a classification report for details on precision recall and f1-score
print("Creating a classification report...")
class_rep = classification_report(y_test_binary, binary_lr_pred)
print(class_rep)

## Saving classification report 
report_dir = '/home/ebi/Blunomy_cs/models/binary_lr_clf/reports/'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

with open(report_dir + 'classification_report.txt', 'w') as f:
    f.write(class_rep)

## Create a confusion matrix using the y_test and predicted test data
print("Creating plots and saving to binary_lf_clf/plts folder...")

plts_dir = '/home/ebi/Blunomy_cs/models/binary_lr_clf/plts/'
if not os.path.exists(plts_dir):
    os.makedirs(plts_dir)

cm = confusion_matrix(y_test_binary, binary_lr_pred)

## Plot the graphs to help visualize the accuracy and quality of the model
plt_conf_matrix(cm, plts_dir + 'confusion_matrix.png')

plt_roc(y_test_binary, binary_lr_pred, y_train_binary, binary_lr_train_pred, plts_dir + 'roc_curve.png')

plt_prec_recall(y_test_binary, binary_lr_pred, plts_dir + 'precision_recall.png')








