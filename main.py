## Since our best model out of the three models chosen is the binary linear regression model 
## we will load that into our main.py for us to test it with custom data
from joblib import load
from datetime import datetime
import os
import sys
sys.path.append('/home/ebi/Blunomy_cs/src/')
from src.data_prep import load_and_prep_data
from src.pre_proc import feature_eng, pre_process
from src.plots import plt_conf_matrix
from sklearn.metrics import classification_report, confusion_matrix



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

loaded_data = load_and_prep_data(dframes)
main_df = feature_eng(loaded_data)
train_test = pre_process(main_df)

# Load test data
print("Loading test data...")
x_test = train_test['x_test']
y_test_binary = train_test['y_test_binary']

# Load saved model
print("Loading saved model...")
model_path = os.path.join('..', 'models', 'binary_lr_clf', 'binary_lr_clf.joblib')
model = load(model_path)

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(x_test)

# Generate and print classification report
class_rep = classification_report(y_test_binary, y_pred)
print(class_rep)

# Save classification report
today_str = datetime.now().strftime('%Y-%m-%d')
report_dir = os.path.join('/home/ebi/Blunomy_cs/models/custom', 'reports')
os.makedirs(report_dir, exist_ok=True)

filename = f'classification_report_{today_str}.txt'
with open(os.path.join(report_dir, filename), 'w') as f:
    f.write(class_rep)

# Generate and save confusion matrix plot
print("Creating and saving confusion matrix plot...")
plts_dir = os.path.join('/home/ebi/Blunomy_cs/models/custom', 'plts')
os.makedirs(plts_dir, exist_ok=True)

cm = confusion_matrix(y_test_binary, y_pred)
plt_conf_matrix(cm, os.path.join(plts_dir, 'confusion_matrix.png'))