# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from imblearn.over_sampling import SMOTE

# # Load data
# file_path = '/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/participants_gaze_summary.csv'
# df = pd.read_csv(file_path)

# # Configuration
# condition = 3  # change as needed
# models = {
#     'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'SVM': SVC(probability=True, random_state=42),
#     'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# }

# # Preprocess failure types based on condition
# if condition == 1:
#     df = df[df['Failure_type'] != "No failure"]
# elif condition == 2:
#     df = df[~df['Failure_type'].isin(["Executional", "Mechanical"])]
# elif condition == 3:
#     df['Failure_type'] = df['Failure_type'].replace({
#         'Executional': 'failure',
#         'Decisional':  'failure',
#         'Mechanical':  'failure'
#     })

# # Collect unique participant IDs
# participants = df['Participant_id'].unique()

# # Prepare storage for metrics
# metrics = {name: {
#     'accuracy': [], 
#     'precision_failure': [], 'recall_failure': [],
#     'precision_no_failure': [], 'recall_no_failure': []
# } for name in models}

# # Loop over each participant as test set
# for pid in participants:
#     # Split train and test
#     df_train = df[df['Participant_id'] != pid].copy()
#     df_test  = df[df['Participant_id'] == pid].copy()
    
#     # Map Failure_type to numeric labels
#     label_map = {'No failure': 0, 'failure': 1}
#     df_train['y'] = df_train['Failure_type'].map(label_map)
#     df_test['y']  = df_test['Failure_type'].map(label_map)
    
#     # Select features
#     feature_cols = [c for c in df.columns 
#                     if c not in ('Participant_id', 'Failure_type', 'Object_number', 'Duration')]
#     X_train = df_train[feature_cols]
#     y_train = df_train['y']
#     X_test  = df_test[feature_cols]
#     y_test  = df_test['y']
    
#     # Balance training data with SMOTE
#     smote = SMOTE(k_neighbors=1, random_state=42)
#     X_res, y_res = smote.fit_resample(X_train, y_train)
    
#     # Train and evaluate each model
#     for name, model in models.items():
#         model.fit(X_res, y_res)
#         y_pred = model.predict(X_test)
        
#         # compute metrics
#         acc = accuracy_score(y_test, y_pred)
#         prec_fail = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
#         rec_fail  = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
#         prec_no   = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
#         rec_no    = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        
#         metrics[name]['accuracy'].append(acc)
#         metrics[name]['precision_failure'].append(prec_fail)
#         metrics[name]['recall_failure'].append(rec_fail)
#         metrics[name]['precision_no_failure'].append(prec_no)
#         metrics[name]['recall_no_failure'].append(rec_no)

# # Compute and display average metrics
# print("Average metrics across participants:")
# for name, vals in metrics.items():
#     print(f"\n{name}:")
#     print(f"  Accuracy            : {np.mean(vals['accuracy']):.3f}")
#     print(f"  Precision (failure) : {np.mean(vals['precision_failure']):.3f}")
#     print(f"  Recall    (failure) : {np.mean(vals['recall_failure']):.3f}")
#     print(f"  Precision (no fail) : {np.mean(vals['precision_no_failure']):.3f}")
#     print(f"  Recall    (no fail) : {np.mean(vals['recall_no_failure']):.3f}")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import defaultdict

# Load data
file_path = '/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/participants_gaze_summary.csv'
df = pd.read_csv(file_path)

# Configuration
condition = 1  # 1 = detect 3 failure types
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

# Preprocess failure types based on condition
if condition == 1:
    df = df[df['Failure_type'] != "No failure"]
elif condition == 2:
    df = df[~df['Failure_type'].isin(["Executional", "Mechanical"])]
elif condition == 3:
    df['Failure_type'] = df['Failure_type'].replace({
        'Executional': 'failure',
        'Decisional':  'failure',
        'Mechanical':  'failure'
    })

# Collect unique participant IDs
participants = df['Participant_id'].unique()

# Prepare storage for per-class recall
recall_by_class = {name: defaultdict(list) for name in models}

# Loop over each participant as test set
for pid in participants:
    df_train = df[df['Participant_id'] != pid].copy()
    df_test = df[df['Participant_id'] == pid].copy()

    # Encode labels
    df_train['y'] = df_train['Failure_type'].astype('category').cat.codes
    df_test['y'] = df_test['Failure_type'].astype('category').cat.codes

    # Store label name mapping
    label_names = dict(enumerate(df_train['Failure_type'].astype('category').cat.categories))

    # Feature selection
    feature_cols = [c for c in df.columns 
                    if c not in ('Participant_id', 'Failure_type', 'Object_number', 'Duration')]
    X_train = df_train[feature_cols]
    y_train = df_train['y']
    X_test = df_test[feature_cols]
    y_test = df_test['y']

    # Apply SMOTE
    smote = SMOTE(k_neighbors=1, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        for class_index, class_name in label_names.items():
            if str(class_index) in report:
                recall = report[str(class_index)]['recall']
            else:
                recall = 0.0  # If class is missing in test set
            recall_by_class[name][class_name].append(recall)

# Print average recall per failure type
print("\nAverage percentage of correct detection (recall) per failure type:\n")
for name, class_dict in recall_by_class.items():
    print(f"\n{name}:")
    for failure_type, recalls in class_dict.items():
        print(f"  {failure_type:12s}: {np.mean(recalls) * 100:.2f}%")
