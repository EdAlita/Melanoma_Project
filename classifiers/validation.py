import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score, f1_score, cohen_kappa_score, recall_score, log_loss
from sklearn.metrics import make_scorer


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd

from time import gmtime, strftime
from tqdm import tqdm
import pickle


# Define the threshold for binary classification
threshold = 0.5

# Define a custom scoring function
def custom_score(y_true, y_pred, fn=accuracy_score):
    # Apply the threshold to obtain binary predictions
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    # Calculate and return the custom metric
    # Replace this with your own custom metric calculation
    return fn(y_true, y_pred_binary)

cv_metrics = {
    'accuracy_score': make_scorer(accuracy_score),
    'cross_entropy_loss': make_scorer(log_loss),
    'average_precision_score' : make_scorer(average_precision_score, average='weighted'),
    'cohen_kappa_score' : make_scorer(cohen_kappa_score),
    'f1_score' : make_scorer(f1_score, average='weighted'),
    'recall_score' : make_scorer(recall_score, average='weighted'),
    'roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
    'specificity_score' : make_scorer(recall_score, pos_label=0, average='binary'),
}

if __name__ == '__main__': 
    
    data = pd.read_csv('./features/all/features_val_HSV_GLCM_shape.csv')

    category_mapping = {'nevus': 1, 'others': 0}
    y =  data['label'].astype('category').map(category_mapping)

    X = data.iloc[:, 1:-1]
    
    # Standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    with open('classifiers/models/2023-10-29_11:47:31_modelKn.pickle','rb') as fp:
        try:
            with tqdm(desc='Loading the saved classifier.....'):
                estimators= pickle.load(fp)
        except  EOFError:
            raise Exception('No classifier saved with that name')
        
    list_of_test_scores = []
    
    for ind, estimator in enumerate(estimators):
        test_scores = {}
        y_preds = estimator.predict(X)
        
        for metric, scorer in cv_metrics.items():
            test_scores[metric] = scorer._score_func(y, y_preds)
        
        list_of_test_scores.append(test_scores)
    
    # Initialize a dictionary to store the sum of values for each key
    sum_dict = {}

    # Iterate over the list of dictionaries
    for d in list_of_test_scores:
        for key, value in d.items():
            if key not in sum_dict:
                sum_dict[key] = 0
            sum_dict[key] += value

    # Calculate the mean values
    
    results = pd.DataFrame()
    
    average_of_test_scores = {key: sum_val / len(list_of_test_scores) for key, sum_val in sum_dict.items()}
    
            # Delete estimators
    
    for i, clf in enumerate(estimators):
        clf_key = str(clf)
        for j, key in enumerate(average_of_test_scores.keys()):
            results.loc[clf_key, key] = average_of_test_scores[key]

    filename = f'classifiers/results/validation_colort_texture_shape_mean_SVC_125F.csv'
    
    results.to_csv(filename)
    
    print(average_of_test_scores)    