import numpy as np
import pickle
from time import gmtime, strftime

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score, f1_score, cohen_kappa_score, recall_score, log_loss
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier    
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lars, ElasticNet, RidgeClassifier, BayesianRidge
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
import pandas as pd
from tqdm import tqdm
import re
from sklearn.feature_selection import SelectKBest, mutual_info_classif


time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

classifiers = [
                # LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5, max_iter=200),
                # LogisticRegression(solver='liblinear', penalty='l1'),
                # LogisticRegression(penalty='l2', max_iter=200),
                # ExtraTreesClassifier(criterion='entropy', n_estimators=100, random_state=0),
                # GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=0),
                # KNeighborsClassifier(1),
                SVC(kernel="rbf", C=1),
                # SVC(gamma='auto', C=1),   
                # DecisionTreeClassifier(criterion='entropy', max_depth=20),
                # RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=10, max_features=1),
                # BernoulliNB(),
                # OneClassSVM(),
                # SGDClassifier(),
                # RidgeClassifier(solver='lsqr'),
                # PassiveAggressiveClassifier(),
                # GradientBoostingClassifier(),
                # RadiusNeighborsClassifier(),
                # Lasso(),
                # LinearSVC(),
                # ElasticNet(),
                # BayesianRidge(),
                # NearestCentroid(),
                # KernelRidge(alpha = 0.1),
                # NuSVC(),
                # MLPClassifier(alpha=1, max_iter=1000),
                # AdaBoostClassifier(),
                # GaussianNB(),
                # LinearDiscriminantAnalysis(),
                # GaussianMixture(),
                # QuadraticDiscriminantAnalysis(),
            ]

cv_metrics = {
    'accuracy_score': make_scorer(accuracy_score)
    }

# Define the threshold for binary classification
threshold = 0.7

# Define a custom scoring function
def custom_score(y_true, y_pred, fn=accuracy_score):
    # Apply the threshold to obtain binary predictions
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    # Calculate and return the custom metric
    # Replace this with your own custom metric calculation
    return fn(y_true, y_pred_binary)



def eval_classifiers(X, y):


    # Define the list of scoring metrics
    mean_res = pd.DataFrame()
    std_res = pd.DataFrame()
    
    for i, clf in tqdm(enumerate(classifiers), desc="Classifiers are running...."):
        # ax = plt.subplot(len(classifiers) + 1, i)
        clf_key = str(clf)
        
        
        clf = Pipeline(steps=[
                            ('estimator',clf)])
        
        # Apply cross-validated model here.
        cv = StratifiedKFold(n_splits=10, shuffle=True)  # Specify the number of desired folds
        cv_scores = cross_validate(clf, X, y, cv=cv, scoring=cv_metrics, return_train_score=False, return_estimator=True, verbose=4)  # Specify the list of scoring metrics
        # print(cv_scores)
        # print(np.array(cv_scores.values()))
        estimators = cv_scores['estimator']

        # Delete estimators
        del cv_scores['estimator']
        # Use sklearn metrics AUC.
        for j, key in enumerate(cv_scores.keys()):
            mean_res.loc[clf_key, key] = np.mean(cv_scores[key])
            std_res.loc[clf_key, key] = np.std(cv_scores[key])
    
    filename = f'classifiers/results/train_color_texture_shape_mean_testFeatureS3.csv'

    mean_res.to_csv(filename)

    filename = f'classifiers/results/train_colort_texture_shape_std_testFeatureS3.csv'

    std_res.to_csv(filename)
    
    return estimators

if __name__ == "__main__":

    from sklearn.model_selection import train_test_split

    data = pd.read_csv('./features/all/features_train_HSV_GLCM_shape_gloh.csv')

    category_mapping = {'nevus': 0, 'others': 1} # Should we switch?
    y_train =  data['label'].astype('category').map(category_mapping)
    X_train = data.iloc[:,1:-1]
    
    data = pd.read_csv('./features/all/features_val_HSV_GLCM_shape_gloh.csv')

    category_mapping = {'nevus': 0, 'others': 1}
    y_val =  data['label'].astype('category').map(category_mapping)

    X_val = data.iloc[:, 1:-1]
    
    data = pd.read_csv('./features/all/features_test_HSV_GLCM_shape_gloh.csv')
    category_mapping = {'nevus': 0, 'others': 1}
    y_tets =  data['label'].astype('category').map(category_mapping)
    X_test = data.iloc[:, 1:-1]
    
    
    scaler = MinMaxScaler()
    X_train_ = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # PCA
    pca = PCA(0.95)
    pca.fit(X_train_)
    X_train_ = pca.transform(X_train_)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    
    print(f'Principal components selection: {pca.n_components_}')
    
    estimators = eval_classifiers(X_train_, y_train) 
    cv = StratifiedKFold(n_splits=10, shuffle=True)  # Specify the number of desired folds
    list_of_test_scores = []
    for ind, estimator in enumerate(estimators):
        test_scores = {}
        y_preds = estimator.predict(X_val)
        
        for metric, metrics in cv_metrics.items():
            test_scores[metric] = metrics._score_func(y_val, y_preds)
        
        list_of_test_scores.append(test_scores)
    
    results = pd.DataFrame()
       
    for ind, estimator in enumerate(estimators):
        test_scores = {}
        y_preds = estimator.predict(X_test)
    
    results[estimator] = y_preds
    filename = f'classifiers/results/binary_output_2.csv'

    results.to_csv(filename)   
        
        

    # Initialize a dictionary to store the sum of values for each key
    sum_dict = {}

    # Iterate over the list of dictionaries
    for d in list_of_test_scores:
        for key, value in d.items():
            if key not in sum_dict:
                sum_dict[key] = 0
            sum_dict[key] += value

    # Calculate the mean values
    average_of_test_scores = {key: sum_val / len(list_of_test_scores) for key, sum_val in sum_dict.items()}

    print(average_of_test_scores)
    
    
    with open(f'classifiers/models/{time}_modelKn.pickle', 'wb') as fp:
        tqdm(pickle.dump(estimators,fp),desc='Saving the model.....')
        fp.close()

