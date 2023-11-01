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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
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
                KNeighborsClassifier(1),
                # SVC(kernel="rbf", C=1),
                # SVC(gamma='auto', C=1),   
                # DecisionTreeClassifier(criterion='entropy', max_depth=20),
                RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=10, max_features=1),
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

# Define the threshold for binary classification
threshold = 0.5

# Define a custom scoring function
def custom_score(y_true, y_pred, fn=accuracy_score):
    # Apply the threshold to obtain binary predictions
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    # Calculate and return the custom metric
    # Replace this with your own custom metric calculation
    return fn(y_true, y_pred_binary)



def eval_classifiers(X, y, labels, **kwargs):

    cv_scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    # 'cross_entropy_loss': make_scorer(log_loss, labels=labels),
    # 'average_precision_score' : make_scorer(average_precision_score, average='weighted', pos_label =0),
    'cohen_kappa_score' : make_scorer(cohen_kappa_score, labels=labels),
    'f1_score' : make_scorer(f1_score, average='weighted', labels=labels),
    'recall_score' : make_scorer(recall_score, average='weighted', labels=labels),
    # 'roc_auc_score': make_scorer(roc_auc_score, average='weighted', labels=labels, multi_class = 'ovr'),
    # 'specificity_score' : make_scorer(recall_score, pos_label=0, average='binary', labels=labels),
    }
    # Define the list of scoring metrics
    mean_res = pd.DataFrame()
    std_res = pd.DataFrame()
    
    for i, clf in tqdm(enumerate(classifiers), desc="Classifiers are running...."):
        # ax = plt.subplot(len(classifiers) + 1, i)
        clf_key = str(clf)
        
        
        clf = Pipeline(steps=[('scaler',StandardScaler()),
                            ('pca', PCA(n_components=200)), 
                            ('estimator',clf)])
        
        # Apply cross-validated model here.
        cv = StratifiedKFold(n_splits=100, shuffle=True)  # Specify the number of desired folds
        cv_scores = cross_validate(clf, X, y, cv=cv, scoring=cv_scorers, return_train_score=False, return_estimator=True, n_jobs=-1,verbose=2)  # Specify the list of scoring metrics
        # print(cv_scores)
        # print(np.array(cv_scores.values()))
        estimators = cv_scores['estimator']

        # Delete estimators
        del cv_scores['estimator']
        # Use sklearn metrics AUC.
        for j, key in enumerate(cv_scores.keys()):
            mean_res.loc[clf_key, key] = np.mean(cv_scores[key])
            std_res.loc[clf_key, key] = np.std(cv_scores[key])
    
    filename = f'classifiers/results/train_color_texture_shape_mean_Kn_MC.csv'

    mean_res.to_csv(filename)

    filename = f'classifiers/results/train_colort_texture_shape_std_Kn_MC_pca095.csv'

    std_res.to_csv(filename)
    
    return estimators, cv_scorers

if __name__ == "__main__":

    from sklearn.model_selection import train_test_split

    data = pd.read_csv('./features/all/features_train_HSV_GLCM_shape_MC_pca095.csv')

    if len(data['label'].unique()) == 2:
        category_mapping = {'nevus': 1, 'others': 0} # Should we switch?
        labels = [0, 1]

    else:

        print(len(data))
        data_exc = data[data['label'].isin(['mel', 'bcc', 'scc'])]
        print(len(data_exc))

        print(data_exc['label'])
        category_mapping = {
                            # 'nev': 0, 
                            # 'ack': 1, 
                            'bcc': 1, 
                            # 'bkl': 3, 
                            # 'def': 4, 
                            'mel': 0, 
                            'scc': 2, 
                            # 'vac': 7
                            }
        labels = [0, 1, 2]
        

    y =  data_exc['label'].astype('category').map(category_mapping)

    X = data_exc.iloc[:, 1:-1]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)
    
    # Standardization
    scaler = StandardScaler()
    X_train_ = scaler.fit_transform(X)

    # PCA
    pca = PCA(0.95)
    pca.fit(X_train_)
    X_train_ = pca.transform(X_train_)

    estimators, cv_scorers = eval_classifiers(X_train_, y, labels = labels) 
    
    for ind, estimator in enumerate(estimators):
        with open(f'classifiers/models/{time}_modelKn.pickle', 'wb') as fp:
            tqdm(pickle.dump(estimator,fp),desc='Saving the model.....')
            fp.close()

