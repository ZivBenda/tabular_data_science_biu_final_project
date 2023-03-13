#imports

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn import svm as skl_svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor,XGBClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
from xgboost import plot_importance

#datasets urls and names
IRIS_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/IRIS.csv"
TITANIC_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/titanic.csv"
CANCER_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/cancer.csv"
SLEEPING_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/sleeping.csv"
IRIS_DATASET = "iris_dataset"
TITANIC_DATASET = "titanic_dataset"
CANCER_DATASET = "cancer_dataset"
SLEEPING_DATASET = "sleeping_dataset"

shap.initjs()
#using a SEED in order to reproduce results 
SEED = 2023
np.random.seed(SEED)

# this method loads the datasets into pandas objects and store them on dictionary. 
def reload_datasets():
    return { IRIS_DATASET: pd.read_csv(IRIS_URL),
             TITANIC_DATASET: pd.read_csv(TITANIC_URL),
             SLEEPING_DATASET: pd.read_csv(SLEEPING_URL),
             CANCER_DATASET: pd.read_csv(CANCER_URL)
            }

#Preprocess the datasets
#selecting only potential columns for classification task i.e ignoring (id, name) coulums etc.

def preprocess_dataset(name, dataset):
    if name == IRIS_DATASET:
        Y = dataset['species']
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        X = dataset[features]

    elif name == TITANIC_DATASET:
        Y = dataset['Survived']
        # Ommiting Passenger-id, Name and Ticket columns
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
        X = dataset[features]

    elif name == SLEEPING_DATASET:
        dataset['sl'] = dataset['sl'].astype(int)
        Y = dataset['sl']
        features = ['sr1', 'rr', 't', 'lm', 'bo', 'rem', 'sr2', 'hr']
        X = dataset[features]

    elif name == CANCER_DATASET:
        Y = dataset['diagnosis']
        # Ommiting id and name
        features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                     'smoothness_mean', 'compactness_mean', 'concavity_mean',
                     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                     'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                     'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                     'fractal_dimension_se', 'radius_worst', 'texture_worst',
                     'perimeter_worst', 'area_worst', 'smoothness_worst',
                     'compactness_worst', 'concavity_worst', 'concave points_worst',
                     'symmetry_worst', 'fractal_dimension_worst']
        X = dataset[features]
    return X, Y

# Models Definitions
# here we creating 4 model trainig methods (KNN, SVM, Decision_Tree & Logistic Regression with shap plots.
                                          
def KNN(X_train, Y_train, X_test, dataset_name, class_names):
    print(f'Executing KNN on {dataset_name}')
    plt.subplot(121)
    n_neighbors = 15
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    knn.fit(X_train.values, Y_train)
    knn_explainer = shap.KernelExplainer(knn.predict, X_test, seed=SEED)
    knn_shap_values = knn_explainer.shap_values(X_test)
    shap.summary_plot(knn_shap_values, X_test, show=False, class_names=class_names)
    plt.savefig(f'{dataset_name}_KNN.png')

    y_predict = knn.predict(X_test)
    return y_predict


def SVM(X_train, Y_train, X_test, dataset_name, class_names):
    print(f'Executing SVM on {dataset_name}')
    plt.subplot(121)
    svm = skl_svm.SVC(gamma='scale', decision_function_shape='ovo')
    svm.fit(X_train.values, Y_train)
    svm_explainer = shap.KernelExplainer(svm.predict, X_test, seed=SEED)
    svm_shap_values = svm_explainer.shap_values(X_test)
    shap.summary_plot(svm_shap_values, X_test, show=False, class_names=class_names)
    plt.savefig(f'{dataset_name}_SVM.png')

    y_predict = svm.predict(X_test)
    return y_predict


def Decision_Tree(X_train, Y_train, X_test, dataset_name, class_names):
    print(f'Executing Decision_Tree on {dataset_name}')
    model = DecisionTreeClassifier(random_state=1, max_depth=5)
    model.fit(X_train, Y_train)
    decision_tree_explainer = shap.TreeExplainer(model, seed=SEED)
    decision_tree_shap_values = decision_tree_explainer.shap_values(X_test)
    shap.summary_plot(decision_tree_shap_values, X_test, show=False, class_names=class_names)
    plt.savefig(f'{dataset_name}_decision_tree.png')

    y_predict = model.predict(X_test)
    return y_predict


def Logistic_Regression(X_train, Y_train, X_test, dataset_name, class_names):
    print(f'Executing Logistic_Regression on {dataset_name}')
    logistic_reg_model = LogisticRegression(solver='lbfgs')
    logistic_reg_model.fit(X_train, Y_train)
    logistic_reg_explainer = shap.KernelExplainer(logistic_reg_model.predict, X_test, seed=SEED)
    logistic_reg_shap_values = logistic_reg_explainer.shap_values(X_test)
    shap.summary_plot(logistic_reg_shap_values, X_test, show=False, class_names=class_names)
    plt.savefig(f'{dataset_name}_logistic_regression.png')

    y_predict = logistic_reg_model.predict(X_test)
    return y_predict

#XGBOOST model
# We try to compare between the results of SHAP with the above models and the result of the tree-based XGBOOST model.
def XGBoostClassifier(X_train, Y_train, dataset_name):
    xgb = XGBClassifier()
    xgb.fit(X_train, Y_train)
    # plot feature importance
    plot_importance(xgb)
    pyplot.show()
    plt.savefig(f'{dataset_name}_XGBOOST.png')


# models dictionary
# a wrapper for our classification models methods mentioned above
models={'KNN' : lambda X_train, Y_train, X_test, dataset_name, class_names : KNN(X_train, Y_train, X_test, dataset_name, class_names),
        'SVM' : lambda X_train, Y_train, X_test, dataset_name, class_names : SVM(X_train, Y_train, X_test, dataset_name, class_names),
        'Decision_Tree' : lambda X_train, Y_train, X_test, dataset_name, class_names : Decision_Tree(X_train, Y_train, X_test, dataset_name, class_names),
        'Logistic_Regression' : lambda X_train, Y_train, X_test, dataset_name, class_names:Logistic_Regression(X_train.values, Y_train, X_test, dataset_name, class_names)
        }

#Models training method
# this method purpose is to train all models on specific dataset and print classification report for each of them.
def models_train(X_train, y_train, X_test, y_test, dataset_name, target_strings):
    for model in models.values():
        y_predict = model(X_train, y_train, X_test, dataset_name, target_strings)
        print(classification_report(y_test, y_predict, target_names=target_strings))
        plt.show()

# helper ploting function
def plot_pie_train_test(y_train, y_test):
    plt.figure(figsize=(10, 15))

    plt.subplot(121)
    plt.pie(y_train.value_counts(), labels=y_train.unique(), autopct='%1.2f%%')
    plt.title('Training Dataset')

    plt.subplot(122)
    plt.pie(y_test.value_counts(), labels=y_test.unique(), autopct='%1.2f%%')
    plt.title('Test Dataset')

    plt.tight_layout()
    plt.show()

# this method encodes text-labels into numbers
def encode_labels(y_train, y_test, Y, name):
    le = preprocessing.LabelEncoder()
    trained_le = le.fit(y_train)
    y_train = trained_le.transform(y_train)
    y_test = trained_le.transform(y_test)
    if name == IRIS_DATASET:
        target_strings = le.inverse_transform(np.arange(len(Y.unique())))
    if name == TITANIC_DATASET:
        target_strings = np.array(['Not Survived', 'Survived'])
    if name == SLEEPING_DATASET:
        target_strings = np.array(['3', '1', '0', '2', '4'])
    if name == CANCER_DATASET:
        target_strings = Y.unique()
    return y_train, y_test, target_strings


def feature_selection(X_train, X_test, name):
    if name == "iris_dataset":
        X_train = X_train.drop(columns=['sepal_length'])
        X_test = X_test.drop(columns=['sepal_length'])
    elif name == "titanic_dataset":
        X_train = X_train.drop(columns=['Parch'])
        X_test = X_test.drop(columns=['Parch'])
    elif name == "sleeping_dataset":
        X_train = X_train.drop(columns=['rr'])
        X_test = X_test.drop(columns=['rr'])
    elif name == "cancer_dataset":
        X_train = X_train.drop(
            columns=['compactness_worst', 'concave points_se', 'concavity_se', 'fractal_dimension_se', 'radius_mean',
                     'smoothness_worst', 'symmetry_worst'])
        X_test = X_test.drop(
            columns=['compactness_worst', 'concave points_se', 'concavity_se', 'fractal_dimension_se', 'radius_mean',
                     'smoothness_worst', 'symmetry_worst'])
    return X_train, X_test

# we have only missing values on titanic dataset
def check_missing_values(X, name):
    print(X.isnull().sum())
    if name == TITANIC_DATASET:
        X['Age'].fillna(X['Age'].mean(), inplace=True)
        X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)
        X.drop(['Cabin'], axis=1, inplace=True)
    return X


def main():
    dtfs = reload_datasets()
    for dataset_name, dataset in dtfs.items():
        dataset = shuffle(dataset)
        X, Y = preprocess_dataset(dataset_name, dataset)

        X = check_missing_values(X, dataset_name)
        # Encoding categorical features
        if dataset_name == TITANIC_DATASET:
            X['Sex'] = LabelEncoder().fit_transform(X['Sex'])
            X['Embarked'] = LabelEncoder().fit_transform(X['Embarked'])

        # Split dataset to train and test with the same ratio as before
        plt.pie(Y.value_counts(), labels=Y.unique(), autopct='%1.2f%%')
        splitter=StratifiedShuffleSplit(n_splits=1,random_state=12, test_size=0.2)
        for train, test in splitter.split(X, Y):  # this will splits the index
            X_train = X.iloc[train]
            y_train = Y.iloc[train]
            X_test = X.iloc[test]
            y_test = Y.iloc[test]

        plot_pie_train_test(y_train, y_test)

        # Encode labels
        y_train, y_test, target_strings = encode_labels(y_train, y_test, Y, dataset_name)

        # Train the models
        models_train(X_train, y_train, X_test, y_test, dataset_name, target_strings)
        XGBoostClassifier(X_train, y_train, dataset_name)

        feature_selection(X_train, X_test, dataset_name)

        # Train the models again after feature selection with SHAP
        models_train(X_train, y_train, X_test, y_test, dataset_name, target_strings)


if __name__ == '__main__':
    main()
