import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
from sklearn import svm as skl_svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

shap.initjs()
np.random.seed(0)

url = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/winequality-red.csv"
df = pd.read_csv(url, sep=';')

IRIS_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/IRIS.csv"
TITANIC_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/titanic.csv"
HOUSES_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/houses.csv"
CANCER_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/cancer.csv"
SLEEPING_URL = "https://raw.githubusercontent.com/ZivBenda/tabular_data_science_biu_final_project/main/datasets/sleeping.csv"


def reload_datasets():
    return {  # "iris_dataset": pd.read_csv(IRIS_URL),
              # "titanic_dataset": pd.read_csv(TITANIC_URL),
            "sleeping_dataset": pd.read_csv(SLEEPING_URL),
              #  "cancer_dataset": pd.read_csv(CANCER_URL)
            }


def preprocess_wine_dataset(df):
    df['quality'] = df['quality'].astype(int)
    Y = df['quality']
    X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']]
    X_train, X_test, Y_train, _ = train_test_split(X, Y, test_size=0.1)
    return X_train, X_test, Y_train


def preprocess_dataset(name, dataset):
    if name == "iris_dataset":
        dataset['species'] = LabelEncoder().fit_transform(dataset['species'])
        dataset['species'] = dataset['species'].astype(int)
        dataset = shuffle(dataset)
        Y = dataset['species']
        X = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    elif name == "titanic_dataset":
        dataset['Sex'] = LabelEncoder().fit_transform(dataset['Sex'])
        dataset['Ticket'] = LabelEncoder().fit_transform(dataset['Ticket'])
        dataset['Cabin'] = LabelEncoder().fit_transform(dataset['Cabin'])
        dataset['Embarked'] = LabelEncoder().fit_transform(dataset['Embarked'])
        dataset['Survived'] = dataset['Survived'].astype(int)
        dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
        Y = dataset['Survived']
        X = dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    elif name == "sleeping_dataset":
        dataset['sl'] = dataset['sl'].astype(int)
        Y = dataset['sl']
        X = dataset[['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sr', 'hr']]
    elif name == "cancer_dataset":
        dataset['diagnosis'] = LabelEncoder().fit_transform(dataset['diagnosis'])
        Y = dataset['diagnosis']
        X = dataset[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                     'smoothness_mean', 'compactness_mean', 'concavity_mean',
                     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                     'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                     'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                     'fractal_dimension_se', 'radius_worst', 'texture_worst',
                     'perimeter_worst', 'area_worst', 'smoothness_worst',
                     'compactness_worst', 'concavity_worst', 'concave points_worst',
                     'symmetry_worst', 'fractal_dimension_worst']]
    elif name == "wine_dataset":
        df['quality'] = df['quality'].astype(int)
        Y = df['quality']
        X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']]
    X_train, X_test, Y_train, _ = train_test_split(X, Y, test_size=0.1)
    return X_train, X_test, Y_train


def KNN(X_train, Y_train, X_test, dataset_name):
    n_neighbors = 15
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    knn.fit(X_train.values, Y_train)
    knn_explainer = shap.KernelExplainer(knn.predict, X_test)
    knn_shap_values = knn_explainer.shap_values(X_test)
    shap.summary_plot(knn_shap_values, X_test, show=False)
    plt.savefig(f'{dataset_name}_KNN.png')


def SVM(X_train, Y_train, X_test, dataset_name):
    svm = skl_svm.SVC(gamma='scale', decision_function_shape='ovo')
    svm.fit(X_train.values, Y_train)
    svm_explainer = shap.KernelExplainer(svm.predict, X_test)
    svm_shap_values = svm_explainer.shap_values(X_test)
    shap.summary_plot(svm_shap_values, X_test, show=False)
    plt.savefig(f'{dataset_name}_SVM.png')


def Linear_Regression(X_train, Y_train, X_test, dataset_name):
    linear_model = LinearRegression()
    linear_model.fit(X_train, Y_train)
    linear_reg_explainer = shap.KernelExplainer(linear_model.predict, X_test)
    linear_reg_shap_values = linear_reg_explainer.shap_values(X_test)
    shap.summary_plot(linear_reg_shap_values, X_test, show=False)
    plt.savefig(f'{dataset_name}_linear_regression.png')


def Logistic_Regression(X_train, Y_train, X_test, dataset_name):
    logistic_reg_model = LogisticRegression(solver='lbfgs')
    logistic_reg_model.fit(X_train, Y_train)
    logistic_reg_explainer = shap.KernelExplainer(logistic_reg_model.predict, X_test)
    logistic_reg_shap_values = logistic_reg_explainer.shap_values(X_test)
    shap.summary_plot(logistic_reg_shap_values, X_test, show=False)
    plt.savefig(f'{dataset_name}_logistic_regression.png')


def main():
    dtfs = reload_datasets()
    for dataset_name, dataset in dtfs.items():
        X_train, X_test, Y_train = preprocess_dataset(dataset_name, dataset)
        KNN(X_train, Y_train, X_test, dataset_name)
        SVM(X_train, Y_train, X_test, dataset_name)
        Linear_Regression(X_train, Y_train, X_test, dataset_name)
        Logistic_Regression(X_train, Y_train, X_test, dataset_name)


if __name__ == '__main__':
    main()
