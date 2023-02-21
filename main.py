import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def reload_datasets():
    return {"iris_dataset": pd.read_csv(IRIS_URL),
            "titanic_dataset": pd.read_csv(TITANIC_URL),
            "houses_dataset": pd.read_csv(HOUSES_URL),
            "cancer_dataset": pd.read_csv(CANCER_URL)
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
        # dataset['species'] = dataset['species'].astype(int)
        Y = dataset['species']
        X = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    elif name == "titanic_dataset":
        dataset['Survived'] = dataset['Survived'].astype(int)
        Y = dataset['Survived']
        X = dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
    elif name == "houses_dataset":
        dataset['SalePrice'] = dataset['SalePrice'].astype(int)
        Y = dataset['SalePrice']
        X = dataset[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
                     'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                     'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                     'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                     'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                     'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                     'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                     'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
                     'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
                     'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                     'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
                     'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
                     'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
                     'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                     '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
                     'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']]
    elif name == "cancer_dataset":
        dataset['diagnosis'] = dataset['diagnosis'].astype(int)
        Y = dataset['diagnosis']
        X = dataset.drop([['diagnosis', 'id']])
        """
        X = dataset[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                     'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
                     'symmetry_mean', 'fractal_dimension_mean', 'radius_se']]
        """
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
