import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from MyImputerMissing import MyImputerMissing
from EDA import EDA
from typing import Tuple

class private_scaler():
    def __init__(self):
        self.scaler = {
            "enc_ord": OrdinalEncoder(),  # Изменено: используем OrdinalEncoder для нескольких столбцов
            "enc_onh": OneHotEncoder(sparse_output=False),  # Добавлен sparse_output=False
            "mm": MinMaxScaler(),
            "rb": RobustScaler()
        }
        self.mask_scale = {
            "enc_ord": [],  # Изменено с enc_lb на enc_ord
            "enc_onh": [],
            "mm": [],
            "rb": []
        }
        self.label = ["Gender", "HasDependents", "PaperlessBilling"]
        self.label_encoders = {}  # Новый атрибут для хранения отдельных LabelEncoder
    
    def __define_mask(self, X: pd.DataFrame, with_support_calls: bool = True):
        for column in X.columns:
            if ((X[column].dtype == "object") or (X[column].dtype == "category")):
                if column in self.label:
                    self.mask_scale["enc_ord"].append(column)  # Используем OrdinalEncoder
                else:
                    self.mask_scale["enc_onh"].append(column)
            elif column == "CustomerSatisfaction":
                self.mask_scale["mm"].append(column)
            elif column == "SupportCalls":
                if with_support_calls:
                    self.mask_scale["rb"].append(column)
                else:
                    continue
            else:
                self.mask_scale["rb"].append(column)
    
    def fit(self, X: pd.DataFrame, with_support_calls: bool = True):
        self.__define_mask(X=X, with_support_calls=with_support_calls)
        
        # Обрабатываем каждый тип масштабирования отдельно
        for key in self.mask_scale.keys():
            if self.mask_scale[key]:  # Проверяем, есть ли столбцы для обработки
                if key == "enc_ord":
                    # Для OrdinalEncoder передаем все столбцы одновременно
                    self.scaler[key].fit(X[self.mask_scale[key]])
                elif key == "enc_onh":
                    # Для OneHotEncoder также передаем все столбцы
                    self.scaler[key].fit(X[self.mask_scale[key]])
                else:
                    # Для MinMaxScaler и RobustScaler
                    self.scaler[key].fit(X[self.mask_scale[key]])
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        
        # Важно сохранять порядок преобразований
        order = ["enc_ord", "mm", "rb", "enc_onh"]
        
        for key in order:
            if self.mask_scale[key]:
                if key == "enc_onh":
                    # OneHotEncoder создает новые столбцы
                    transformed = self.scaler[key].transform(X_transformed[self.mask_scale[key]])
                    # Получаем имена новых столбцов
                    feature_names = self.scaler[key].get_feature_names_out(self.mask_scale[key])
                    # Создаем DataFrame с новыми столбцами
                    transformed_df = pd.DataFrame(
                        transformed, 
                        columns=feature_names, 
                        index=X_transformed.index
                    )
                    # Удаляем исходные столбцы
                    X_transformed = X_transformed.drop(columns=self.mask_scale[key])
                    # Добавляем новые столбцы
                    X_transformed = pd.concat([X_transformed, transformed_df], axis=1)
                else:
                    # Для остальных преобразований заменяем значения в существующих столбцах
                    X_transformed[self.mask_scale[key]] = self.scaler[key].transform(
                        X_transformed[self.mask_scale[key]]
                    )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, with_support_calls: bool = True) -> pd.DataFrame:
        self.fit(X=X, with_support_calls=with_support_calls)
        return self.transform(X=X)

def age_category(X: pd.DataFrame):
    def change(age: int):
        if age <= 19:
            return "Children"
        elif age <= 34:
            return "Youth"
        elif age <= 54:
            return "Middle"
        else:
            return "Elderly"
                
    X["Age"] = X["Age"].apply(change)
    X["Age"] = X["Age"].astype("category")
    
    return X

def without_blowouts(X: pd.DataFrame, Y: pd.Series, method: str = "intersection"):
    full_index = list()
    
    if (method != "intersection") and (method != "quantile"):
        raise AttributeError
    
    for column in X.columns:
        if (X[column].dtype == "category") or (X[column].dtype == "object"):
            continue
        
        index = EDA.find_blowouts(X[column], method_searching=method)
        full_index.extend(index)
        
    full_index = set(full_index)
    
    return X.drop(full_index, axis=0), Y.drop(full_index, axis=0)

def apply_lin_robust_scaler(X: pd.Series, method: str = "robust") -> pd.Series:
    match(method.lower()):
        case "robust":
            rb_scaler = RobustScaler()
            X_scaled = rb_scaler.fit_transform(X.values.reshape(-1, 1))
            return pd.Series(X_scaled.flatten(), index=X.index)
        case "linspace":
            return pd.Series(np.linspace(start=0, stop=1, num=len(X)), index=X.index)
        case _:
            raise AttributeError

def full_train(model: DecisionTreeClassifier, X: pd.DataFrame, Y: pd.Series, with_support_calls: bool = True) -> Tuple[float]:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=47)
    
    scaler = private_scaler()
    X_train = scaler.fit_transform(X=X_train, with_support_calls=with_support_calls)
    X_test = scaler.transform(X=X_test)
    
    model.fit(X_train, y_train)
    score1 = model.best_estimator_.score(X=X_test, y=y_test)
    
    return score1 

def combinate_all_batches(X: pd.DataFrame, Y: pd.Series):
    main_batches = ["O_!B", "O_B"]
    combo1_batches = ["AI", "AC"]
    combo2_batches = ["LN", "RB"]
    
    result = dict()
    
    param_grid = {
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [2, 5, 10],
        "min_samples_split": [5, 10, 20]
    }
    
    for main in main_batches:
        match(main):
            case "O_!B":
                model = GridSearchCV(
                    estimator=DecisionTreeClassifier(random_state=47),
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1
                )
                X_, Y_ = without_blowouts(X=X, Y=Y, method="quantile")
                result[main] = full_train(model=model, X=X_, Y=Y_)
                del model
            
            case "O_B":
                model = GridSearchCV(
                    estimator=DecisionTreeClassifier(random_state=47),
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1
                )
                X_, Y_ = X, Y
                result[main] = full_train(model=model, X=X_, Y=Y_)
                del model
        
        for cmb1 in combo1_batches:
            if cmb1 == "AC":
                X_ = age_category(X_.copy())  # Создаем копию, чтобы не менять исходный DataFrame
                
            for cmb2 in combo2_batches:
                model = GridSearchCV(
                    estimator=DecisionTreeClassifier(random_state=47),
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1
                )
                match(cmb2):
                    case "LN":
                        X_temp = X_.copy()
                        X_temp["SupportCalls"] = apply_lin_robust_scaler(X_temp["SupportCalls"], method="linspace")
                        result[f'{main}_{cmb1}_{cmb2}'] = full_train(model=model, X=X_temp, Y=Y_, with_support_calls=False)
                    case "RB":
                        X_temp = X_.copy()
                        X_temp["SupportCalls"] = apply_lin_robust_scaler(X_temp["SupportCalls"], method="robust")
                        result[f'{main}_{cmb1}_{cmb2}'] = full_train(model=model, X=X_temp, Y=Y_, with_support_calls=False)
                del model
    
    return result

if __name__ == "__main__":
    try:
        df = pd.read_csv("D:\\Customer churn\\churn_synthetic_dataset.csv")
    except:
        df = pd.read_csv("E:\\Customer churn\\churn_synthetic_dataset.csv")

    df = df.drop(['CustomerID'], axis=1)
    df[["Gender", "ContractType", "InternetService", "PaymentMethod", "Churn", "HasDependents", "PaperlessBilling", "SupportCalls"]] = df[["Gender", "ContractType", "InternetService", "PaymentMethod", "Churn", "HasDependents", "PaperlessBilling", "SupportCalls"]].astype("category")

    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    imputer = MyImputerMissing()
    imputer.fit(dataset=X)
    X = imputer.impute()

    scores = combinate_all_batches(X=X, Y=y)
    for key, value in scores.items():
        print(f'{key} : {value}')