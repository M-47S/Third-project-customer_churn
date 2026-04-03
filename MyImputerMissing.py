import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from typing import Union, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import time
from Logging import Logging
import Errors as er

class MyImputerMissing():
#~~~~~~~~~~~~~~~~~PRIVATE~~~~~~~~~~~~~~~~~#
    def __init__(self):
        self.__models = {}
        self.__encoders = {}
        self.__feature_names = {}
        self.__log_system = Logging()
    
    def __highlight_the_features(self) -> Dict:
        self.__log_system.push_log("Начало выделения признаков по типам")
        
        dict_of_features = dict()
        dict_of_features["Numerical"] = []
        dict_of_features["Categorical"] = []
        dict_of_features["Missing"] = []

        pf = self.__dataset.isna().sum()
        dict_of_features["Missing"] = np.array(pf[pf != 0].index)
        del pf

        for feature in self.__dataset.dtypes.index:
            if self.__dataset.dtypes[feature] == "category":
                dict_of_features["Categorical"].append(feature)
            else:
                dict_of_features["Numerical"].append(feature)

        dict_of_features["Categorical"] = np.array(dict_of_features["Categorical"])
        dict_of_features["Numerical"] = np.array(dict_of_features["Numerical"])
        
        self.__log_system.push_log(f"Выделено признаков: числовых - {len(dict_of_features['Numerical'])}, "
                                  f"категориальных - {len(dict_of_features['Categorical'])}, "
                                  f"с пропусками - {len(dict_of_features['Missing'])}")
        
        return dict_of_features
    
    def __get_prepared_data_for_train(self, y_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        self.__log_system.push_log(f"Подготовка данных для обучения модели признака {y_name}")
        
        X = self.__dataset.drop([y_name], axis=1)
        Y = self.__dataset[y_name]

        # Удаляем строки, где целевая переменная отсутствует
        Y_missing = Y.isna()
        index_of_missing = Y_missing[Y_missing == True].index
        X = X.drop(index_of_missing, axis=0)
        Y = Y.drop(index_of_missing, axis=0)
        
        self.__log_system.push_log(f"Для признака {y_name}: X.shape = {X.shape}, Y.shape = {Y.shape}")
        
        return X, Y
    
    def __prepare_features(self, X: pd.DataFrame, y_name: str, fit: bool = False) -> np.ndarray:
        """Подготовка признаков с использованием OneHotEncoder"""
        try:
            # Исключаем целевую переменную из списков признаков
            categorical_cols = [col for col in self.__dict_of_features["Categorical"] 
                              if col in X.columns and col != y_name]
            numerical_cols = [col for col in self.__dict_of_features["Numerical"] 
                            if col in X.columns and col != y_name]
            
            if fit:
                self.__log_system.push_log(f"Обучение OneHotEncoder для признака {y_name}")
                # Создаем и обучаем новый encoder
                self.__encoders[y_name] = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                
                if len(categorical_cols) > 0:
                    # Обучаем encoder на категориальных признаках
                    encoded_cat = self.__encoders[y_name].fit_transform(X[categorical_cols])
                    # Сохраняем имена признаков
                    self.__feature_names[y_name] = list(self.__encoders[y_name].get_feature_names_out(categorical_cols))
                else:
                    encoded_cat = np.empty((X.shape[0], 0))
                    self.__feature_names[y_name] = []
                
                if len(numerical_cols) > 0:
                    encoded_num = X[numerical_cols].values
                    self.__feature_names[y_name].extend(numerical_cols)
                else:
                    encoded_num = np.empty((X.shape[0], 0))
                
                result = np.hstack([encoded_cat, encoded_num]) if encoded_cat.shape[1] > 0 or encoded_num.shape[1] > 0 else np.empty((X.shape[0], 0))
                
                self.__log_system.push_log(f"Размерность подготовленных признаков для {y_name}: {result.shape}")
                return result
            else:
                self.__log_system.push_log(f"Применение OneHotEncoder для признака {y_name}")
                
                if y_name not in self.__encoders:
                    raise er.FEATURE_IS_NOT_EXIST(f"Encoder для признака {y_name} не обучен!")
                
                if len(categorical_cols) > 0:
                    encoded_cat = self.__encoders[y_name].transform(X[categorical_cols])
                else:
                    encoded_cat = np.empty((X.shape[0], 0))
                
                if len(numerical_cols) > 0:
                    encoded_num = X[numerical_cols].values
                else:
                    encoded_num = np.empty((X.shape[0], 0))
                
                result = np.hstack([encoded_cat, encoded_num]) if encoded_cat.shape[1] > 0 or encoded_num.shape[1] > 0 else np.empty((X.shape[0], 0))
                return result
                
        except Exception as e:
            self.__log_system.push_log(f"Ошибка при подготовке признаков для {y_name}: {str(e)}")
            raise
    
    def __train_models(self) -> bool:
        self.__dict_of_features = self.__highlight_the_features()
        
        if len(self.__dict_of_features["Missing"]) == 0:
            self.__log_system.push_log("Нет признаков с пропущенными значениями для импутации")
            return False
        
        param_grid_reg = {
            "criterion": ["squared_error", "absolute_error"],
            "min_samples_leaf": [5, 10],
            "min_samples_split": [10, 20]
        }
                          
        param_grid_class = {
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [5, 10],
            "min_samples_split": [10, 20]
        }
        
        start_time = time.time()
        
        for miss in self.__dict_of_features["Missing"]:
            try:
                self.__log_system.push_log(f"Начало обучения модели для признака: {miss}")
                
                # Получаем данные для обучения
                X_raw, Y = self.__get_prepared_data_for_train(y_name=miss)
                
                # Подготавливаем признаки (с обучением encoder)
                X = self.__prepare_features(X_raw, miss, fit=True)
                
                # Проверяем, что есть данные для обучения
                if X.shape[0] == 0:
                    self.__log_system.push_log(f"Нет данных для обучения модели для признака {miss}")
                    continue
                
                self.__log_system.push_log(f"Размерность X для {miss}: {X.shape}")
                
                # Выбираем тип модели в зависимости от типа признака
                if miss in self.__dict_of_features["Numerical"]:
                    self.__models[miss] = GridSearchCV(
                        estimator=DecisionTreeRegressor(random_state=42),
                        param_grid=param_grid_reg,
                        cv=3,
                        n_jobs=-1
                    )
                    model_type = "DecisionTreeRegressor"
                else:
                    self.__models[miss] = GridSearchCV(
                        estimator = DecisionTreeClassifier(random_state=42),
                        param_grid = param_grid_class,
                        cv=3,
                        n_jobs=-1
                    )
                    model_type = "DecisionTreeClassifier"
                
                # Обучаем модель
                self.__log_system.push_log(f"Обучение {model_type} для признака {miss}")
                self.__models[miss].fit(X, Y)
                
                self.__log_system.push_log(f"Лучшие параметры для {miss}: {self.__models[miss].best_params_}")
                self.__log_system.push_log(f"Завершено обучение модели для {miss}")
                
            except Exception as e:
                self.__log_system.push_log(f"Ошибка при обучении модели для {miss}: {str(e)}")
                continue
        
        end_time = time.time()
        training_time = round(end_time - start_time, 2)
        self.__log_system.push_log(f"Обучение 'MyImputer': {training_time} сек")
        
        return True
    
    def __get_index_missing_rows(self, y_name: Union[str, None] = None) -> Union[Dict, np.ndarray]:
        try:
            if not hasattr(self, '_MyImputerMissing__dict_of_features'):
                raise er.FEATURE_IS_NOT_EXIST("Сначала вызовите метод fit() для инициализации признаков!")
            
            misses = self.__dataset[self.__dict_of_features["Missing"]].isna()

            if y_name is None:
                dict_of_misses = dict()
                for miss in self.__dict_of_features["Missing"]:
                    index_of_misses = misses[misses[miss] == True].index
                    dict_of_misses[miss] = index_of_misses
                
                return dict_of_misses
            elif y_name not in self.__dict_of_features["Missing"]:
                raise er.FEATURE_IS_NOT_EXIST(f"Обращение к признаку {y_name} невозможно! Признак отсутствует в списке признаков с пропусками.")
            else:
                return misses[misses[y_name] == True].index
                
        except Exception as e:
            self.__log_system.push_log(f"Ошибка при получении индексов пропущенных строк: {str(e)}")
            raise
        
#~~~~~~~~~~~~~~~~~~PUBLIC~~~~~~~~~~~~~~~~~~#
    def fit(self, dataset: pd.DataFrame) -> bool:
        try:
            self.__log_system.push_log("Начало обучения MyImputerMissing")
            self.__log_system.push_log(f"Размер входных данных: {dataset.shape}")
            
            # Сохраняем копию датасета
            self.__dataset = dataset.copy().fillna(np.nan)
            
            # Обучаем модели
            result = self.__train_models()
            
            if result:
                self.__log_system.push_log("Обучение MyImputerMissing завершено успешно")
            else:
                self.__log_system.push_log("Обучение MyImputerMissing завершено: нет признаков для импутации")
            
            return result
            
        except Exception as e:
            self.__log_system.push_log(f"Критическая ошибка в методе fit(): {str(e)}")
            raise
        
    def impute(self) -> pd.DataFrame:
        try:
            self.__log_system.push_log("Начало процесса импутации пропущенных значений")
            
            if not hasattr(self, '_MyImputerMissing__dataset'):
                raise ValueError("Сначала вызовите метод fit() для обучения моделей!")
            
            if not self.__models:
                self.__log_system.push_log("Нет обученных моделей для импутации. Возвращается исходный датасет.")
                return self.__dataset.copy()
            
            dataset_imputed = self.__dataset.copy()
            index_missing_rows = self.__get_index_missing_rows()
            
            total_imputed = 0
            
            for miss in self.__dict_of_features["Missing"]:
                if miss not in self.__models:
                    self.__log_system.push_log(f"Модель для признака {miss} не обучена, пропускаем")
                    continue
                
                # Получаем индексы пропущенных значений
                miss_indices = index_missing_rows[miss]
                
                if len(miss_indices) == 0:
                    self.__log_system.push_log(f"Нет пропущенных значений для признака {miss}")
                    continue
                
                self.__log_system.push_log(f"Импутация {len(miss_indices)} значений для признака {miss}")
                
                try:
                    # Получаем данные для предсказания
                    X_raw = self.__dataset.loc[miss_indices].drop(miss, axis=1)
                    
                    # Подготавливаем признаки (без обучения, используем сохраненный encoder)
                    X = self.__prepare_features(X_raw, miss, fit=False)
                    
                    # Предсказываем
                    y_impute = self.__models[miss].predict(X)
                    
                    # Заполняем пропущенные значения
                    dataset_imputed.loc[miss_indices, miss] = y_impute
                    
                    total_imputed += len(miss_indices)
                    self.__log_system.push_log(f"Успешно импутировано {len(miss_indices)} значений для признака {miss}")
                    
                except Exception as e:
                    self.__log_system.push_log(f"Ошибка при импутации признака {miss}: {str(e)}")
                    continue
            
            self.__log_system.push_log(f"Импутация завершена. Всего импутировано значений: {total_imputed}")
            
            return dataset_imputed
            
        except Exception as e:
            self.__log_system.push_log(f"Критическая ошибка в методе impute(): {str(e)}")
            raise

# Тестовый код
if __name__ == "__main__":
    try:
        # Пробуем загрузить ваш датасет
        try:
            df = pd.read_csv("D:\\Customer churn\\churn_synthetic_dataset.csv")
        except:
            df = pd.read_csv("E:\\Customer churn\\churn_synthetic_dataset.csv")
            
        df = df.drop(['CustomerID'], axis=1)
        cat_cols = ["Gender", "ContractType", "InternetService", "PaymentMethod"]
        df[cat_cols] = df[cat_cols].astype("category")
        
        X_miss = df.drop(["Churn"], axis=1)
        
        print("=" * 50)
        print("ИСХОДНЫЕ ДАННЫЕ")
        print("=" * 50)
        print(df.info())
        print("\nПропущенные значения:")
        print(df.isna().sum(axis=0))
        
        print("\n" + "=" * 50)
        print("НАЧАЛО ИМПУТАЦИИ")
        print("=" * 50)
        
        imputer = MyImputerMissing()
        imputer.fit(X_miss)
        df_imputed = imputer.impute()
        
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ ИМПУТАЦИИ")
        print("=" * 50)
        print(df_imputed.info())
        print("\nПропущенные значения после импутации:")
        print(df_imputed.isna().sum(axis=0))
        
    except FileNotFoundError:
        print("Файл не найден. Создаю тестовый датасет...")
        
        # Создаем тестовый датасет
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 70, n_samples),
            'ContractType': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 100, n_samples),
            'TotalCharges': np.random.uniform(100, 5000, n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'Tenure': np.random.randint(1, 72, n_samples),
            'Churn': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Добавим пропущенные значения
        mask = np.random.rand(n_samples) < 0.1
        df.loc[mask, 'Age'] = np.nan

        mask = np.random.rand(n_samples) < 0.15
        df.loc[mask, 'MonthlyCharges'] = np.nan

        mask = np.random.rand(n_samples) < 0.05
        df.loc[mask, 'TotalCharges'] = np.nan
        
        # Преобразуем категориальные признаки
        cat_cols = ["Gender", "ContractType", "InternetService", "PaymentMethod"]
        df[cat_cols] = df[cat_cols].astype("category")
        
        X_miss = df.drop(["Churn"], axis=1)
        
        print("\n" + "=" * 50)
        print("НАЧАЛО ИМПУТАЦИИ (ТЕСТОВЫЙ ДАТАСЕТ)")
        print("=" * 50)
        
        imputer = MyImputerMissing()
        imputer.fit(X_miss)
        df_imputed = imputer.impute()
        
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ ИМПУТАЦИИ")
        print("=" * 50)
        print(df_imputed.info())
        print("\nПропущенные значения после импутации:")
        print(df_imputed.isna().sum(axis=0)) 