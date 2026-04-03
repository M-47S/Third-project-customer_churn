from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

class MyRegression(LogisticRegression):
    def __init__(self, threshold = 0.5, penalty = "l2", *, dual = False, tol = 0.0001, C = 1, fit_intercept = True, intercept_scaling = 1, class_weight = None, random_state = None, solver = "lbfgs", max_iter = 100, multi_class = "auto", verbose = 0, warm_start = False, n_jobs = None, l1_ratio = None):
        self.threshold = threshold
        super().__init__(penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)
        
    def predict(self, X):
        probabilities = super().predict_proba(X=X)
        result = pd.Series(probabilities[:, 1])
        
        result[probabilities[:, 1] <= self.threshold] = 0
        result[probabilities[:, 1] > self.threshold] = 1

        result = result.to_numpy()
        
        return result.astype(int)

    def get_params(self, deep=True):
        # Получаем параметры родителя
        params = super().get_params(deep=deep)
        # Добавляем наш параметр threshold
        params['threshold'] = self.threshold
        return params
    
    def set_params(self, **params):
        if 'threshold' in params.keys():
            self.threshold = params["threshold"]
            params.__delitem__("threshold")
        
        super().set_params(**params)
        return self