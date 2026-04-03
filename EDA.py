import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Dict
from copy import deepcopy
import os
from math import log2, log10, log
from matplotlib.colors import CSS4_COLORS
from random import randint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from MyImputerMissing import MyImputerMissing

class EDA:
    def __init__(self, figsize:Tuple[int] = (10,4)):
        self.fig = plt.figure(figsize = figsize)
        self.colors = list(CSS4_COLORS.keys())

    def create_hist(self, data:Union[pd.Series, List, Tuple], show_quantile: bool = False, bins: int = None, color:str = None, xlabel:Union[str, float, int] = None, ylabel:Union[str, float, int] = None, download: bool = False, fpath: str = None) -> Axes:   
        ax = self.fig.add_subplot(1,1,1)

        if (data.dtype == "category") or (data.dtype == "object"):
            X = data.value_counts().index
            height = data.value_counts().to_numpy()
        
            if color is None:
                color = self.colors[:len(X)]

            ax.bar(x = X, height = height, color = color, edgecolor = "black")

        else:
            if color is None:
                color = self.colors[randint(0, len(self.colors))]

            if bins is None:
                bins = int(log2(len(data))/log2(1.24255))

            ax.hist(data, color = color, edgecolor = "black", bins = bins)
        
        if not(xlabel is None):
            ax.xaxis.set_label_text(xlabel)
            
        if not(ylabel is None):
            ax.yaxis.set_label_text(ylabel)
        
        if (show_quantile == True) and (data.dtype != "category"):
            ax.axvline(x=data.quantile(0.01), color = "blue", linestyle = "--", alpha = 1)
            ax.axvline(x=data.median(), color = "red", linestyle = "--", alpha = 1, label = "median")
            ax.axvline(x=data.mean(), color = "green", linestyle = "--", alpha = 0.8, label = "mean")
            ax.axvline(x=data.quantile(0.99), color = "blue", linestyle = "--", alpha = 1, label = "0.01 -> 0.99 quantile")

        self.fig.legend()

        if download == True:
            directory = os.getcwd()

            if not(fpath is None):
                self.fig.savefig(fname=f'{fpath}')
                print(f'Сохранено в "{fpath}"')

            else:
                if not(xlabel is None) and not(ylabel is None):
                    self.fig.savefig(fname=f'{directory}\\{xlabel}_{ylabel}.png')
                    print(f'Сохранено "{directory}\\{xlabel}_{ylabel}.png"')

                elif not(xlabel is None):
                    self.fig.savefig(fname=f'{directory}\\X_{xlabel}.png')
                    print(f'Сохранено "{directory}\\X_{xlabel}.png"')

                elif not(ylabel is None):
                    self.fig.savefig(fname=f'{directory}\\Y_{ylabel}.png')
                    print(f'Сохранено "{directory}\\Y_{ylabel}.png"')
                else:
                    self.fig.savefig(fname=f'{directory}\\unknown.png')
                    print(f'Сохранено "{directory}\\unknown.png"')
        ax_ = deepcopy(ax)
        self.fig.clf()
        return ax_
    
    def create_boxplot(self, data:Union[pd.Series, np.ndarray[float]], vert: bool = True, box_name: str = "Unnamed", xlabel:Union[str, float, int] = None, ylabel:Union[str, float, int] = None, download: bool = False, fpath: str = None) -> Axes:
        ax = self.fig.add_subplot(1,1,1)

        if data.dtype != "category":
            ax.boxplot(data.to_numpy(), vert = vert, labels = [box_name])

        if not(xlabel is None):
            ax.xaxis.set_label_text(xlabel)
            
        if not(ylabel is None):
            ax.yaxis.set_label_text(ylabel)

        if download == True:
            directory = os.getcwd()

            if not(fpath is None):
                self.fig.savefig(fname=f'{fpath}')
                print(f'Сохранено в "{fpath}"')

            else:
                if not(xlabel is None) and not(ylabel is None):
                    self.fig.savefig(fname=f'{directory}\\{xlabel}_{ylabel}.png')
                    print(f'Сохранено "{directory}\\{xlabel}_{ylabel}.png"')

                elif not(xlabel is None):
                    self.fig.savefig(fname=f'{directory}\\X_{xlabel}.png')
                    print(f'Сохранено "{directory}\\X_{xlabel}.png"')

                elif not(ylabel is None):
                    self.fig.savefig(fname=f'{directory}\\Y_{ylabel}.png')
                    print(f'Сохранено "{directory}\\Y_{ylabel}.png"')
                else:
                    self.fig.savefig(fname=f'{directory}\\unknown.png')
                    print(f'Сохранено "{directory}\\unknown.png"')


        ax_ = deepcopy(ax)
        self.fig.clf()
        return ax_

    def make_analysis_of_distribution(self, data: Union[pd.DataFrame, pd.Series], show_quantile: bool = False, bins: int = None, download:bool = False, path_to_dir: Union[str, None] = None) -> List[Axes]:
        axes_ = list()

        if isinstance(data, pd.Series):
            try:
                x_label = data.name
            except:
                x_label = None
                
            axes_.append(self.create_hist(data = data, show_quantile = show_quantile, xlabel = x_label, ylabel = "Frequency", download = download, fpath = f'{path_to_dir}\\{x_label}_Frequency.png'))
        else:
            for column in data.columns:
                axes_.append(self.create_hist(data = data[column], show_quantile = show_quantile, xlabel = column, ylabel = "Frequency", download = download, fpath = f'{path_to_dir}\\{column}_Frequency.png'))
            
        return axes_

    def make_analysis_of_blowout(self, data: pd.DataFrame, method_searching: str = "IQR", download: bool = False, path_to_dir: Union[str, None] = None) -> Tuple[List[Axes], Dict[str, List[int]]]:
        axes_ = list()
        blowouts_ = dict()

        for column in data.columns:
            if data[column].dtype != "category":
                ax = self.create_boxplot(data = data[column], xlabel = column, ylabel = "Value", box_name = "", download = download, fpath = f'{path_to_dir}\\{column}_Value.png')
                axes_.append(ax)
                blowouts_[column] = self.find_blowouts(data = data[column], method_searching = method_searching)
            else:
                continue

        return axes_, blowouts_ 
    
    @staticmethod
    def find_blowouts(data: Union[pd.Series, List, Tuple], method_searching: str = "IQR") -> List[int]:
        def iqr_method():
            quant_25 = data.quantile(0.25)
            quant_75 = data.quantile(0.75)

            iqr = abs(quant_75 - quant_25)*1.5
            blowouts = data[(data < (quant_75 - iqr)) | (data > (quant_25 + iqr))].index

            return blowouts
        
        def quantile_method():
            quant_99 = data.quantile(0.99)
            quant_01 = data.quantile(0.01)
            blowouts = data[(data < quant_01) | (data > quant_99)].index

            return blowouts
        
        def sigma3_method(): 
            std3 = 3*data.std(axis = 0)
            mean = data.mean(axis = 0)

            blowouts = data[(data < (mean - std3)) | (data > (mean + std3))].index

            return blowouts

        if (isinstance(data, list) or isinstance(data, tuple)):
            data = pd.Series(data=data)
        
        match(method_searching.upper()):
                case "IQR":
                    blowouts = iqr_method()

                case "QUANTILE":
                    blowouts = quantile_method()
                
                case "SIGMA3":
                    blowouts = sigma3_method()
                
                case "UNION":
                    blowouts_iqr = set(iqr_method())
                    blowouts_qnt = set(quantile_method())
                    blowouts_sgm = set(sigma3_method())

                    blowouts = (blowouts_iqr | blowouts_qnt) | blowouts_sgm

                case "INTERSECTION":
                    blowouts_iqr = set(iqr_method())
                    blowouts_qnt = set(quantile_method())
                    blowouts_sgm = set(sigma3_method())

                    blowouts = (blowouts_iqr & blowouts_qnt) & blowouts_sgm

                case _:
                    raise AttributeError(name = f'Value "{method_searching}" is not access!')

        return sorted(list(blowouts))
    
    @staticmethod
    def get_stats_about_blowouts(data: pd.DataFrame) -> pd.DataFrame:
        methods = ["IQR", "Quantile", "Sigma3", "Union", "Intersection"]
        table_of_stats = dict()
        for column in data.columns:
            if (data[column].dtype != "category") and (data[column].dtype != "object"):
                table_of_stats[column] = list()
                
                for method in methods:
                    table_of_stats[column].append(len(EDA.find_blowouts(data = data[column], method_searching = method)))
                
        return pd.DataFrame(data = table_of_stats, index = methods)
    
    @staticmethod
    def log_data(data: pd.Series, base: float = 10) -> pd.Series:
        return data.apply(func=log, args=tuple([base]))
        
if __name__ == "__main__":
    # print(f"Запущен файл {os.path.basename(sys.argv[0])}")
    try:
        df = pd.read_csv("D:\\Customer churn\\churn_synthetic_dataset.csv")
    except:
        df = pd.read_csv("E:\\Customer churn\\churn_synthetic_dataset.csv")

    df = df.drop(['CustomerID'], axis=1)
    df[["Gender", "ContractType", "InternetService", "PaymentMethod", "Churn", "HasDependents", "PaperlessBilling", "SupportCalls"]] = df[["Gender", "ContractType", "InternetService", "PaymentMethod", "Churn", "HasDependents", "PaperlessBilling", "SupportCalls"]].astype("category")
    
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    imputer = MyImputerMissing()
    imputer.fit(dataset = X)
    df = imputer.impute()

    eda = EDA()
    spent = df["Income"]
    lg_spent = eda.log_data(spent, base=10)
    eda.make_analysis_of_distribution(spent, show_quantile = True)
    eda.make_analysis_of_distribution(lg_spent, show_quantile = True)
   
    