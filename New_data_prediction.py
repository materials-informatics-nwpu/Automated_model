import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_percentage_error as mape
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from statistics import mean
import numpy as np



# ========== config begin ==========
train_file = "aaa111.csv"
data_file = "new_data_file.csv"
output_file = "all_data.csv"
# ========== config end ============


# 待预测数据 加载
data = pd.read_csv(data_file)
# 训练数据 加载
data_set_re = pd.read_csv(train_file)

data_set_re = data_set_re.iloc[:,1:]
train_inf = np.isinf(data_set_re)
data_set_re[train_inf] = 0

model = TabularPredictor(label='ln_creep_life').fit(data_set_re, num_stack_levels=2, num_bag_folds=3)
result_pre = model.predict(data)
data["ln_creep_life"] = result_pre

data.to_csv(output_file, index=False)







