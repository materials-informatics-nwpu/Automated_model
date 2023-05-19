#auto-sklearn 2.0 0.15.0
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.svm import LinearSVR
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import KFold
from statistics import mean

import autosklearn.regression

warnings.filterwarnings("ignore")

# 读取数据集
df = pd.read_csv("266+8.csv")
df = df.iloc[:,1:]
train_inf = np.isinf(df)
df[train_inf] = 0

# 分离特征变量和目标变量
X = df.drop("ln_life", axis=1).values
y = df["ln_life"]


mae_list = []
mse_list = []
r2_list = []
mape_list = []

temp_folder = "/tmp/autosklearn_regression_example_tmp"
if os.path.exists(temp_folder):
    os.remove(temp_folder)
    


for i in range(1):
    # 删除已存在的 temp 目录
    temp_folder = "/tmp/autosklearn_regression_example_tmp"
    if os.path.exists(temp_folder):
        os.remove(temp_folder)
        
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # AutoSklearnRegressor
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder=temp_folder,
    )
    
    automl.fit(X_train, y_train)

    # 计算训练集和测试集内的预测结果
    y_train_pred = automl.predict(X_train)
    y_test_pred = automl.predict(X_test)

    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)
     

    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    mape_list.append(mape)

    print("AutoSklearnRegressor 第{}次循环,  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mse, mae, r2, mape))

print("AutoSklearnRegressor 各项均值为  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))
print("")


# 绘图
fig, ax = plt.subplots(figsize=(6,6))

# fontManager.addfont("./times.ttf")
plt.rcParams['font.sans-serif'] = "Times New Roman"


ax.plot(y_test_pred, y_test,"o", color='orange', label='test');
ax.plot(y_train_pred, y_train,"o",color='blue', label='train');
plt.plot([3, 9], [3, 9], '--');
plt.xlim((3, 9))
plt.ylim((3, 9))
ax.set_ylabel('ln(Prediction creep rupture life)(h)',fontproperties="Times New Roman",fontsize=14)
ax.set_xlabel('ln(Measured creep rupture life)(h)',fontproperties="Times New Roman",fontsize=14)
ax.tick_params(labelsize=12)
plt.yticks(fontproperties="Times New Roman", size=12)#设置大小及加粗
plt.xticks(fontproperties="Times New Roman", size=12)
ax.set_title('AutoSklearn Regressor', fontproperties="Times New Roman",fontsize=14,fontweight='bold')
ax.legend(loc='best')

plt.savefig('AutoSklearnRegressor.png',dpi=500, bbox_inches='tight')
plt.show()


