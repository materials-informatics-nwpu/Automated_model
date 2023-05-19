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

# Read data that has been processed for features
data_set_re = pd.read_csv("data.csv")
data_set_re = data_set_re.iloc[:,1:]
train_inf = np.isinf(data_set_re)
data_set_re[train_inf] = 0

mae_list = []
mse_list = []
r2_list = []
mape_list = []

# 循环
for i in range(5):
    kf = KFold(n_splits = 10, shuffle=True)   

    mae_per_list = []
    mse_per_list = []
    r2_per_list = []
    mape_per_list = []
    #'Target feature' refers to the label of the target feature
    for train_data, test_data in kf.split(data_set_re):   
        train_data = data_set_re.iloc[train_data,:]  
        test_data = data_set_re.iloc[test_data,:]         

        predictor_re = TabularPredictor(label='ln_life').fit(train_data, num_stack_levels=2, num_bag_folds=3)  # Fit models for 120s
        result_test = predictor_re.predict(test_data)
        result_train = predictor_re.predict(train_data)

        results = predictor_re.fit_summary(show_plot=True)
        feature_imp = predictor_re.feature_importance(data_set_re)
        feature_imp.to_csv("importance.csv")
        print(feature_imp)

        MAE = mae(test_data.loc[:,'ln_life'],result_test)
        MSE = mse(test_data.loc[:,'ln_life'],result_test)
        R2 = r2(test_data.loc[:,'ln_life'],result_test)
        MAPE = mape(test_data.loc[:,'ln_life'],result_test)

        mae_per_list.append(MAE)
        mse_per_list.append(MSE)
        r2_per_list.append(R2)
        mape_per_list.append(MAPE)
        
    print("第{}次循环各项均值为  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(i+1, mean(mse_per_list), mean(mae_per_list), mean(r2_per_list), mean(mape_per_list)))      
    
    mae_list.append(mean(mae_per_list))
    mse_list.append(mean(mse_per_list))
    r2_list.append(mean(r2_per_list))
    mape_list.append(mean(mape_per_list))

#Output various predictive performance indicators
print("各项均值为  MSE:{}, MAE:{}, R2:{}, MAPE:{}".format(mean(mse_list), mean(mae_list), mean(r2_list), mean(mape_list)))

#Output predicted scatter plot
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(result_train, train_data["ln_life"],"o",color='blue', label='Train');
ax.plot(result_test, test_data["ln_life"],"o",color='orange', label='Test');
plt.plot([3,12], [3,12], '--');
plt.xlim((3,12))
plt.ylim((3,12))
plt.title('AutoGluon',fontsize=14,fontproperties='Times New Roman',fontweight='bold')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.legend(loc="upper left")
plt.tick_params(labelsize=12)
plt.yticks(fontproperties='Times New Roman', size=12)#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=12)
plt.ylabel('Prediction',weight='bold',fontproperties='Times New Roman',fontsize=14)
plt.xlabel('Truth',weight='bold',fontproperties='Times New Roman',fontsize=14)
plt.savefig('AutoGluon.png',dpi=500, bbox_inches='tight')


