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
output_file = "new_data_file.csv"
# ========== config end ============




def create_new_data():
    data = pd.DataFrame(data=None,columns=["Ni", "Re", "Co", "Al", "Ti", "W", "Mo", "Cr", "Ta","C", "B", "Y",
                                           "solution_treatment_time", 
                                           "first_step_aging_treatment_time",
                                           "second_step_aging_treatment_time",
                                           "solution_treatment_temperature", 
                                           "first_step_aging_treatment_temperature",
                                           "second_step_aging_treatment_temperature",
                                           "temperature",
                                           "applied_Stress"])
                                           
    return data


data = create_new_data()
data.to_csv(output_file, index=False)


#生成材料搜索空间
i = 0
j = 0
for Re in np.linspace(3.1, 6.5, 3):
    for Co in np.linspace(0.1, 8.5, 3):
        for Al in np.linspace(5.2, 7.5, 3):
            for Ti in np.linspace(0.5, 3.68, 3):
                for W in np.linspace(3.5, 12.2, 3):
                    for Mo in np.linspace(0.5, 3.65, 3):
                        for Cr in np.linspace(5.5, 10.6, 3):
                            for Ta in np.linspace(1.5, 11.5, 3):
                                for C in np.linspace(0, 0.1, 3):
                                    for B in np.linspace(0.004, 0.05, 3):
                                        for Y in np.linspace(0.004, 0.05, 3):
                                            
                                                    if len(data) > 30000:
                                                        j += 1
                                                        # data = data/10
                                                        data.to_csv(output_file, mode='a', header=False, index=False)
                                                        data = create_new_data()
                                                        print(j, i)
                                                        #if j > 5:
                                                            #exit()

                                                    for solution_treatment_time in np.linspace(1, 4, 4):
                                                       for first_step_aging_treatment_time in np.linspace(1, 6, 3):
                                                            for second_step_aging_treatment_time in np.linspace(16, 32, 3):
                                                                for solution_treatment_temperature in np.linspace(1100, 1350, 3):
                                                                    for first_step_aging_treatment_temperature in np.linspace(1000, 1200, 3):
                                                                        for second_step_aging_treatment_temperature in np.linspace(600, 800, 3):
                                                                            for temperature in np.linspace(900, 900, 1):
                                                                                for applied_Stress in np.linspace(800,800, 1):
                                                                                    one_col = []
                                                                                    one_col.append(100-Re-Co-Al-Ti-W-Mo-Cr-Ta-C-B-Y) # Ni
                                                                                    one_col.append(Re) # Re
                                                                                    one_col.append(Co) # Co
                                                                                    one_col.append(Al) # Al
                                                                                    one_col.append(Ti) # Ti
                                                                                    one_col.append(W) # W
                                                                                    one_col.append(Mo) # Mo
                                                                                    one_col.append(Cr) # Cr
                                                                                    one_col.append(Ta) # Ta
                                                                                    one_col.append(C) # C
                                                                                    one_col.append(B) # B
                                                                                    one_col.append(Y) # Y
                                                                                    one_col.append(solution_treatment_time) # solution_treatment_time
                                                                                    one_col.append(first_step_aging_treatment_time) # first_step_aging_treatment_time
                                                                                    one_col.append(second_step_aging_treatment_time) # second_step_aging_treatment_time
                                                                                    one_col.append(solution_treatment_temperature) # solution_treatment_temperature
                                                                                    one_col.append(first_step_aging_treatment_temperature) # first_step_aging_treatment_temperature
                                                                                    one_col.append(second_step_aging_treatment_temperature) # first_step_aging_treatment_temperature
                                                                                    one_col.append(temperature) # temperature
                                                                                    one_col.append(applied_Stress) # applied_Stress
                                                                                    #print("11111111", len(one_col), one_col)
                                                                                    data.loc[i] = one_col
                                                                                    i = i+1


data = data/100
data.to_csv(output_file, mode='a', header=False, index=False)
print("done")

