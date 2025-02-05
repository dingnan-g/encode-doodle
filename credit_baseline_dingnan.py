# ---------------------------------------------------------------------
# 个贷违约预测
# ---------------------------------------------------------------------
#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""开发人员:丁楠
create_date:20180415
update_date:20200518"""

import warnings
import pandas as pd
warnings.filterwarnings('ignore')

train_bank = pd.read_csv('/home/aistudio/data/data112982/train_public.csv')
train_internet = pd.read_csv('/home/aistudio/data/data112982/train_internet.csv')
train_bank


# In[2]:


test = pd.read_csv('/home/aistudio/data/data112982/test_public.csv')


# ### 数据预处理

# In[3]:


common_cols = []
for col in train_bank.columns:
    if col in train_internet.columns:
        common_cols.append(col)
    else: continue
len(common_cols)


# In[4]:


print(len(train_bank.columns))
print(len(train_internet.columns))


# In[5]:


train_bank_left = list(set(list(train_bank.columns)) - set(common_cols))
train_internet_left = list(set(list(train_internet.columns)) - set(common_cols))

train_bank_left


# In[6]:


train_internet_left


# In[7]:


train1_data = train_internet[common_cols]
train2_data = train_bank[common_cols]
test_data = test[common_cols[:]]


# In[8]:


import datetime

# 日期类型：issueDate，earliesCreditLine
# 转换为pandas中的日期类型
train1_data['issue_date'] = pd.to_datetime(train1_data['issue_date'])
# 提取多尺度特征
train1_data['issue_date_y'] = train1_data['issue_date'].dt.year
train1_data['issue_date_m'] = train1_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train1_data['issue_date_diff'] = train1_data['issue_date'].apply(lambda x: x-base_time).dt.days
train1_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
train1_data.drop('issue_date', axis = 1, inplace = True)


# In[9]:


# 日期类型：issueDate，earliesCreditLine
# 转换为pandas中的日期类型
train2_data['issue_date'] = pd.to_datetime(train2_data['issue_date'])
# 提取多尺度特征
train2_data['issue_date_y'] = train2_data['issue_date'].dt.year
train2_data['issue_date_m'] = train2_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
train2_data['issue_date_diff'] = train2_data['issue_date'].apply(lambda x: x-base_time).dt.days
train2_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
train2_data.drop('issue_date', axis = 1, inplace = True)
train2_data


# In[10]:


employer_type = train1_data['employer_type'].value_counts().index
industry = train1_data['industry'].value_counts().index


# In[11]:


emp_type_dict = dict(zip(employer_type, [0,1,2,3,4,5]))
industry_dict = dict(zip(industry, [i for i in range(15)]))


# In[12]:


train1_data['work_year'].fillna('10+ years', inplace=True)
train2_data['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
train1_data['work_year']  = train1_data['work_year'].map(work_year_map)
train2_data['work_year']  = train2_data['work_year'].map(work_year_map)

train1_data['class'] = train1_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
train2_data['class'] = train2_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

train1_data['employer_type'] = train1_data['employer_type'].map(emp_type_dict)
train2_data['employer_type'] = train2_data['employer_type'].map(emp_type_dict)

train1_data['industry'] = train1_data['industry'].map(industry_dict)
train2_data['industry'] = train2_data['industry'].map(industry_dict)


# In[13]:


# 日期类型：issueDate，earliesCreditLine
#train[cat_features]
# 转换为pandas中的日期类型
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])
# 提取多尺度特征
test_data['issue_date_y'] = test_data['issue_date'].dt.year
test_data['issue_date_m'] = test_data['issue_date'].dt.month
# 提取时间diff
# 设置初始的时间
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
# 转换为天为单位
test_data['issue_date_diff'] = test_data['issue_date'].apply(lambda x: x-base_time).dt.days
test_data[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']]
test_data.drop('issue_date', axis = 1, inplace = True)
test_data['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
test_data['work_year']  = test_data['work_year'].map(work_year_map)
test_data['class'] = test_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
test_data['employer_type'] = test_data['employer_type'].map(emp_type_dict)
test_data['industry'] = test_data['industry'].map(industry_dict)


# ## 模型使用
# 1) LigthGBM
# 2) NN

# ##### 使用internet和bank数据共同特征总量训练

# In[16]:


import lightgbm
from sklearn import metrics

X_train1 = train1_data.drop(['is_default','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train1 = train_internet['is_default']

X_train2 = train2_data.drop(['is_default','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train2 = train_bank['isDefault']

X_train = pd.concat([X_train1, X_train2])
y_train = pd.concat([y_train1, y_train2])
X_train.drop('issue_date', axis = 1, inplace = True)

X_test = test_data.drop(['earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)

# 利用Internet数据预训练模型1
clf_ex=lightgbm.LGBMRegressor(num_leaves=31)
clf_ex.fit(X = X_train, y = y_train)
clf_ex.booster_.save_model('LGBMmode.txt')
pred = clf_ex.predict(X_test)


# In[17]:


# submission
submission = pd.DataFrame({'id':test['loan_id'], 'is_default':pred})
submission.to_csv('submission.csv', index = None)


# #### NN

# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split


# In[19]:


# 数据标准化
import numpy as np
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def standardize(x): 
    return (x-mean_px)/std_px


# In[20]:


# 缺失值填补
X_train.fillna(0, inplace = True)
X_test.fillna(0, inplace = True)


# In[22]:


X_train_NN =(X_train - mean_px) / std_px
X_test_NN  = (X_test - mean_px) / std_px

X_train_NN = (X_train.values).astype('float32') # all pixel values
y_train_NN = y_train.astype('int32')

X_test_NN = (X_test.values).astype('float32') # all pixel values


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_NN,y_train_NN,test_size=0.2)

from sklearn.model_selection import GridSearchCV
estimator = lightgbm.LGBMRegressor(num_leaves=31)

# 配合scikit-learn的网格搜索交叉验证选择最优超参数

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train_NN, y_train_NN)

print('用网格搜索找到的最优超参数为:')
print(gbm.best_params_)

# Score achieved with best parameter combination
print("最优参数的最优分数:",gbm.best_score_)

# all combinations of hyperparameters
print("所有的参数组合",gbm.cv_results_['params'])

# average scores of cross-validation
print("平均测试分数:",gbm.cv_results_['mean_test_score'])

# In[32]:
estimator = lightgbm.LGBMRegressor(num_leaves=31,learning_rate=0.1,n_estimators=40)
estimator.fit(X = X_train_NN, y = y_train_NN)
estimator.booster_.save_model('LGBMmode.txt')
pred_NN = estimator.predict(X_test)

submission = pd.DataFrame({'id':test['loan_id'], 'is_default':pred_NN})
submission.to_csv('submission.csv', index = None)

#计算AUC值
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
gbm_auc = roc_auc_score(y_test, gbm_y_proba[:, 1])  # 计算auc
gbm_fpr, gbm_tpr, gbm_threasholds = roc_curve(y_test, gbm_y_proba[:, 1])  # 计算ROC的值
plt.title("roc_curve of %s(AUC=%.4f)" % ('gbm', gbm_auc))
plt.xlabel('Specificity')  # specificity = 1 - np.array(gbm_fpr))
plt.ylabel('Sensitivity')  # sensitivity = gbm_tpr
plt.plot(list(1 - np.array(gbm_fpr)), gbm_tpr)
plt.gca().invert_xaxis()  # 将X轴反转
plt.show()

