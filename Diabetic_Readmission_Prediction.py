#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv('diabetic_data.csv')
df.head(10).T


# In[6]:


df.columns


# In[9]:


df.info()


# In[5]:


df.groupby('readmitted').size()


# In[6]:


df['readmitted'] = df['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})


# In[7]:


df.groupby('readmitted').size()


# In[8]:


df.to_csv('./diabetes_data_biclass.csv')


# In[9]:


df = pd.read_csv('diabetes_data_biclass.csv')


# In[10]:


df.head()


# In[11]:


df.columns


# In[12]:


df = df.drop(columns=['Unnamed: 0'])


# In[13]:


df.head()


# In[14]:


import pandas_profiling


# In[15]:


pandas_profiling.ProfileReport(df)


# In[16]:


df.info()


# In[17]:


df.describe().transpose()


# In[18]:


for col in df.columns:
    if df[col].dtype == object:
        print(col, df[col][df[col] == '?'].count())


# In[19]:


print('gender', df['gender'][df['gender'] == 'Unknown/Invalid'].count())


# In[20]:


df = df.drop(['weight', 'payer_code', 'medical_specialty'], axis = 1)


# In[21]:


df.shape


# In[22]:


df.head()


# In[23]:


df['race'] = df['race'].replace('?','others')


# In[24]:


df['race'].value_counts()


# In[25]:


drop_Idx = set(df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index)


# In[26]:


drop_Idx = drop_Idx.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))


# In[27]:


drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 11].index))


# In[28]:


drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 19].index))


# In[29]:


drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 20].index))


# In[30]:


drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 21].index))


# In[31]:


new_Idx = list(set(df.index) - set(drop_Idx))


# In[32]:


df = df.iloc[new_Idx]


# In[33]:


df.shape


# In[34]:


df.head()


# In[35]:


df = df.drop(['citoglipton', 'examide'], axis = 1)


# In[36]:


df.shape


# In[37]:


df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']


# In[38]:


df = df.drop(columns=['number_outpatient','number_emergency','number_inpatient'])


# In[39]:


df.shape


# In[40]:


df['admission_type_id'] = df['admission_type_id'].replace(2,1)
df['admission_type_id'] = df['admission_type_id'].replace(7,1)
df['admission_type_id'] = df['admission_type_id'].replace(6,5)
df['admission_type_id'] = df['admission_type_id'].replace(8,5)


# In[41]:


df['admission_type_id'].value_counts()


# In[42]:


df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(6,1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(8,1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(9,1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(13,1)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(3,2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(4,2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(5,2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(14,2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(22,2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(23,2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(24,2)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(12,10)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(15,10)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(16,10)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(17,10)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(25,18)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(26,18)


# In[43]:


df['discharge_disposition_id'].value_counts()


# In[44]:


df['admission_source_id'] = df['admission_source_id'].replace(2,1)
df['admission_source_id'] = df['admission_source_id'].replace(3,1)
df['admission_source_id'] = df['admission_source_id'].replace(5,4)
df['admission_source_id'] = df['admission_source_id'].replace(6,4)
df['admission_source_id'] = df['admission_source_id'].replace(10,4)
df['admission_source_id'] = df['admission_source_id'].replace(22,4)
df['admission_source_id'] = df['admission_source_id'].replace(25,4)
df['admission_source_id'] = df['admission_source_id'].replace(15,9)
df['admission_source_id'] = df['admission_source_id'].replace(17,9)
df['admission_source_id'] = df['admission_source_id'].replace(20,9)
df['admission_source_id'] = df['admission_source_id'].replace(21,9)
df['admission_source_id'] = df['admission_source_id'].replace(13,11)
df['admission_source_id'] = df['admission_source_id'].replace(14,11)


# In[45]:


df['admission_source_id'].value_counts()


# In[46]:


keys = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
        'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
        'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone','metformin-rosiglitazone', 
        'glimepiride-pioglitazone', 'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide']

for col in keys:
    colname = str(col) + 'temp'
    df[colname] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)

df['numchange'] = 0

for col in keys:
    colname = str(col) + 'temp'
    df['numchange'] = df['numchange'] + df[colname]
    del df[colname]


# In[47]:


df['numchange'].value_counts()


# In[48]:


for col in keys:
    df[col] = df[col].replace('No', 0)
    df[col] = df[col].replace('Steady', 1)
    df[col] = df[col].replace('Up', 1)
    df[col] = df[col].replace('Down', 1)


# In[49]:


df['nummed'] = 0

for col in keys:
    df['nummed'] = df['nummed'] + df[col]


# In[50]:


df['nummed'].value_counts()


# In[51]:


# Creating additional columns for diagnosis
df['level1_diag1'] = df['diag_1']
df['level1_diag2'] = df['diag_2']
df['level1_diag3'] = df['diag_3']


# In[52]:


df.loc[df['diag_1'].str.contains('V'), ['level1_diag1']] = 0
df.loc[df['diag_1'].str.contains('E'), ['level1_diag1']] = 0
df.loc[df['diag_2'].str.contains('V'), ['level1_diag2']] = 0
df.loc[df['diag_2'].str.contains('E'), ['level1_diag2']] = 0
df.loc[df['diag_3'].str.contains('V'), ['level1_diag3']] = 0
df.loc[df['diag_3'].str.contains('E'), ['level1_diag3']] = 0
df['level1_diag1'] = df['level1_diag1'].replace('?', -1)
df['level1_diag2'] = df['level1_diag2'].replace('?', -1)
df['level1_diag3'] = df['level1_diag3'].replace('?', -1)


# In[53]:


df['level1_diag1'] = df['level1_diag1'].astype(float)
df['level1_diag2'] = df['level1_diag2'].astype(float)
df['level1_diag3'] = df['level1_diag3'].astype(float)


# In[54]:


for index, row in df.iterrows():
    if (row['level1_diag1'] >= 390 and row['level1_diag1'] < 460) or (np.floor(row['level1_diag1']) == 785):
        df.loc[index, 'level1_diag1'] = 1
    elif (row['level1_diag1'] >= 460 and row['level1_diag1'] < 520) or (np.floor(row['level1_diag1']) == 786):
        df.loc[index, 'level1_diag1'] = 2
    elif (row['level1_diag1'] >= 520 and row['level1_diag1'] < 580) or (np.floor(row['level1_diag1']) == 787):
        df.loc[index, 'level1_diag1'] = 3
    elif (np.floor(row['level1_diag1']) == 250):
        df.loc[index, 'level1_diag1'] = 4
    elif (row['level1_diag1'] >= 800 and row['level1_diag1'] < 1000):
        df.loc[index, 'level1_diag1'] = 5
    elif (row['level1_diag1'] >= 710 and row['level1_diag1'] < 740):
        df.loc[index, 'level1_diag1'] = 6
    elif (row['level1_diag1'] >= 580 and row['level1_diag1'] < 630) or (np.floor(row['level1_diag1']) == 788):
        df.loc[index, 'level1_diag1'] = 7
    elif (row['level1_diag1'] >= 140 and row['level1_diag1'] < 240):
        df.loc[index, 'level1_diag1'] = 8
    else:
        df.loc[index, 'level1_diag1'] = 0
        
    if (row['level1_diag2'] >= 390 and row['level1_diag2'] < 460) or (np.floor(row['level1_diag2']) == 785):
        df.loc[index, 'level1_diag2'] = 1
    elif (row['level1_diag2'] >= 460 and row['level1_diag2'] < 520) or (np.floor(row['level1_diag2']) == 786):
        df.loc[index, 'level1_diag2'] = 2
    elif (row['level1_diag2'] >= 520 and row['level1_diag2'] < 580) or (np.floor(row['level1_diag2']) == 787):
        df.loc[index, 'level1_diag2'] = 3
    elif (np.floor(row['level1_diag2']) == 250):
        df.loc[index, 'level1_diag2'] = 4
    elif (row['level1_diag2'] >= 800 and row['level1_diag2'] < 1000):
        df.loc[index, 'level1_diag2'] = 5
    elif (row['level1_diag2'] >= 710 and row['level1_diag2'] < 740):
        df.loc[index, 'level1_diag2'] = 6
    elif (row['level1_diag2'] >= 580 and row['level1_diag2'] < 630) or (np.floor(row['level1_diag2']) == 788):
        df.loc[index, 'level1_diag2'] = 7
    elif (row['level1_diag2'] >= 140 and row['level1_diag2'] < 240):
        df.loc[index, 'level1_diag2'] = 8
    else:
        df.loc[index, 'level1_diag2'] = 0
    
    if (row['level1_diag3'] >= 390 and row['level1_diag3'] < 460) or (np.floor(row['level1_diag3']) == 785):
        df.loc[index, 'level1_diag3'] = 1
    elif (row['level1_diag3'] >= 460 and row['level1_diag3'] < 520) or (np.floor(row['level1_diag3']) == 786):
        df.loc[index, 'level1_diag3'] = 2
    elif (row['level1_diag3'] >= 520 and row['level1_diag3'] < 580) or (np.floor(row['level1_diag3']) == 787):
        df.loc[index, 'level1_diag3'] = 3
    elif (np.floor(row['level1_diag3']) == 250):
        df.loc[index, 'level1_diag3'] = 4
    elif (row['level1_diag3'] >= 800 and row['level1_diag3'] < 1000):
        df.loc[index, 'level1_diag3'] = 5
    elif (row['level1_diag3'] >= 710 and row['level1_diag3'] < 740):
        df.loc[index, 'level1_diag3'] = 6
    elif (row['level1_diag3'] >= 580 and row['level1_diag3'] < 630) or (np.floor(row['level1_diag3']) == 788):
        df.loc[index, 'level1_diag3'] = 7
    elif (row['level1_diag3'] >= 140 and row['level1_diag3'] < 240):
        df.loc[index, 'level1_diag3'] = 8
    else:
        df.loc[index, 'level1_diag3'] = 0


# In[55]:


df[['diag_1','level1_diag1']].head().T


# In[56]:


df['level1_diag1'].value_counts()


# In[57]:


df['level1_diag2'].value_counts()


# In[58]:


df['level1_diag3'].value_counts()


# In[59]:


df = df.drop(['level1_diag2', 'level1_diag3'], axis = 1)


# In[60]:


df = df.drop(['diag_1','diag_2','diag_3'], axis = 1)


# In[61]:


df['change'] = df['change'].replace('Ch', 1)
df['change'] = df['change'].replace('No', 0)


# In[62]:


df['diabetesMed'] = df['diabetesMed'].replace('Yes', 1)
df['diabetesMed'] = df['diabetesMed'].replace('No', 0)


# In[63]:


df['gender'] = df['gender'].replace('Male', 1)
df['gender'] = df['gender'].replace('Female', 0)


# In[64]:


age_dict = {'[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95}


# In[65]:


df['age'] = df.age.map(age_dict)
df['age'] = df['age'].astype('int64')


# In[66]:


df['age'].value_counts()


# In[67]:


df['A1Cresult'] = df['A1Cresult'].replace('>7', 1)
df['A1Cresult'] = df['A1Cresult'].replace('>8', 1)
df['A1Cresult'] = df['A1Cresult'].replace('Norm', 0)
df['A1Cresult'] = df['A1Cresult'].replace('None', -99)


# In[68]:


df['max_glu_serum'] = df['max_glu_serum'].replace('>200', 1)
df['max_glu_serum'] = df['max_glu_serum'].replace('>300', 1)
df['max_glu_serum'] = df['max_glu_serum'].replace('Norm', 0)
df['max_glu_serum'] = df['max_glu_serum'].replace('None', -99)


# In[69]:


df = df.drop_duplicates(subset= ['patient_nbr'], keep = 'first')
df.shape


# In[70]:


df = df.drop(['encounter_id', 'patient_nbr'], axis = 1)


# In[71]:


df.shape


# In[72]:


df.dtypes


# In[73]:


df.columns


# In[74]:


df.to_csv('./diabetic_data_preprocessed.csv')


# In[75]:


df = pd.read_csv("diabetic_data_preprocessed.csv", index_col=0)
df.shape


# In[76]:


# convert data type of nominal features in dataframe to 'object' type
i = ['gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
     'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
     'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose','miglitol',
     'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
     'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
     'max_glu_serum', 'level1_diag1']

df[i] = df[i].astype('object')


# In[77]:


df.dtypes


# In[78]:


# get list of only numeric features
num_col = list(set(list(df._get_numeric_data().columns))- {'readmitted'})


# In[79]:


num_col


# In[250]:


corrmat = df.corr()
plt.subplots(figsize = (10, 10))
sns.heatmap(corrmat, annot = True, square = True)


# In[80]:


import scipy as sp


# In[81]:


# standardize function
def standardize(raw_data):
    return ((raw_data - np.mean(raw_data, axis = 0)) / np.std(raw_data, axis = 0))


# In[82]:


df[num_col] = standardize(df[num_col])


# In[83]:


df[num_col].head()


# In[84]:


df.shape


# In[85]:


df = df[(np.abs(sp.stats.zscore(df[num_col])) < 3).all(axis=1)]


# In[86]:


df.shape


# In[87]:


df.dtypes


# In[88]:


df_pd = pd.get_dummies(df, columns=['race', 'gender', 'admission_type_id', 'discharge_disposition_id',
                                    'admission_source_id', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
                                    'level1_diag1'], drop_first = True)


# In[89]:


df_pd.columns


# In[90]:


df_pd['readmitted'].value_counts()


# In[91]:


df_pd.shape


# In[92]:


feature_set = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
               'num_medications', 'number_diagnoses', 'service_utilization',
               'numchange', 'nummed', 'race_Asian', 'race_Caucasian', 'race_Hispanic',
               'race_Other', 'race_others', 'gender_1', 'admission_type_id_3',
               'admission_type_id_4', 'admission_type_id_5',
               'discharge_disposition_id_2', 'discharge_disposition_id_7',
               'discharge_disposition_id_10', 'discharge_disposition_id_18',
               'discharge_disposition_id_27', 'discharge_disposition_id_28',
               'admission_source_id_4', 'admission_source_id_7',
               'admission_source_id_8', 'admission_source_id_9',
               'admission_source_id_11', 'max_glu_serum_0', 'max_glu_serum_1',
               'A1Cresult_0', 'A1Cresult_1', 'change_1', 'diabetesMed_1',
               'level1_diag1_1.0', 'level1_diag1_2.0', 'level1_diag1_3.0',
               'level1_diag1_4.0', 'level1_diag1_5.0', 'level1_diag1_6.0',
               'level1_diag1_7.0', 'level1_diag1_8.0']


# In[93]:


train_input = df_pd[feature_set]
train_output = df_pd['readmitted']


# In[94]:


train_input.head(2)


# In[95]:


train_output.head()


# In[96]:


from imblearn.over_sampling import SMOTE


# In[97]:


from collections import Counter


# In[98]:


print('Original dataset shape {}'.format(Counter(train_output)))


# In[99]:


sm = SMOTE(random_state = 0)


# In[100]:


train_input_new, train_output_new = sm.fit_sample(train_input, train_output)


# In[101]:


print('New dataset shape {}'.format(Counter(train_output_new)))


# In[102]:


train_input_new = pd.DataFrame(train_input_new, columns = list(train_input.columns))


# In[103]:


from sklearn.model_selection import train_test_split


# In[104]:


train_set, test_set, train_labels, test_labels = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)


# ## Naive Bayes

# In[105]:


from sklearn.naive_bayes import GaussianNB


# In[106]:


nb = GaussianNB()


# In[107]:


nb.fit(train_set, train_labels)


# In[108]:


y_nb_predict = nb.predict(test_set)


# In[109]:


print(nb.score(test_set, test_labels))


# In[111]:


from sklearn import metrics


# In[112]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_nb_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[113]:


print(metrics.classification_report(test_labels, y_nb_predict))


# In[114]:


probs = nb.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[115]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Logistic Regression

# In[116]:


from sklearn.linear_model import LogisticRegression


# In[117]:


logreg = LogisticRegression()


# In[118]:


logreg.fit(train_set, train_labels)


# In[119]:


y_logreg_predict = logreg.predict(test_set)


# In[120]:


print(logreg.score(test_set, test_labels))


# In[121]:


from sklearn import metrics


# In[122]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_logreg_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[123]:


print(metrics.classification_report(test_labels, y_logreg_predict))


# In[124]:


feature_names = train_set.columns
feature_imports = logreg.coef_[0]
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(20, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Logistic Regression')
plt.show()


# In[125]:


from sklearn.metrics import roc_curve, auc


# ## ROC Curve Metrics

# In[126]:


probs = logreg.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[127]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Random Forest

# In[128]:


from sklearn.ensemble import RandomForestClassifier


# In[129]:


rf = RandomForestClassifier()


# In[130]:


rf.fit(train_set, train_labels)


# In[131]:


y_rf_predict = rf.predict(test_set)


# In[132]:


print(rf.score(test_set, test_labels))


# In[133]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_rf_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[134]:


print(metrics.classification_report(test_labels, y_rf_predict))


# In[135]:


feature_names = train_set.columns
feature_imports = rf.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(20, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Random Forest')
plt.show()


# In[136]:


print (pd.Series(rf.feature_importances_, index = list(train_set)).sort_values(ascending=False))


# In[137]:


probs = rf.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[138]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Decision Tree

# In[139]:


from sklearn.tree import DecisionTreeClassifier


# In[140]:


dt = DecisionTreeClassifier(criterion = 'entropy')


# In[141]:


dt.fit(train_set, train_labels)


# In[142]:


y_dt_predict = dt.predict(test_set)


# In[143]:


print(dt.score(test_set, test_labels))


# In[144]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_dt_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[145]:


print(metrics.classification_report(test_labels, y_dt_predict))


# In[146]:


feature_names = train_set.columns
feature_imports = dt.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(20, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Decision Tree')
plt.show()


# In[147]:


print (pd.Series(dt.feature_importances_, index = list(train_set)).sort_values(ascending=False))


# In[148]:


import graphviz
from IPython.display import Image
import pydotplus
from sklearn import tree


# In[149]:


dot_dt_q2 = tree.export_graphviz(dt, out_file="dt_q2.dot", feature_names=train_set.columns, max_depth = 2, class_names=["No","Readm"], filled=True, rounded=True, special_characters=True)
graph_dt_q2 = pydotplus.graph_from_dot_file('dt_q2.dot')
Image(graph_dt_q2.create_png())


# In[150]:


probs = dt.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[151]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Bagging Classifier

# In[152]:


from sklearn.ensemble import BaggingClassifier


# In[153]:


bgcl = BaggingClassifier(bootstrap = True, oob_score = True)


# In[154]:


bgcl.fit(train_set, train_labels)


# In[155]:


y_bgcl_predict = bgcl.predict(test_set)


# In[156]:


print(bgcl.score(test_set, test_labels))


# In[157]:


bgcl.oob_score_


# In[158]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_bgcl_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[159]:


print(metrics.classification_report(test_labels, y_bgcl_predict))


# In[160]:


probs = bgcl.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[161]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## AdaBoostClassifier

# In[162]:


from sklearn.ensemble import AdaBoostClassifier


# In[163]:


abcl = AdaBoostClassifier(base_estimator = dt)


# In[164]:


abcl.fit(train_set, train_labels)


# In[165]:


y_abcl_predict = abcl.predict(test_set)


# In[166]:


print(abcl.score(test_set, test_labels))


# In[167]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_abcl_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[168]:


print(metrics.classification_report(test_labels, y_abcl_predict))


# In[169]:


probs = abcl.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[170]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## GradientBoostingClassifier

# In[171]:


from sklearn.ensemble import GradientBoostingClassifier


# In[172]:


gbcl = GradientBoostingClassifier(learning_rate = 0.01)


# In[173]:


gbcl.fit(train_set, train_labels)


# In[174]:


y_gbcl_predict = gbcl.predict(test_set)


# In[175]:


print(gbcl.score(test_set, test_labels))


# In[176]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_gbcl_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[177]:


print(metrics.classification_report(test_labels, y_gbcl_predict))


# In[178]:


probs = gbcl.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[179]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## KNearestNeighbors

# In[180]:


from sklearn.neighbors import KNeighborsClassifier


# In[181]:


knn = KNeighborsClassifier(n_neighbors = 100)


# In[182]:


knn.fit(train_set, train_labels)


# In[183]:


y_knn_predict = knn.predict(test_set)


# In[184]:


print(knn.score(test_set, test_labels))


# In[185]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_knn_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[186]:


print(metrics.classification_report(test_labels, y_knn_predict))


# In[187]:


probs = knn.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[188]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Stacked Model

# In[197]:


from sklearn.ensemble import VotingClassifier


# In[198]:


from sklearn.model_selection import cross_val_score


# # Stacked Model of Random Forest, Bagging Classifier, GradientBoostingClassifier, AdaBoostClassifier

# In[199]:


stacked_model = VotingClassifier(estimators = [('rf', rf), ('bgcl', bgcl), ('gbcl', gbcl), ('abcl', abcl)], voting = 'hard')


# ## KFold Cross Validation

# In[200]:


for clf, label in zip([rf, bgcl, gbcl, abcl, stacked_model], ['Random Forest', 'BaggingClassifier', 'GradientBoostingClassifier', 'AdaBoostClassifier', 'StackedModel']):
    scores = cross_val_score(clf, train_set, train_labels, cv=10, scoring='accuracy')
    print("Accuracy: %0.02f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# ## Implementing SHAP

# In[251]:


test_set.head()


# In[252]:


row_to_show = 5
data_for_prediction = test_set.iloc[row_to_show]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
rf.predict_proba(data_for_prediction_array)


# In[253]:


import shap


# In[254]:


explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(data_for_prediction)


# In[255]:


shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)


# In[256]:


explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(test_set)
shap.summary_plot(shap_values[1], test_set)


# In[257]:


shap.dependence_plot('time_in_hospital', shap_values[1], test_set, interaction_index = 'age')


# In[258]:


shap.dependence_plot('time_in_hospital', shap_values[1], test_set, interaction_index = "num_lab_procedures")


# In[259]:


shap.dependence_plot('time_in_hospital', shap_values[1], test_set, interaction_index = "num_medications")


# In[260]:


shap.dependence_plot('time_in_hospital', shap_values[1], test_set, interaction_index = "service_utilization")


# In[261]:


shap.dependence_plot('time_in_hospital', shap_values[1], test_set, interaction_index = "number_diagnoses")


# In[262]:


shap.dependence_plot('age', shap_values[1], test_set, interaction_index = 'num_lab_procedures')


# In[263]:


shap.dependence_plot('age', shap_values[1], test_set, interaction_index = 'num_medications')


# In[264]:


shap.dependence_plot('age', shap_values[1], test_set, interaction_index = 'service_utilization')


# In[265]:


shap.dependence_plot('age', shap_values[1], test_set, interaction_index = 'number_diagnoses')


# In[266]:


shap.dependence_plot('num_lab_procedures', shap_values[1], test_set, interaction_index = 'num_medications')


# In[267]:


shap.dependence_plot('num_lab_procedures', shap_values[1], test_set, interaction_index = 'number_diagnoses')


# In[268]:


shap.dependence_plot('num_lab_procedures', shap_values[1], test_set, interaction_index = 'service_utilization')


# In[269]:


shap.dependence_plot('num_lab_procedures', shap_values[1], test_set, interaction_index = 'num_procedures')


# ##  Out of Bag Error Rate

# In[201]:


# Setting the RandomForest state for reproductibity
fit_rf = RandomForestClassifier(random_state = 42)


# In[202]:


fit_rf.set_params(warm_start = True, oob_score = True)


# In[203]:


min_estimators = 350
max_estimators = 375


# In[204]:


error_rate = {}


# In[205]:


for i in range(min_estimators, max_estimators + 1):
    fit_rf.set_params(n_estimators=i)
    fit_rf.fit(train_set, train_labels)

    oob_error = 1 - fit_rf.oob_score_
    error_rate[i] = oob_error


# In[206]:


# Dictionary to Panda's series for plotting
oob_series = pd.Series(error_rate)


# In[207]:


fig, ax = plt.subplots(figsize=(20, 20))
oob_series.plot(kind='line', color = 'red')
plt.axhline(0.055, color='#875FDB', linestyle='--')
plt.axhline(0.05, color='#875FDB', linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')


# ## Hyperparameter Tuning

# In[208]:


model = RandomForestClassifier (random_state = 42, oob_score = True)


# In[209]:


# Hyperparameters set
params = {
          'n_estimators': [375],
          'max_depth': [21],
          'min_samples_leaf': [3],
          'min_samples_split': [10],
          'max_features': [16]
         }


# In[210]:


from sklearn.model_selection import GridSearchCV


# In[211]:


model_rf = GridSearchCV(model, param_grid = params)


# In[212]:


model_rf.fit(train_set, train_labels)


# In[213]:


print("Best Hyper Parameters:", model_rf.best_params_)


# In[214]:


## Final Hypertuned Model; Setting the best hyper parameters given by GridSearch


# In[215]:


model_final_rf = RandomForestClassifier(n_estimators = 375 , max_depth = 21 , min_samples_leaf = 3 , min_samples_split = 10, max_features = 16 , random_state = 42, oob_score = True)


# In[216]:


model_final_rf.fit(train_set, train_labels)


# In[217]:


y_model_final_rf_predict = model_final_rf.predict(test_set)


# In[218]:


print(model_final_rf.score(test_set, test_labels))


# In[219]:


confusion_df = pd.DataFrame(metrics.confusion_matrix(test_labels, y_model_final_rf_predict),
                            columns=["Predicted Class " + str(class_name) for class_name in [0,1]],
                            index = ["Actual Class " + str(class_name) for class_name in [0,1]])
print(confusion_df)


# In[220]:


print(metrics.classification_report(test_labels, y_model_final_rf_predict))


# In[221]:


probs = model_final_rf.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test_labels, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[222]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[223]:


feature_names = train_set.columns
feature_imports = model_final_rf.feature_importances_
most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(20, "Importance")
most_imp_features.sort_values(by="Importance", inplace=True)
plt.figure(figsize=(10,6))
plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)
plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)
plt.xlabel('Importance')
plt.title('Most important features - Random Forest')
plt.show()


# In[224]:


for clf, label in zip([model_final_rf], ['Random Forest']):
    scores = cross_val_score(clf, train_set, train_labels, cv=10, scoring='accuracy')
    print("Accuracy: %0.02f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# ### Delivering solution to the Customer

# In[239]:


from sklearn.preprocessing import StandardScaler


# In[230]:


import pickle


# In[231]:


file_Name = "DiabetesReadmissionPrediction"
# open the file for writing
fileObject = open(file_Name, 'wb')


# In[232]:


# Writing the object model to the file named 'DiabetesReadmissionPrediction'
pickle.dump(model_final_rf, fileObject)


# In[234]:


# Closing the fileObject
fileObject.close()


# In[235]:


# Open the file for reading
fileObject = open('DiabetesReadmissionPrediction','rb') 


# In[236]:


# Load the object from the file
reloadedModel = pickle.load(fileObject)


# In[ ]:




