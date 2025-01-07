#!/usr/bin/env python
# coding: utf-8

# ## Holiday Package Prediciton
# 
# ### 1) Problem statement.
# "Trips & Travel.Com" company wants to enable and establish a viable business model to expand the customer base.
# One of the ways to expand the customer base is to introduce a new offering of packages. Currently, there are 5 types of packages the company is offering * Basic, Standard, Deluxe, Super Deluxe, King. Looking at the data of the last year, we observed that 18% of the customers purchased the packages. However, the marketing cost was quite high because customers were contacted at random without looking at the available information.
# The company is now planning to launch a new product i.e. Wellness Tourism Package. Wellness Tourism is defined as Travel that allows the traveler to maintain, enhance or kick-start a healthy lifestyle, and support or increase one's sense of well-being.
# However, this time company wants to harness the available data of existing and potential customers to make the marketing expenditure more efficient.
# ### 2) Data Collection.
# The Dataset is collected from https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction
# The data consists of 20 column and 4888 rows.

# In[1035]:


## importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import missingno as msno

warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[1036]:


df = pd.read_csv("Travel.csv")
df.head()


# ## Data Cleaning
# ### Handling Missing values
# 1. Handling Missing values
# 2. Handling Duplicates
# 3. Check data type
# 4. Understand the dataset

# In[1037]:


df.isnull().sum()


# In[1038]:


# Columns containing missing values
for i in df.columns:
    if df[i].isnull().sum() > 0:
        print(i)
        print('the total null values are:', df[i].isnull().sum())
        print('the datatype is', df[i].dtypes)
        print()
msno.matrix(df);


# In[1039]:


df.info()


# In[1040]:


## get all the numeric features
# Creating two lists containing categorical and numerical variables
cat = []
num = []
for i in df.columns:
  if df[i].dtype == 'object':
    cat.append(i)
  else:
    num.append(i)
print('categorical variables = ',cat)
print('numerical variables = ',num)


# In[1041]:


### Check all the categories 
df['Gender'].value_counts()


# In[1042]:


df['MaritalStatus'].value_counts()


# In[1043]:


df['TypeofContact'].value_counts()


# In[1044]:


df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')


# In[1045]:


### Check all the categories 
df['Gender'].value_counts()


# In[1046]:


### Check all the categories 
df['MaritalStatus'].value_counts()


# ### EDA

# ### Splitting data into numerical and categorical variables

# In[1047]:


cats = ['ProdTaken', 'CityTier', 'PreferredPropertyStar', 'Passport',
        'OwnCar', 'PitchSatisfactionScore', 'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
        'MaritalStatus', 'Designation']
nums = ['CustomerID', 'Age', 'DurationOfPitch',
       'NumberOfPersonVisiting', 'NumberOfFollowups',
       'NumberOfTrips',
       'NumberOfChildrenVisiting', 'MonthlyIncome']


# #### Distribution of the numerical data

# In[1048]:


plt.figure(figsize=(15,10))
for i in range(1, len(nums)):
    plt.subplot(3, 4, i)
    sns.distplot(df[nums[i]])
    plt.ylabel('')
    plt.yticks([])
    plt.tight_layout()


# In[1051]:


plt.figure(figsize=(15,5))
for i in range(1, len(nums)):
    plt.subplot(1, 7, i)
    sns.boxplot(y=df[nums[i]],palette="pastel")
    plt.tight_layout()
plt.show()


# In[1054]:


plt.figure(figsize=(15,8))
corr_matrix = df[num].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f')
plt.title("Correlation Matrix")
plt.show()


# In[1056]:


plt.figure(figsize=(15,10))
for i, col in enumerate(nums[1:]):
    plt.subplot(4, 4, i+1)
    plt.tick_params(axis='both', which='major', labelsize=7)
    sns.histplot(data=df, x=col, hue='ProdTaken', multiple='stack')
    
    legend = plt.gca().get_legend()
    legend.set_title('Taken')
    legend.get_texts()[0].set_text('No')
    legend.get_texts()[1].set_text('Yes')

plt.tight_layout()

plt.show()


# #### Distribution of the categorical data

# In[1049]:


df[cats] = df[cats].astype('object')


# In[1052]:


plt.figure(figsize=(15,10))
for i, col in enumerate(cats):
    plt.subplot(4, 3, i+1)
    sns.countplot(x=df[col],color="lightcoral")
    plt.ylabel('')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()


# In[1053]:


# Group the DataFrame by TypeofContact and ProdTaken, and count occurrences
for i in cat:
    grouped = df.groupby([i, 'ProdTaken']).size().unstack()
    colors = ['thistle', 'beige']
    ax = grouped.plot(kind = 'bar', stacked=True, color=colors)

for container in ax.containers:
    for bar in container.patches:
        count = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2 + bar.get_y(),
                round(count), ha = 'center', color = 'black', size = 7)
# Sum of values
grouped['Total'] = grouped.sum(axis=1)
percentage_grouped = grouped.div(grouped['Total'], axis=0) * 100
# Total values labels
for i, total in enumerate(grouped['Total']):
    ax.text(i, total + 50, round(total),
            ha = 'center', color = 'black', size = 8)
ax.set_ylabel('Count')
ax.set_title('TypeofContact and Product taken')
ax.legend(title = 'Product taken', loc = 'upper left')
plt.show()
print(percentage_grouped)


# In[1055]:


labels=['Not Taken','Taken']
explode = [0,0.1]
plt.pie(df['ProdTaken'].value_counts().values, labels=labels, autopct='%.1f%%', explode=explode)
plt.title("Distribution of Product Taken")
plt.legend()
plt.show()


# In[1057]:


df.head()


# In[1058]:


## Check Misssing Values
##these are the features with nan value
features_with_na=[features for features in df.columns if df[features].isnull().sum()>=1]
for feature in features_with_na:
    print(feature,np.round(df[feature].isnull().mean()*100,5), '% missing values')


# In[1059]:


# statistics on numerical columns (Null cols)
df[features_with_na].select_dtypes(exclude='object').describe().T


# In[1060]:


#NumberOfChildrenVisiting
df.NumberOfChildrenVisiting.fillna(df.NumberOfChildrenVisiting.mode()[0], inplace=True)


# In[1061]:


# create new column for feature
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)


# In[1062]:


df.head()


# In[1063]:


df.shape


# In[1064]:


df.drop(['CustomerID'],axis=1, inplace=True)


# ##  Train test split followed by Imputing Null values
# 1. Impute Median value for Age column
# 2. Impute Mode for Type of Contract
# 3. Impute Median for Duration of Pitch
# 4. Impute Mode for NumberofFollowup as it is Discrete feature
# 5. Impute Mode for PreferredPropertyStar
# 6. Impute Median for NumberofTrips
# 7. Impute Mode for NumberOfChildrenVisiting
# 8. Impute Median for MonthlyIncome

# In[1065]:


from sklearn.model_selection import train_test_split


# In[1066]:


X = df.drop(['ProdTaken'], axis=1)
y = df['ProdTaken']


# In[1067]:


# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# In[1068]:


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# In[1069]:


imputer1 = SimpleImputer(strategy='median')#for numerical variable
imputer2 = SimpleImputer(strategy='most_frequent')#for the categorical variable 


# In[1070]:


trf = ColumnTransformer([
    ('imputer1',imputer1,['Age','DurationOfPitch','NumberOfTrips','MonthlyIncome']),
    ('imputer2',imputer2,['TypeofContact','NumberOfFollowups','PreferredPropertyStar'])
],remainder='passthrough')


# In[1071]:


# Fit and transform the training set
X_train_transformed = trf.fit_transform(X_train)

# Transform the test set
X_test_transformed = trf.transform(X_test)

# Manually reconstruct the column names
# For imputer1 and imputer2 (we assume the imputers retain the same number of columns)
numerical_columns = ['Age', 'DurationOfPitch', 'NumberOfTrips', 'MonthlyIncome']
categorical_columns = ['TypeofContact', 'NumberOfFollowups', 'PreferredPropertyStar']
# The remaining columns are passed through as they are, so we need to append them too
remaining_columns = [col for col in X_train.columns if col not in numerical_columns + categorical_columns]

# New column names
new_columns = numerical_columns + categorical_columns + remaining_columns

# Convert the transformed array back to a DataFrame and assign the correct column names
X_train_df = pd.DataFrame(X_train_transformed, columns=new_columns)
X_test_df = pd.DataFrame(X_test_transformed, columns=new_columns)

# Display the resulting DataFrame with correct column names
print(X_train_df.head())


# In[1072]:


X_train_df


# In[1073]:


X_test_df


# In[1074]:


# trf.named_transformers_['imputer1'].statistics_


# In[1075]:


# a=df[['Age','DurationOfPitch','NumberOfTrips','MonthlyIncome']].median()
# print(a)


# In[1076]:


X_train_df.shape


# In[1077]:


X_test_df.shape


# In[1078]:


y_train


# In[1079]:


y_test


# In[1080]:


X_train_df.shape,X_test_df.shape


# In[1081]:


X_test_df.isnull().sum()


# In[1082]:


X = pd.concat([X_train_df, X_test_df], axis=0)


# In[1083]:


X.head()


# In[1084]:


X.info()


# In[1085]:


dtype_mapping = {
    
    'Age': 'float64',
    'TypeofContact': 'object',
    'CityTier': 'int64',
    'DurationOfPitch': 'object',
    'Occupation': 'object',
    'Gender': 'object',
    'NumberOfFollowups': 'float64',
    'ProductPitched': 'object',
    'PreferredPropertyStar': 'object',
    'MaritalStatus': 'object',
    'NumberOfTrips': 'float64',
    'Passport': 'object',
    'PitchSatisfactionScore': 'int64',
    'OwnCar': 'int64',
    'Designation': 'object',
    'MonthlyIncome': 'float64',
    'TotalVisiting':'float64'
}

# Convert columns to respective data types
X = X.astype(dtype_mapping)

# Verify the changes
print(X.dtypes)


# In[1086]:


X=pd.DataFrame(X)


# In[1087]:


# Create Column Transformer with 3 types of transformers
cat_features = X.select_dtypes(include="object").columns
num_features = X.select_dtypes(exclude="object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first',handle_unknown='ignore')

preprocessor = ColumnTransformer(
    [
         ("OneHotEncoder", oh_transformer, cat_features),
          ("StandardScaler", numeric_transformer, num_features)
    ]
)


# In[1088]:


preprocessor


# In[1089]:


X.head()


# In[1090]:


X_train_df.head()


# In[1091]:


## applying Trnsformation in training(fit_transform)
X_train_df=preprocessor.fit_transform(X_train_df)


# In[1092]:


X_train_df=pd.DataFrame(X_train_df.toarray())


# In[1093]:


X_train_df


# In[1094]:


X_train_df.shape


# In[1095]:


## apply tansformation on test(transform)
X_test_df=preprocessor.transform(X_test_df)


# In[1096]:


X_test_df=pd.DataFrame(X_test_df.toarray())


# ## XgboostBoost Classifier Training
# #### We can also combine multiple algorithms
# 

# In[1097]:


X_train_df


# In[1098]:


y_train


# In[1099]:


get_ipython().system('pip install xgboost')


# In[1100]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 


# In[1105]:


y_train.astype('int')


# In[1108]:


# Check and encode the target if necessary
from sklearn.preprocessing import LabelEncoder
if y_train.dtype == 'O':  # Target is an object type
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)


# In[1109]:


models={
    "Logisitic Regression":LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "Gradient Boost":GradientBoostingClassifier(),
    "Adaboost":AdaBoostClassifier(),
    "Xgboost":XGBClassifier()
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train_df, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train_df)
    y_test_pred = model.predict(X_test_df)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
    model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
    model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)


    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
    model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
    model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred) #Calculate Roc


    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))

    
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

    
    print('='*35)
    print('\n')


# In[1120]:


## Hyperparameter Training
dt_param={
    'criterion':['gini','entropy', 'log_loss'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5],
    'max_features':['None','sqrt','log2']}
rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20, 30],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}
adaboost_param={"n_estimators":[50,60,70,80,90],
                "algorithm":['SAMME','SAMME.R']}
gradient_params={"loss": ['log_loss','deviance','exponential'],
             "criterion": ['friedman_mse','squared_error','mse'],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500],
              "max_depth": [5, 8, 15, None, 10]}


# In[1121]:


rf_params


# In[1122]:


xgboost_params


# In[1123]:


# Models list for Hyperparameter tuning
randomcv_models = [
    ("DecisionTree",DecisionTreeClassifier(),dt_param),
    ("RF", RandomForestClassifier(), rf_params),
    ("Xgboost", XGBClassifier(), xgboost_params),("AB", AdaBoostClassifier(), adaboost_param),\
    ("GradientBoost", GradientBoostingClassifier(), gradient_params)
                   
                   ]


# In[1114]:


randomcv_models


# In[1124]:


from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(X_train_df, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])


# In[1128]:


from math import sqrt


# In[1132]:


models={
    
    "Random Forest":RandomForestClassifier(n_estimators=1000,min_samples_split=2,
                                          max_features=7,max_depth=None),
    "Xgboost":XGBClassifier(n_estimators=200,max_depth=12,learning_rate=0.1,
                           colsample_bytree=1)
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train_df, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train_df)
    y_test_pred = model.predict(X_test_df)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted') # Calculate F1-score
    model_train_precision = precision_score(y_train, y_train_pred) # Calculate Precision
    model_train_recall = recall_score(y_train, y_train_pred) # Calculate Recall
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)


    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score
    model_test_precision = precision_score(y_test, y_test_pred) # Calculate Precision
    model_test_recall = recall_score(y_test, y_test_pred) # Calculate Recall
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred) #Calculate Roc


    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))

    
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

    
    print('='*35)
    print('\n')


# In[1134]:


## Plot ROC AUC Curve
from sklearn.metrics import roc_auc_score,roc_curve
plt.figure()

# Add the models to the list that you want to view on the ROC plot
auc_models = [
{
    'label': 'Xgboost',
    'model':XGBClassifier(n_estimators=200,max_depth=12,learning_rate=0.1,
                           colsample_bytree=1),
    'auc':  0.8497
},
    
]
# create loop through all model
for algo in auc_models:
    model = algo['model'] # select the model
    model.fit(X_train_df, y_train) # train the model
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_df)[:,1])
# Calculate Area under the curve to display on the plot
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (algo['label'], algo['auc']))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show() 

