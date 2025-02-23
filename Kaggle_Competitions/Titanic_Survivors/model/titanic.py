import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder   
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import xgboost as xgb

#from sklearn.impute import SimpleImputer

df_train = pd.read_csv(r'D:\\DS\\Kaggle\\titanic\\train.csv')
df_test = pd.read_csv(r'D:\\DS\\Kaggle\\titanic\\test.csv')
df_merged = pd.concat([df_train, df_test], sort=False).drop(['Survived','PassengerId'], axis=1)
print(df_merged.head())

def dataPreprocessingForDecisionTree(df):
    df['Sex'] = df['Sex'].replace('male', 0).replace('female', 1)
    df['Embarked'] = df['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)
    #df['Family']=df['SibSp']+df['Parch']
    df["Alone"] = 0
    df.loc[(df["Parch"]==0) & (df["SibSp"]==0),"Alone"]=1
    #df['Family_Category']=pd.cut(df['Family'],bins=[-np.inf,0,2,4,np.inf],labels=(0,1,2,3))
    df['Age_Category']=pd.cut(df['Age'], bins=[0,15,20,30,45,60,np.inf],labels=(0,1,2,3,4,5))
    df['Fare_Range'] = pd.cut(df['Fare'], bins=[-np.inf,10,20,32,np.inf],labels=(0,1,2,3))
    #df.drop(['Ticket','Name','SibSp','Parch','Cabin','Age','Fare','Title','Family'], axis=1, inplace=True)
    df.drop(['Ticket','Name','Cabin','Age','Fare','Title','Parch','SibSp'], axis=1, inplace=True)
    return df

def fillMissingValues(df):
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    medians = df.groupby('Pclass')['Fare'].transform('median')
    df['Fare'] = df['Fare'].fillna(medians)
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Title'] = df['Title'].apply(assign_title_category)
    df['Age']=df.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    return df

def oneHotEncoding(df):
    df = pd.get_dummies(df, columns=['Embarked','Age_Category','Fare_Range'])
    return df

def assign_title_category(title):
    if title in ['Mr', 'Sir', 'Don', 'Capt', 'Col', 'Major', 'Rev', 'Jonkheer', 'Dr']:
        return 1
    elif title in ['Mrs', 'Lady', 'the Countess', 'Dona', 'Mme']:
        return 2
    else:
        return 3

def determine_criteria(row):
    if row['Age_Category'] == 0:
        return '0'
    elif row['Sex'] == '1' and row['Age_Category'] != 0:
        return '1'
    elif row['Age_Category'] == 5:
        return '3'
    else:
        return '2'


df_merged = fillMissingValues(df_merged)


df_merged = dataPreprocessingForDecisionTree(df_merged)
df_merged = oneHotEncoding(df_merged)
print(df_merged.head())
print(df_merged.info())
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df_merged)
df_merged = pd.DataFrame(scaled_data, columns=df_merged.columns, index=df_merged.index)

df_train_copy = df_merged[0:891]
df_test_copy = df_merged[891:]
df_train_copy['Survived'] = df_train['Survived']

def trainRandomForest(X_train, y_train, X_test,df_test):
    pipe = Pipeline([ ('pca', PCA()),
        ('rf', RandomForestClassifier(random_state=42))])
    param_grid = {
         'pca__n_components': [0.95, 0.99, None],  # Variance to keep, or None for all components
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [5, 10, 15, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    df_op = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred})
    df_op.to_csv(r'D:\DS\Kaggle\titanic\output.csv', index=False)


def trainLogisticRegressionWithPCA(X_train, y_train, X_test, df_test):
    # Define a pipeline with PCA and LogisticRegression
    pipe = Pipeline([
        ('pca', PCA()),
        ('log_reg', LogisticRegression())
    ])
    
    # Parameter grid with options for both PCA and LogisticRegression
    param_grid = {
        'pca__n_components': [0.95, 0.99, None],  # Variance to keep, or None for all components
        'log_reg__C': [0.1, 1, 10, 100],  # Regularization strength
        'log_reg__penalty': ['l1', 'l2'],  # Regularization type
        'log_reg__solver': ['liblinear', 'saga'],  # Solvers that support L1 penalty
        'log_reg__max_iter': [100, 200, 500]  # Maximum number of iterations
    }

    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    # Make predictions using the best found model
    y_pred = grid_search.predict(X_test)

    # Create a DataFrame for output
    df_op = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred})
    df_op.to_csv(r'D:\DS\Kaggle\titanic\output.csv', index=False)

def trainSVM(X_train, y_train, X_test, df_test):
    # Initialize the SVM model
    svm = SVC()

    # Define the parameter grid for the SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization strength
        'kernel': ['linear', 'rbf', 'poly'],  # Type of kernel
        'gamma': ['scale', 'auto'],  # Kernel coefficient
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Predict using the best model
    y_pred = grid_search.predict(X_test)

    # Create a DataFrame with the results
    df_op = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred})

    # Save to CSV
    df_op.to_csv(r'D:\DS\Kaggle\titanic\output.csv', index=False)

def trainXGBoost(X_train, y_train, X_test, df_test):
    # Initialize XGBoost classifier
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    
    # Parameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees
        'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
        'max_depth': [3, 5, 7],  # Maximum tree depth
        'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight needed in a child
        'subsample': [0.8, 0.9, 1],  # Subsample ratio of the training instances
        'colsample_bytree': [0.8, 0.9, 1],  # Subsample ratio of columns when constructing each tree
    }

    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)

    # Make predictions using the best found model
    y_pred = grid_search.predict(X_test)

    # Create a DataFrame for output
    df_op = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred})
    df_op.to_csv(r'D:\DS\Kaggle\titanic\output.csv', index=False)

trainXGBoost(df_train_copy.drop(['Survived'], axis=1), df_train_copy['Survived'],df_test_copy,df_test)








def trainLogisticRegressionTest(X_train,y_train):
    model = LogisticRegression()
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # 'liblinear' works well with l1 and l2 penalties
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    # Best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Evaluate on the test set
    y_pred = grid_search.predict(X_train)
    test_accuracy = accuracy_score(y_train, y_pred)
    print("Test set accuracy:", test_accuracy)







def peekData(df):
    print(df_merged.head())
    print(df_merged.info())
    print(df_merged.columns)    
    missing_values_count = df_merged.isna().sum()
    columns_with_missing_values = missing_values_count[missing_values_count > 0]
    print(f"\nColumns with missing values:\n{columns_with_missing_values}")
    for column in df_merged.columns:
        if df_merged[column].dtype == 'object' or df_merged[column].dtype.name == 'category':
            print(f"Value Counts for {column}:\n{df_merged[column].value_counts()}\n")




def changeCategoricalData(df):
    df['Sex'] = df['Sex'].replace('male', 0).replace('female', 1)
    #df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else x)
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Age_Category']=pd.cut(df['Age'], bins=[0,15,20,30,45,60,np.inf],labels=(0,1,2,3,4,5))
    df['Age_Gender_Criteria'] = df.apply(determine_criteria, axis=1)
    df['Family']=df['SibSp']+df['Parch']
    df.drop(['Name','Title','Ticket', 'Age','Sex','Age_Category','SibSp','Parch','Cabin'], axis=1, inplace=True)
    return pd.get_dummies(df, columns=['Embarked','Age_Gender_Criteria'])



def visualize(df):
   
    pivot_data = df.pivot_table(index='Age_Category', columns=['Survived', 'Sex'], aggfunc='size')
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_data.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Adding data labels
    for bars in ax.containers:
        ax.bar_label(bars, label_type='edge', padding=3, fontsize=8, fmt='%.0f')

    plt.title('Count of Binary Values for Each Category with Gender Side by Side', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)  # Set rotation to 0 for better readability if needed
    plt.legend(title='Binary Value and Gender', 
           labels=['0 (Male)', '0 (Female)', '1 (Male)', '1 (Female)'],
           fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_corelation(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Fare'], df['Pclass'], alpha=0.7)  # alpha for transparency
    plt.title('Scatter Plot of Column1 vs. Column2')
    plt.xlabel('Column1')
    plt.ylabel('Column2')
    plt.grid(True)
    plt.show()


