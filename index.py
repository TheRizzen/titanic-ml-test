import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import svm
import numpy as np

def correctMissingAge(pdf):
    new_df = pdf
    for pclass in range(1, 4):
        for sex in range(0, 2):
            ref_df = new_df[(new_df['Pclass'] == pclass) & (new_df['Sex'] == sex)]['Age'].dropna()
            guessed_age = ref_df.median()
            new_df.loc[(new_df['Age'].isnull()) & (new_df['Sex'] == sex) & (new_df['Pclass'] == pclass), 'Age'] = guessed_age
    new_df['Age'] = new_df['Age'].astype(int)
    return new_df

def processDataFrame(pdf):
    new_df = pdf
    new_df = new_df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Name', 'Cabin'])
    new_df['Relatives'] = new_df['Parch'] + new_df['SibSp']
    new_df['Sex'] = new_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    new_df = new_df.drop(columns=['SibSp', 'Parch'])
    new_df = correctMissingAge(new_df)
    new_df['Embarked'] = new_df['Embarked'].fillna(new_df['Embarked'].dropna().mode()[0])
    new_df['Embarked'] = new_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    new_df = new_df.dropna()  # Shouldn't drop anything
    return new_df


df = pd.read_csv('train.csv')
df = processDataFrame(df)

X = df.drop(columns='Survived')
Y = df['Survived']

df_test = pd.read_csv('test.csv')
df_test = processDataFrame(df_test)

X_test = df_test.copy()

svc = svm.SVC()
svc.fit(X, Y)

Y_test = svc.predict(X_test)

#logreg = linear_model.LogisticRegression()
#logreg.fit(X, Y)
#
#Y_test = logreg.predict(X_test)
