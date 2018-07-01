import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import numpy as np

def correctMissingAge(df):
    new_df = df
    for pclass in range(1, 4):
        for sex in range(0, 2):
            ref_df = new_df[(new_df['Pclass'] == pclass) & (new_df['Sex'] == sex)]['Age'].dropna()
            guessed_age = ref_df.median()
            new_df.loc[(new_df['Age'].isnull()) & (new_df['Sex'] == sex) & (new_df['Pclass'] == pclass), 'Age'] = guessed_age
    new_df['Age'] = new_df['Age'].astype(int)
    return new_df

def processDataFrame(df):
    new_df = df
    new_df = new_df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Name', 'Cabin'])
    new_df['Relatives'] = new_df['Parch'] + new_df['SibSp']
    new_df['Sex'] = new_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    new_df = new_df.drop(columns=['SibSp', 'Parch'])
    new_df = correctMissingAge(new_df)
    return new_df


df = pd.read_csv('train.csv')
df = processDataFrame(df)
print(df)

X = df.drop(columns='Survived')
Y = df['Survived']

#logreg = linear_model.LogisticRegression()
#logreg.fit(X, Y)
#
#df_test = pd.read_csv('test.csv')
#df_test['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
#df_test['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
#df_test = df_test.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch', 'Fare', 'Cabin'])
#df_test = df_test.dropna()
#X_test = df_test.drop(columns=['PassengerId'])
#Y_test = logreg.predict(X_test)
#
#df_test['Survived'] = Y_test
#
#fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True)
#sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df, ax=ax0)
#sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df_test, ax=ax1)
#
#plt.show()
