import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

df = pd.read_csv('train.csv')
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'])
df = df.dropna()
df['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

X = df.drop(columns='Survived')
Y = df['Survived']

logreg = linear_model.LogisticRegression()
logreg.fit(X, Y)

df_test = pd.read_csv('test.csv')
df_test['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
df_test = df_test.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'])
df_test = df_test.dropna()
X_test = df_test.drop(columns=['PassengerId'])
Y_test = logreg.predict(X_test)

df_test['Survived'] = Y_test
print(df_test)

fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True)
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df, ax=ax0)
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df_test, ax=ax1)

plt.show()
