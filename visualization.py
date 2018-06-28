import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv').set_index('PassengerId')
df = df.drop(columns=['Name', 'Ticket'])

sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df)

plt.show()
