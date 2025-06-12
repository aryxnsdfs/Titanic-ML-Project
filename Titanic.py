import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"c:\Users\aryan\Downloads\Titanic-Dataset.csv")
df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex']=df['Sex'].map({'male':0,'female':1})

df=pd.get_dummies(df,columns=['Embarked','Pclass'],drop_first=True)

df['Title']=df['Name'].str.extract('([A-Za-z]+)\.',expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt','Col',
                                   'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                   'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df=pd.get_dummies(df,columns=['Title'],drop_first=True)

df['FamilySize']=df['SibSp']+df['Parch']+1

df['IsAlone']=0
df.loc[df['FamilySize']==1,'IsAlone']=1
df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

x=df.drop('Survived',axis=1)
y=df['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
tra=model.fit(x_train,y_train)

pred=model.predict(x_test)
print(accuracy_score(y_test,pred))
print(df.isnull().sum())

impt=model.feature_importances_

names=x.columns
fea=pd.Series(impt,index=names).sort_values(ascending=True)

plt.figure(figsize=(10,12))
sns.barplot(x=fea,y=fea.index)
plt.tight_layout()
plt.show()