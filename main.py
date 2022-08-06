from cgi import test
from turtle import ycor
import pandas as pd
df = pd.read_csv("120 csv.csv")

print(df.head())

from sklearn.model_selection import train_test_split

X = df[["age", "education-num"]]
y = df["income"]

x_train , x_test , y_train, y_test = train_test_split(X , y , test_size = 0.25, random_state = 42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

model = GaussianNB()
model.fit(x_train, y_train)

y_pred_1 = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred_1)
print(accuracy)

