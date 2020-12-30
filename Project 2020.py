import pandas as pd
import matplotlib as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

price = pd.read_csv('lumberprice.csv')
cottage = pd.read_csv('cottage.csv')
home = pd.read_csv('home.csv')
lumber = pd.read_csv('lumber.csv')
lumber_price = pd.read_csv('lumber_price.csv')
wooden_home = pd.read_csv('wooden_home.csv')

df = pd.concat([cottage, wooden_home, home, lumber, lumber_price], axis = 1).reset_index()
del df['week1']

dates = []
for i in range(cottage.shape[0]):
    date = cottage.iloc[i]['week'].split('-')
    dates.append(date)

df[['year', 'month', 'day']] = dates

df_mean = df.groupby(['year', 'month']).mean().reset_index()[['cttg', 'whome', 'lumber', 'lumber_price']]
#df_mean

price = pd.DataFrame(price.to_numpy()[::-1])
del price[0]
#price

delta = 2
df_array = np.array(df_mean)
df_shifted = df_array[delta+1:] - df_array[delta:-1]
data = pd.DataFrame(df_shifted)
data['oldprice'] = price[delta:-1].to_numpy()
data['price'] = price[delta+1:].to_numpy()
#data

N = 100
res = []
prices = data['price']
features = data.drop('price', axis = 1)
for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(features, prices,
                                                    test_size=0.2, random_state=N*10+N*N)
    model = LinearRegression()
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    res.append(mean_absolute_error(y_test, y_pred))
print(sum(res)/N)
print(model.score(X_test, y_test))
print(model.coef_)