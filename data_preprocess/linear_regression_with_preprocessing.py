import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

names = [ 'apartment_id', 'contract date', 'floor', 'angle', 'area', 'price']

data = pd.read_csv('./data_train2.csv', names=names)
data = data.assign(pure_price=data['price'])
data = data.assign(apartment_price=data['price'])
data['contract date'] = pd.to_datetime(data['contract date'])
data['contract date'] = pd.to_numeric(data['contract date'] - data['contract date'].min())
data['contract date'] = data['contract date']/data['contract date'].max()
data['floor'] = -np.power(8-data['floor'],2)
data['angle'] = np.sin(data['angle'])

data_train, data_test = train_test_split(data, test_size = 0.2, random_state = 102)


def get_X_y(data):

    y = data['pure_price']
    X = data.drop(columns=['apartment_price','pure_price','price', 'apartment_id'])
    return X, y


def get_pure_apartment_price(data,n):
    
    datalist=[data[data.apartment_id==i] for i in set(data['apartment_id'])]
    Xylist = [get_X_y(data) for data in datalist]
    model_list = [LinearRegression() for i in range(len(Xylist))]
    
    for i in range(len(model_list)):
        X,y = Xylist[i]
        model_list[i].fit(X,y)
        
    intercept_list = [[model_list[i].intercept_]*len(datalist[i]['apartment_price']) for i in range(len(model_list))]
    l = []
    for i in intercept_list:
        l.extend(i)
    data['apartment_price']=l
    data['pure_price']=data['price']-data['apartment_price']
    
    for i in range(n):
        total_model = LinearRegression()
        X,y = get_X_y(data)
        total_model.fit(X,y)
        coeffiecient = total_model.coef_
        data['apartment_price'] = data['price']-data.drop(columns=['apartment_price','pure_price','price', 'apartment_id']).dot(coeffiecient)
        datalist=[data[data.apartment_id==i] for i in set(data['apartment_id'])]
        med_list = [[np.median(datalist[i])]*len(datalist[i]['apartment_price']) for i in range(len(datalist))]
        l=[]
        for i in med_list:
            l.extend(i)
        data['apartment_price']=l
        data['pure_price']=data['price']-data['apartment_price']
    
    return data

def preprocess_test_data(data, apartment_map):
    apartment_price = []
    for i in data['apartment_id']:
        if i not in apartment_map:
            print(1)
            apartment_price.append(0)
        else:
            apartment_price.append(apartment_map[i])
    data['apartment_price'] = apartment_price
    return data

def train_test(X_train,y_train,X_test,y_test):
    
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)
    result = 0          
    for i in range(len(X_test)):
        result += abs((y_test.iloc[i]-predictions[i])/y_test.iloc[i])
    return 1-result/len(X_test)

N = 2
data_train=data
data_train = get_pure_apartment_price(data_train,N)
apartment_map=dict()
for i in data_train['apartment_id']:
    if i not in apartment_map:
        price=data_train[data_train.apartment_id==i]
        apartment_map[i]=price['apartment_price'].iloc[0]
for i in apartment_map:
    print(i,apartment_map[i])
data_test = preprocess_test_data(data_test, apartment_map)
y_train = data_train['price']
X_train = data_train.drop(columns=['apartment_id','pure_price','price'])
y_test = data_test['price']
X_test = data_test.drop(columns=['apartment_id','pure_price','price'])
print(train_test(X_train,y_train,X_test,y_test))
