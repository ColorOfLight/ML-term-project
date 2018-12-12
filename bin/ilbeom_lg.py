import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Ilbeom_Linear(object):
    
    def __init__(self):
        self.lm = None
        self.apartment_map = None
        
    def fit(self,X,y):
        
        drop_columns = ['latitude', 'longtitude', 'altitude', '1st region id', '2nd region id', 'road id',
                    
                        'parking lot limit', 'parking lot area', 'parking lot external',
                        
                        'management fee', 'households', 'age of residents', 'builder id', 'completion date', 'built year',

                        'schools', 'bus stations', 'subway stations']

        data = X.drop(columns = drop_columns)
        data = data.assign(price=y)
        data = data.assign(pure_price=data['price'])
        data = data.assign(apartment_price=data['price'])
        data['contract date'] = pd.to_datetime(data['contract date'])
        data['contract date'] = pd.to_numeric(data['contract date'] - data['contract date'].min())
        data['contract date'] = data['contract date']/data['contract date'].max()
        data['angle'] = np.sin(data['angle'])
        datalist=[data[data.apartment_id==i] for i in set(data['apartment_id'])]
        newlist = []
        for i in datalist:
            if len(i['angle'])>0:
                newlist.append(i)
        data = pd.concat(newlist)
        
        def get_X_y(data):
            y = data['pure_price']
            X = data.drop(columns=['apartment_price','pure_price','price', 'apartment_id'])
            return X, y

        def get_pure_apartment_price(data,n):
            datalist=[data[data.apartment_id==i] for i in set(data['apartment_id'])]
            lengthlist = [len(i['angle']) for i in datalist]
            #idlist = [i['apatment_id'].iloc[0] for i in datalist]
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
        
        
        N = 2
        data = get_pure_apartment_price(data,N)
        apartment_map=dict()
        for i in data['apartment_id']:
            if i not in apartment_map:
                price=data[data.apartment_id==i]
                apartment_map[i]=price['apartment_price'].iloc[0]
        lm = LinearRegression()
        y = data['price']
        X=data.drop(columns=['price','pure_price','apartment_id'])
        lm.fit(X,y)
        self.lm = lm
        self.apartment_map = apartment_map
    def predict(self,X):
        apartment_map = self.apartment_map
        def preprocess_test_data(data, apartment_map):
            apartment_price = []
            for i in data['apartment_id']:
                if i not in apartment_map:
                    pass
                else:
                    apartment_price.append(apartment_map[i])
            apartment_price.sort()
            median_value = apartment_price[len(apartment_price)//2]
            new_price = []
            for i in data['apartment_id']:
                if i not in apartment_map:
                    new_price.append(median_value)
                else:
                    new_price.append(apartment_map[i])                
            data['apartment_price'] = new_price
            return data
        lm = self.lm
        drop_columns = ['latitude', 'longtitude', 'altitude', '1st region id', '2nd region id', 'road id',
    
                            'parking lot limit', 'parking lot area', 'parking lot external',
    
                            'management fee', 'households', 'age of residents', 'builder id', 'completion date', 'built year',
    
                            'schools', 'bus stations', 'subway stations']
    
        data = X.drop(columns = drop_columns)
        data['contract date'] = pd.to_datetime(data['contract date'])
        data['contract date'] = pd.to_numeric(data['contract date'] - data['contract date'].min())
        data['contract date'] = data['contract date']/data['contract date'].max()
        data['angle'] = np.sin(data['angle'])
        data = data.assign(apartment_price=data['angle'])
        X = preprocess_test_data(data, apartment_map)
        X = X.drop(columns = ['apartment_id'])
        return lm.predict(X)
# def test():
#     names = ['contract date', 'latitude', 'longtitude', 'altitude', '1st region id', '2nd region id', 'road id',
             
#             'apartment_id', 'floor', 'angle', 'area', 'parking lot limit', 'parking lot area', 'parking lot external',
            
#             'management fee', 'households', 'age of residents', 'builder id', 'completion date', 'built year',
            
#             'schools', 'bus stations', 'subway stations', 'price']
    
#     data = pd.read_csv('./data_train_original.csv', names=names)
#     y = data['price']
#     X = data.drop(columns =['price'])
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 110)
#     model = Linear_Regression_with_drop_features()
#     model.fit(X_train,y_train)
#     predictions = model.predict(X_test)
#     result = 0
#     for i in range(len(X_test)):
#         result += abs((y_test.iloc[i]-predictions[i])/y_test.iloc[i])
#     print(1-result/len(X_test))
    
# def main():
#     test()
# main()