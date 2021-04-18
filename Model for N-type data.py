# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:24:29 2021

@author: Sunmarg Das
"""
import pandas as pd
df = pd.read_excel (r'Thermocouple_data.xlsx',sheet_name='N_Type')
df.drop(['Temp'],axis=1,inplace=True)
df['Error']=abs(df['Error'])
'''
def replace_missing (attribute):
    return attribute.interpolate(inplace=True)
replace_missing(df['Error'])

'''
data=df.copy()
data.drop([25],axis=0,inplace=True)
X_f=data.iloc[:,0:-1].values

X_f=X_f[-5:]

df=df[df['CI'].notna()]
df['Error'].fillna((df['Error'].mean()), inplace=True)
y=df.iloc[:,-1].values
X=df.iloc[:,0:-1].values

X_train=X[:-5,:]
y_train=y[:-5]
X_test=X[-5:,:]
y_test=y[-5:]

'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_p= classifier.predict(X_f)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
'''
import xgboost as xgb
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train)
reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                early_stopping_rounds=50, #stop if 50 consequent rounds without decrease of error
                verbose=False)
X_norm=reg.predict(X_test)
    
X_pred=reg.predict(X_f)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=1)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)
X_f_1=X_f[:,0]
pred_1=pol_reg.predict(poly_reg.fit_transform(X_test))
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)
X_f_1=X_f[:,0]
pred_2=pol_reg.predict(poly_reg.fit_transform(X_test))
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import r2_score

print(metrics.mean_absolute_error( y_test, X_pred))
print(metrics.mean_squared_error(y_test, X_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, X_pred)))
predict = np.array([ 8, 11, 10, 10, 12])
actual = np.array([7,7,7,7,7])
corr_matrix = np.corrcoef(actual, predict)
corr = corr_matrix[0,1]
R_sq = corr**2
print(r2_score(y_test,pred_2))

'''
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))
# Fit the scaler using available training data
input_scaler = scaler_x.fit(X_train)
y_train=y_train.reshape(-1,1)
output_scaler = scaler_y.fit(y_train)
# Apply the scaler to training data
train_y_norm = output_scaler.transform(y_train)
train_x_norm = input_scaler.transform(X_train)
# Apply the scaler to test data
y_test=y_test.reshape(-1,1)
test_y_norm = output_scaler.transform(y_test)
test_x_norm = input_scaler.transform(X_test)
import numpy as np
#Convert to numpy arrays
X_train, y_train = np.array(train_x_norm), np.array(train_y_norm)

#Reshape the data into 3-D array
x_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import GRU

# Initialising the RNN
model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and Dropout layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and Dropout layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and and Dropout layer
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
# For Full connection layer we use dense
# As the output is 1D so we use unit=1
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 30, batch_size = 50)









def create_model(units, m):
    model = Sequential()
    model.add(GRU (units = units, return_sequences = True,
                input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.2))
    model.add(m (units = units))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model
model_gru = create_model(64,GRU)
'''
