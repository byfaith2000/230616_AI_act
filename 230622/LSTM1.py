import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import os

class L:
   scale_cols = [] # 
   feature_cols = []
   label_cols = [] 
   TEST_SIZE = 150
   WINDOW_SIZE = 20
   
   def __init__(self, data_path, header=0, encoding='cp949', sep=None, na_values=None, parse_dates=None):
      self.data_path = data_path
      self.dataset = pd.read_csv(self.data_path, header=header, encoding=encoding, sep=sep, na_values=na_values, parse_dates=parse_dates)

   def loadNconvert_dataset(self):
      self.date_name = self.dataset.columns[0]

      self.dataset['Date'] = pd.to_datetime(self.dataset[self.date_name])

      self.dataset['Year'] = self.dataset['Date'].dt.year
      self.dataset['Month'] = self.dataset['Date'].dt.month
      self.dataset['Weekday'] = self.dataset['Date'].dt.weekday
      self.dataset['Day'] = self.dataset['Date'].dt.day
    
      if len(self.dataset[self.dataset['Date'].dt.hour==0]) != len(self.dataset):
         self.dataset['Hour'] = self.dataset['Date'].dt.hour

   def info(self):
      print(self.dataset.info())

   def show_dataset_infos(self, y, x1 = 'Year', x2 = 'Month', x3 = 'Day', x4 = 'Hour', x5 = 'Year', h5 = 'Month'):
       self.fig = plt.figure(figsize=[12, 10])

       self.fig.add_subplot(2,2,1)
       sns.barplot(x=x1, y=y, data=self.dataset.groupby(x1)[y].mean().reset_index(), palette='Paired')
       self.fig.add_subplot(2,2,2)
       sns.barplot(x=x2, y=y, data=self.dataset.groupby(x2)[y].mean().reset_index(), palette='Paired')

       self.fig.add_subplot(2,2,3)
       sns.barplot(x=x3,y=y,data=self.dataset.groupby(x3)[y].mean().reset_index(), palette='Paired')

       if len(self.dataset[self.dataset['Date'].dt.hour==0]) !=len(self.dataset):
          self.fig.add_subplot(2,2,4)
          sns.barplot(x=x4,y=y,data=self.dataset.groupby(x4)[x4][y].mean().reset_index(), palette='Paired')

          self.fig = plt.figure(figsize=[12,10])
          sns.pointplot(x=x5,y=y,hue=h5,data=self.dataset.groupby([x5,h5])[y].mean().reset_index(), palette='tab10')
          plt.xlabel(x5)
          plt.ylabel(y)

          self.fig = plt.figure(figsize=[30,30])
          sns.heatmap(self.dataset.corr(), annot=True, square=True, cmap='Reds')
          plt.show()

   def normalize_dataset(self):
       self.normalized=False
       if not self.scale_cols:
          return
       
       self.dataset.sort_index(ascending=False).reset_index(drop=True)
       self.scaler = MinMaxScaler()

       self.dataset_scaled = self.scaler.fit_transform(self.dataset[self.scale_cols])
       self.dataset_scaled = pd.DataFrame(self.dataset_scaled)
       self.dataset_scaled.columns = self.scale_cols
       self.normalized = True

   def make_dataset(self, data, label):
       feature_list = []
       label_list= []
       for i in range(len(data) - self.WINDOW_SIZE):
            feature_list.append(np.array(data.iloc[i:i+self.WINDOW_SIZE]))
            label_list.append(np.array(label.iloc[i+self.WINDOW_SIZE]))

       return np.array(feature_list), np.array(label_list)
    
   def create_train_dataset(self):
       
       if not self.normalized:
          return
       
       if not self.feature_cols or not self.label_cols:
          return
       
       self.train_dataset = self.dataset_scaled

       self.train_feature = self.train_dataset[self.feature_cols]
       self.train_label = self.train_dataset[self.label_cols]
       self.train_feauture, self.train_label = self.make_dataset(self.train_feature, self.train_label)
       self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.train_feature, self.train_label, test_size=0.2)

   def create_test_dataset(self):
    
      if not self.normalized:
         return
       
      if not self.feature_cols or not self.label_cols:
         return
       
      self.test_dataset = self.dataset_scaled
      self.test_feature = self.test_dataset[self.feature_cols]
      self.test_label = self.test_dataset[self.label_cols]
      self.test_feature, self.test_label = self.make_dataset(self.test_feature, self.test_label)

   def create_trainNtest_dataset(self, only_train=False, test_dataset=None):
       
       if not self.normalized:
          return
       
       if not self.feature_cols or not self.label_cols:
          return
       
       self.train_dataset = self.dataset_scaled[:-self.TEST_SIZE]

       self.test_dataset = self.dataset_scaled[-self.TEST_SIZE:]

       self.train_feature = self.train_dataset[self.feature_cols]

       self.train_label = self.train_dataset[self.label_cols]

       self.train_label = self.train_dataset[self.label_cols]

       self.train_feature, self.train_label = self.make_dataset(self.train_feature, self.train_label)

       self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.train_feature, self.train_label, test_size=0.2)

       self.test_feature = self.test_dataset[self.feature_cols]

       self.test_label = self.test_dataset[self.label_cols]

       self.test_feature, self.test_label = self.make_dataset(self.test_feature, self.test_label)


   def create_model(self, model_path):
        self.model_path = model_path # 학습 결과물 경로
        self.model = Sequential() # 레이어 층을 선형으로 구성
        self.model.add(LSTM(16,input_shape=(self.WINDOW_SIZE,len(self.feature_cols)),activation='relu',return_sequences=False)) #LSTM 모델 구축
        self.model.add(Dense(1)) #출력층 추가
        #모델을 기계가 이해할 수 있도록 컴파일
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
   def train(self, epochs=200, batch_size=16):
        # EalryStopping 학습 모델 명, 체크포인트 지정 및 학습
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        filename = os.path.join(self.model_path, 'epoch_{epoch:04d}.h5')
        checkpoint = ModelCheckpoint(filename,monitor='val_loss', verbose = 1, save_best_only = True, mode='auto')
        history = self.model.fit(self.x_train, self.y_train,epochs=epochs,batch_size=batch_size,validation_data=(self.x_valid,self.y_valid),callbacks=[early_stop, checkpoint])
        
   def load_weights(self, model_name):
        # 저장된 모델 로드
        self.model.load_weights(os.path.join(self.model_path, model_name))
        
   def test(self):
        #훈련하고 나온 값들을 predict() 함수를 사용하여 예측
        pred = self.model.predict(self.test_feature)
        plt.figure(figsize = (12, 9)) # Figure 크기 지정
        plt.plot(self.test_label, label='actual') # 그래프에 참값을 그립니다. 
        plt.plot(pred, label='prediction') # 그래프에 예측값을 그립니다. 
        plt.legend() # 범례 표시
        plt.show() # Figure 출력
        
   def performance_evaluation(self):
        pred = self.model.predict(self.test_feature)
        mse = round(mean_squared_error(self.test_label, pred), 6) # MSE
        rmse = round(np.sqrt(mse), 6) #RMSE
        
        mae = mean_absolute_error(self.test_label, pred)
        mae = round(mae, 6) #MAE
        mape = 0
        for i in range(len(self.test_label)):
            mape += abs((self.test_label[i] - pred[i]) / self.test_label[i])
        mape = mape * 100 /len(self.test_label)
        mape = round(mape[0], 6) #MAPE
        print(f"MSE = {mse}, RMSE = {rmse}")
        print(f"MAE = {mae}, MAPE = {mape}")




#__init__(self, data_path, header=0, encoding='cp949', sep=None, na_values=None, parse_dates=None)

#info(self)