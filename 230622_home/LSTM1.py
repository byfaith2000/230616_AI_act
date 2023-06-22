class L:
    
    def loadNconvert_datasrt(self):
        
        
        
        
        
        
        print()
        
        
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
        history = self.model.fit(self.x_train, self.y_train,epochs=epochs,batch_size=batch_size,validation_data=(self.x_valid,self.y_valid),callbacks=[early_stop, check point])
        
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
        mse = round(mean_squared_error(self.test_label pred), 6) # MSE
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            