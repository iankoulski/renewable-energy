# module model

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn import metrics, ensemble, model_selection
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy as np
import datetime
from dateutil import relativedelta
import os
import io
import json
import base64
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.ar_model import AutoReg

np.random.seed(42)
tf.random.set_seed(42)

def buildProphet(train_data_path, test_data_path):    
    print("\nBuilding Prophet model ...")
    df = pd.read_csv(train_data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP',inplace=True)
    y = df['RENEWABLES_PCT']
    daily = y.resample('24H').mean()
    dd = pd.DataFrame(daily)
    dd.reset_index(inplace=True)
    dd.columns = ['ds','y']
    mR = Prophet(daily_seasonality=False)
    mR.fit(dd)
    futureR=mR.make_future_dataframe(periods=365*5)
    forecastR=mR.predict(futureR)

    rmse = -1.0
    if len(test_data_path) > 0:
        dft = pd.read_csv(test_data_path)
        dft['TIMESTAMP'] = dft['TIMESTAMP'].astype('datetime64')
        dft.set_index('TIMESTAMP',inplace=True)
        dft_start_datetime = min(dft.index)
        dft_end_datetime = max(dft.index)
        actual_mean = dft['RENEWABLES_PCT'].resample('24H').mean()
        predicted_mean = forecastR.loc[(forecastR['ds'] >= dft_start_datetime) & (forecastR['ds'] <= dft_end_datetime)]
        predicted_mean.set_index('ds', inplace=True)
        actual_mean = actual_mean[min(predicted_mean.index):]
        mse = metrics.mean_squared_error(actual_mean, predicted_mean.yhat)
        rmse = sqrt(mse)
        print(str.format("Prophet RMSE: {:.2f}", rmse))
    return rmse

def predictProphet(data_path,periods):
    print("\nTraining prophet model with full dataset ...")
    df = pd.read_csv(data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP',inplace=True)
    y = df['RENEWABLES_PCT']
    daily = y.resample('24H').mean()
    dd = pd.DataFrame(daily)
    dd.reset_index(inplace=True)
    dd.columns = ['ds','y']
    m = Prophet(daily_seasonality=False)
    m.fit(dd)
    future=m.make_future_dataframe(periods=periods)
    print(str.format("\nPredicting with prophet model for {0} days ({1} years) ...",periods, int(periods/365)))
    plt.subplot(1,1,1)
    forecast=m.predict(future)
    fig = m.plot(forecast,ylabel='Renewable Power Production %', xlabel='Date')
    plt.suptitle('\nCA Predicted Renewable Power Production %')
    #plt.title('\nCA Predicted Renewable Power Production %')
    axes = plt.gca()
    wd = os.path.dirname(data_path) + '/../images'
    os.makedirs(wd, exist_ok=True)
    fig.savefig(wd + '/prediction-prophet.png')
    forecast.rename(columns={'ds':'TIMESTAMP'}, inplace=True)
    forecast.set_index('TIMESTAMP',inplace=True)
    prediction = pd.DataFrame({'RENEWABLES_PCT_MEAN':forecast['yhat'].resample('1Y').mean(),'RENEWABLES_PCT_LOWER':forecast['yhat_lower'].resample('1Y').mean(),'RENEWABLES_PCT_UPPER':forecast['yhat_upper'].resample('1Y').mean()})
    return prediction

def rmse_calc(actual,predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

def transformDataset(df):
    # Add pct from one and two days ago as well as difference in yesterday-1 and yesterday-1
    df['YESTERDAY'] = df['RENEWABLES_PCT'].shift()
    df['YESTERDAY_DIFF'] = df['YESTERDAY'].diff()
    df['YESTERDAY-1']=df['YESTERDAY'].shift()
    df['YESTERDAY-1_DIFF'] = df['YESTERDAY-1'].diff()
    df=df.dropna()
    x_train=pd.DataFrame({'YESTERDAY':df['YESTERDAY'],'YESTERDAY_DIFF':df['YESTERDAY_DIFF'],'YESTERDAY-1':df['YESTERDAY-1'],'YESTERDAY-1_DIFF':df['YESTERDAY-1_DIFF']})
    y_train = df['RENEWABLES_PCT']
    return x_train,y_train

def buildRandomForestRegression(train_data_path,test_data_path):    
    print("\nBuilding Random Forest Regression Model ...")
    
    print("Preparing training dataset ...")
    df = pd.read_csv(train_data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP',inplace=True)
    df = df.resample('1M').mean()
    x_train, y_train = transformDataset(df)

    print("Preparing testing dataset ...")
    dt = pd.read_csv(test_data_path)
    dt['TIMESTAMP'] = dt['TIMESTAMP'].astype('datetime64')
    dt.set_index('TIMESTAMP',inplace=True)
    x_test, y_test = transformDataset(dt)

    print("Searching for best regressor ...")
    model = ensemble.RandomForestRegressor()
    param_search = {
        'n_estimators': [100],
        'max_features': ['auto'],
        'max_depth': [10]
    }
    tscv = model_selection.TimeSeriesSplit(n_splits=2)
    rmse_score = metrics.make_scorer(rmse_calc, greater_is_better = False)
    gsearch = model_selection.GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring=rmse_score)
    gsearch.fit(x_train, y_train)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    y_true = y_test.values
    print("Predicting with best regressor ...")
    y_pred = best_model.predict(x_test)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    print(str.format("Random Forest Regression RMSE: {:.2f}", rmse))
    return rmse

def predictRandomForestRegression(data_path,periods):
    print("\nTraining Random Forest Regression model with full dataset ...")
    df = pd.read_csv(data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP',inplace=True)
    dfmean = df.resample('1M').mean()
    dfmin = df.resample('1M').min()
    dfmax = df.resample('1M').max()
    x_train,y_train = transformDataset(dfmean)
    xmin_train, ymin_train = transformDataset(dfmin)
    xmax_train, ymax_train = transformDataset(dfmax)

    model = ensemble.RandomForestRegressor()
    model_min = ensemble.RandomForestRegressor()
    model_max = ensemble.RandomForestRegressor()
    param_search = {
        'n_estimators': [100],
        'max_features': ['auto'],
        'max_depth': [10]
    }
    tscv = model_selection.TimeSeriesSplit(n_splits=2)
    rmse_score = metrics.make_scorer(rmse_calc, greater_is_better = False)
    gsearch = model_selection.GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring=rmse_score)
    gsearch_min = model_selection.GridSearchCV(estimator=model_min, cv=tscv, param_grid=param_search, scoring=rmse_score)
    gsearch_max = model_selection.GridSearchCV(estimator=model_max, cv=tscv, param_grid=param_search, scoring=rmse_score)
    gsearch.fit(x_train, y_train)
    gsearch_min.fit(xmin_train, ymin_train)
    gsearch_max.fit(xmax_train, ymax_train)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    best_model_min = gsearch_min.best_estimator_
    best_model_max = gsearch_max.best_estimator_
    
    print("\nPredicting with Random Forest regressor ...")
    prediction = pd.DataFrame(columns=['TIMESTAMP','RENEWABLES_PCT'])
    l = len(x_train)
    x_pred = x_train.iloc[[l-1]]
    y_pred = best_model.predict(x_pred)
    xmin_pred = xmin_train.iloc[[l-1]]
    ymin_pred = best_model_min.predict(xmin_pred)
    xmax_pred = xmax_train.iloc[[l-1]]
    ymax_pred = best_model_max.predict(xmax_pred)
    prediction = prediction.append({'TIMESTAMP':x_pred.index[0],'RENEWABLES_PCT_MEAN':y_pred[0],'RENEWABLES_PCT_LOWER':ymin_pred[0],'RENEWABLES_PCT_UPPER':ymax_pred[0]}, ignore_index=True)
    for i in range(1,periods):
        ti = prediction.iloc[i-1]['TIMESTAMP'] + pd.offsets.DateOffset(months=1)
        xi_pred = pd.DataFrame({'YESTERDAY':y_pred,'YESTERDAY_DIFF':y_pred-x_pred['YESTERDAY'],'YESTERDAY-1':x_pred['YESTERDAY'],'YESTERDAY-1_DIFF':x_pred['YESTERDAY_DIFF']})
        yi_pred = best_model.predict(xi_pred)
        xmini_pred = pd.DataFrame({'YESTERDAY':ymin_pred,'YESTERDAY_DIFF':ymin_pred-xmin_pred['YESTERDAY'],'YESTERDAY-1':xmin_pred['YESTERDAY'],'YESTERDAY-1_DIFF':xmin_pred['YESTERDAY_DIFF']})
        ymini_pred = best_model.predict(xmini_pred)
        xmaxi_pred = pd.DataFrame({'YESTERDAY':ymax_pred,'YESTERDAY_DIFF':ymax_pred-xmax_pred['YESTERDAY'],'YESTERDAY-1':xmax_pred['YESTERDAY'],'YESTERDAY-1_DIFF':xmax_pred['YESTERDAY_DIFF']})
        ymaxi_pred = best_model.predict(xmaxi_pred)
        prediction = prediction.append({'TIMESTAMP':ti,'RENEWABLES_PCT_MEAN':yi_pred[0],'RENEWABLES_PCT_LOWER':ymini_pred[0],'RENEWABLES_PCT_UPPER':ymaxi_pred[0]}, ignore_index=True)
        x_pred = xi_pred
        y_pred = yi_pred
        xmin_pred = xmini_pred
        ymin_pred = ymini_pred
        xmax_pred = xmaxi_pred
        ymax_pred = ymaxi_pred

    prediction.set_index('TIMESTAMP',inplace=True)
    prediction = prediction.resample('1Y').mean()
    p = prediction.plot()
    p.set_title('CA Predicted Renewables % by Random Forest Regression')
    p.set_ylabel('Renewables %')
    wd = os.path.dirname(data_path) + '/../images'
    os.makedirs(wd, exist_ok=True)
    plt.savefig(wd + '/prediction-randomforest.png')

    return prediction

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = model_selection.mean_absolute_error(test[:, -1], predictions)
	return error, test[:, 1], predictions

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict(np.asarray([testX]))
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = metrics.mean_squared_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

def buildXGBoostRegression(train_data_path,test_data_path):
    print("\nBuilding XGBoost Regression model ...")
    df = pd.read_csv(train_data_path)
    dt = pd.read_csv(test_data_path)
    df = df.append(dt)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP',inplace=True)
    dfmean = df.resample('1Y').mean()
    dmean = series_to_supervised(dfmean[['RENEWABLES_PCT']], n_in=2, n_out=1, dropnan=True)
    # transform list into array
    mse, y, yhat = walk_forward_validation(dmean, 8)
    rmse = sqrt(mse)
    print(str.format("XGBoostRegression RMSE: {:.2f}", rmse))
    return rmse

def buildLSTM(train_data_path,test_data_path):
    print("\nBuilding LSTM Model ...")
    time_steps = 3
    print("Preparing training dataset ...")
    df = pd.read_csv(train_data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP',inplace=True)
    df = df[['RENEWABLES_PCT']]
    df = df.resample('1M').mean()
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    scaling_model = scaler.fit(df)
    df = scaling_model.transform(df)
    daily_train = pd.DataFrame(df, columns=['RENEWABLES_PCT']).reset_index()
    x_train, y_train = create_lstm_dataset(pd.DataFrame({'ROW':range(0,len(daily_train)),'RENEWABLES_PCT':daily_train['RENEWABLES_PCT']}), daily_train['RENEWABLES_PCT'], time_steps)

    print("Preparing testing dataset ...")
    dt = pd.read_csv(test_data_path)
    dt['TIMESTAMP'] = dt['TIMESTAMP'].astype('datetime64')
    dt.set_index('TIMESTAMP',inplace=True)
    dt = dt[['RENEWABLES_PCT']]
    daily_test = dt.resample('1M').mean()
    daily_test = scaling_model.transform(dt)
    daily_test = pd.DataFrame(daily_test, columns=['RENEWABLES_PCT']).reset_index()
    x_test, y_test = create_lstm_dataset(pd.DataFrame({'ROW':range(0,len(daily_test)),'RENEWABLES_PCT':daily_test['RENEWABLES_PCT']}), daily_test['RENEWABLES_PCT'], time_steps)

    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=128,input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))

    history = model.fit(x_train, y_train, epochs=50, batch_size=12, validation_split=0.1, verbose=1, shuffle=False)

    y_pred = model.predict(x_test)
    y_pred = scaling_model.inverse_transform(y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print(str.format("LSTM RMSE: {:.2f}", rmse))
    return rmse

def create_lstm_dataset(x, y, time_steps=1):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        v = x.iloc[i:(i + time_steps)].values
        xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(xs), np.array(ys)

def buildARModel(train_data_path, test_data_path):
    print("\nBuilding Auto-Regression Model ...")
    df = pd.read_csv(train_data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP', inplace=True)
    df = df.resample('1M').mean()
    train_series = df['RENEWABLES_PCT']
    dt = pd.read_csv(test_data_path)
    dt['TIMESTAMP'] = dt['TIMESTAMP'].astype('datetime64')
    dt.set_index('TIMESTAMP', inplace=True)
    dt = dt.resample('1M').mean()
    test_series = dt['RENEWABLES_PCT']
    model = AutoReg(train_series, lags=len(test_series)-1)
    model_fit = model.fit()
    print('Coefficients: %s' % model_fit.params)
    predictions = model_fit.predict(start=len(train_series), end = len(train_series)+len(test_series)-1, dynamic=False)
    rmse = sqrt(metrics.mean_squared_error(test_series,predictions))
    plt.plot(test_series)
    plt.plot(predictions, color='red')
    #plt.show()
    print('ARModel RMSE: %.2f' % rmse)
    return rmse

def predictARModel(data_path, periods):
    print("\nTraining Auto-Regression Model with full dataset ...")
    df = pd.read_csv(data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index(['TIMESTAMP'], inplace=True)
    df = df.resample('1M').mean()
    train_series = df['RENEWABLES_PCT']
    model = AutoReg(train_series, lags=12)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train_series), end = len(train_series)+periods-1, dynamic=False)
    plt.plot(train_series)
    plt.plot(predictions, color='red')
    #plt.show()
    return predictions
    
def buildLinearModel(train_data_path, test_data_path):
    print("\nBuilding Linear Model ...")
    df = pd.read_csv(train_data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index(['TIMESTAMP'], inplace=True)
    series = df
    series = series.resample('1M').mean()
    series['ROW'] = range(0,len(series))
    z = np.polyfit(series['ROW'], series['RENEWABLES_PCT'],1)
    p = np.poly1d(z)
    print('Trend: y=%.6fx+(%.6f)'%(z[0],z[1]))
    dt = pd.read_csv(test_data_path)
    dt['TIMESTAMP'] = dt['TIMESTAMP'].astype('datetime64')
    dt.set_index(['TIMESTAMP'], inplace=True)
    dt = dt.resample('1M').mean()
    dt.reset_index(inplace=True)
    ind = range(0,len(dt))
    x = range(len(series),len(series)+len(dt))
    prediction = pd.DataFrame({'ROW':ind,'TIMESTAMP':dt['TIMESTAMP'],'RENEWABLES_PCT':p(x)}, index=ind)
    prediction.set_index(['TIMESTAMP'], inplace=True)
    predicted = prediction['RENEWABLES_PCT']
    actual = dt['RENEWABLES_PCT']
    mse = metrics.mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    print("Linear model RMSE: %.2f" % rmse)
    return rmse

def predictLinearModel(data_path, periods):
    print("\nPredicting with linear model ...")
    df = pd.read_csv(data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index(['TIMESTAMP'], inplace=True)
    dfmean = df
    dfmax = df
    dfmin = df
    dfmean = dfmean.resample('1M').mean()
    dfmax = dfmax.resample('1M').max()
    dfmin = dfmin.resample('1M').min()
    series = dfmean['RENEWABLES_PCT']
    seriesmax = dfmax['RENEWABLES_PCT']
    seriesmin = dfmin['RENEWABLES_PCT']
    series = series.reset_index()
    seriesmax = seriesmax.reset_index()
    seriesmin = seriesmin.reset_index()
    series['ROW'] = range(0,len(series))
    seriesmax['ROW'] = range(0,len(seriesmax))
    seriesmin['ROW'] = range(0,len(seriesmin))
    z = np.polyfit(series['ROW'],series['RENEWABLES_PCT'],1)
    p = np.poly1d(z)
    zmax = np.polyfit(seriesmax['ROW'],seriesmax['RENEWABLES_PCT'],1)
    pmax = np.poly1d(zmax)
    zmin = np.polyfit(seriesmin['ROW'],seriesmin['RENEWABLES_PCT'],1)
    pmin = np.poly1d(zmin)
    ind = range(0,len(series))
    prediction = pd.DataFrame({'ROW':ind,'TIMESTAMP':series['TIMESTAMP'],'RENEWABLES_PCT':p(ind)}, index=ind)
    predictionmax = pd.DataFrame({'ROW':ind,'TIMESTAMP':seriesmax['TIMESTAMP'],'RENEWABLES_PCT':pmax(ind)}, index=ind)
    predictionmin = pd.DataFrame({'ROW':ind,'TIMESTAMP':seriesmin['TIMESTAMP'],'RENEWABLES_PCT':pmin(ind)}, index=ind)
    l = len(series)
    last_date = series['TIMESTAMP'][l-1]
    for i in range(1,periods+1):
        pred = pd.DataFrame({'ROW':l+i-1,'TIMESTAMP':last_date+relativedelta.relativedelta(months=i),'RENEWABLES_PCT':p(l+i-1)}, index=[l+i-1])
        prediction = prediction.append(pred)    
        predmax = pd.DataFrame({'ROW':l+i-1,'TIMESTAMP':last_date+relativedelta.relativedelta(months=i),'RENEWABLES_PCT':pmax(l+i-1)}, index=[l+i-1])
        predictionmax = predictionmax.append(predmax)    
        predmin = pd.DataFrame({'ROW':l+i-1,'TIMESTAMP':last_date+relativedelta.relativedelta(months=i),'RENEWABLES_PCT':pmin(l+i-1)}, index=[l+i-1])
        predictionmin = predictionmin.append(predmin)  
    df.reset_index(inplace=True)
    plt.rcParams["figure.figsize"] = (12,9)
    plt.subplot(1,1,1)
    plt.scatter(df['TIMESTAMP'],df['RENEWABLES_PCT'], color='blue', label='Hourly data')
    plt.scatter(series['TIMESTAMP'],series['RENEWABLES_PCT'],color='yellow', label='Monthly mean')
    plt.title('CA Renewables Overall Production Trend', fontsize=18)
    plt.ylabel('Renewables %', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.plot(prediction['TIMESTAMP'],prediction['RENEWABLES_PCT'],color='black',linestyle='dashed', linewidth=2, label='Monthly mean trend')
    plt.plot(predictionmax['TIMESTAMP'],predictionmax['RENEWABLES_PCT'],color='red',linestyle='dashed', linewidth=1, label='Monthly max trend')
    plt.plot(predictionmin['TIMESTAMP'],predictionmin['RENEWABLES_PCT'],color='red',linestyle='dashed', linewidth=1, label='Monthly min trend')
    plt.grid(b=True, which='major', axis='both')
    plt.legend(loc='upper left')
    wd=os.path.dirname(data_path) + '/..'
    os.makedirs(wd+'/images', exist_ok=True)
    plt.savefig(wd+'/images/prediction-linear.png')
    #plt.show()
    print('Trend: y=%.6fx+(%.6f)'%(z[0],z[1]))
    print('Upper: y=%.6fx+(%.6f)'%(zmax[0],zmax[1]))
    print('Lower: y=%.6fx+(%.6f)'%(zmin[0],zmin[1]))
    print("Saving Linear Model Predictions ...")
    predictions = prediction
    predictions['RENEWABLES_PCT_UPPER'] = predictionmax['RENEWABLES_PCT']
    predictions['RENEWABLES_PCT_LOWER'] = predictionmin['RENEWABLES_PCT']
    predictions.to_csv(os.path.dirname(data_path)+'/prediction-linear.csv')
    return predictions

def predictWithLinearAndBestModel(techniques, preprocessed_data_path):
    print("\nPredicting with Linear and Best model ...")
    prediction = predictLinearModel(preprocessed_data_path, 12*29+2)
    
    print("\nComparing models ...")
    if 'Linear' in techniques.keys():
        techniques.pop('Linear')

    s = sorted(techniques.items(), key=lambda x: x[1])
    bestModel = s[0][0]
    print(str.format('The best model is {}', bestModel))
    if bestModel == 'Prophet':
        prediction = predictProphet(preprocessed_data_path,365*30)
    elif bestModel == 'RandomForest':
        prediction = predictRandomForestRegression(preprocessed_data_path,12*30)
    else:
        pass

    return prediction

def visualizePrediction(wd, prediction):
    
    # Log output
    print("Visualizing prediction ...")
    print("Prediction:")
    print(prediction)
    
    prediction.reset_index(inplace=True)
    print("\nPrediction for CA Renewables % in Key Years:")
    prediction2030 = prediction[prediction['TIMESTAMP'] == pd.to_datetime('12-31-2030')]
    prediction2045 = prediction[prediction['TIMESTAMP'] == pd.to_datetime('12-31-2045')]
    print(str.format("   Prophet prediction 12/31/2030:"))
    print(str.format("      low: {:.2f}, mean: {:.2f}, high: {:.2f}",prediction2030['RENEWABLES_PCT_LOWER'].values[0],prediction2030['RENEWABLES_PCT_MEAN'].values[0],prediction2030['RENEWABLES_PCT_UPPER'].values[0]))
    print(str.format("   Prophet prediction 12/31/2045:"))
    print(str.format("      low: {:.2f}, mean: {:.2f}, high: {:.2f}",prediction2045['RENEWABLES_PCT_LOWER'].values[0],prediction2045['RENEWABLES_PCT_MEAN'].values[0],prediction2045['RENEWABLES_PCT_UPPER'].values[0]))
    
    # Table output
    os.makedirs(wd + '/data/', exist_ok=True)
    prediction.to_csv(wd+'/data/prediction-best.csv', index=False)
    s = io.StringIO()
    prediction.to_csv(s, index=False)
    
    # Plot output
    prediction.set_index('TIMESTAMP', inplace=True)
    plot = prediction.plot()
    plot.set_title('Predicted CA Renewables Procudtion %')
    plot.set_xlabel('Date')
    plot.set_ylabel('Renewables %')
    plt.savefig(wd + '/images/prediction-best.png')
    with open(wd + '/images/prediction-best.png', 'rb') as image_file:
        image = base64.b64encode(image_file.read())
    
    # Metadata
    metadata = {
        'outputs': [{
            'type': 'table',
            'storage': 'inline',
            'format': 'csv',
            'header': ['TIMESTAMP','RENEWABLES_PCT_MEAN','RENEWABLES_PCT_LOWER','RENEWABLES_PCT_UPPER'],
            'source': s.getvalue()
        },
        {
            'type': 'web-app',
            'storage': 'inline',
            'source': '<html><head><title>Plot</title></head><body><div><img src="data:image/png;base64, ' + image.decode('ascii') +'" /></div></body></html>'
        }]
    }
    metadata_path = wd + '/data/mlpipeline-ui-metadata.json'
    with open( metadata_path, 'w') as f:
        json.dump(metadata,f)
    
