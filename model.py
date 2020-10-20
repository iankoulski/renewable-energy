import pandas as pd
from fbprophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics, ensemble, model_selection
from math import sqrt
import numpy as np
import datetime
import os
import io
import json
import base64

def buildProphet(train_data_path, test_data_path):    
    # load data
    print("Building Prophet model ...")
    df = pd.read_csv(train_data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP',inplace=True)

    # plot data
    df.head()
    plt.rcParams["figure.figsize"] = (12,9)
    x = df.index
    y = df['RENEWABLES_RATIO']
    daily = y.resample('24H').mean()
    day = daily.index
    hour = df['HOUR'].astype(int)
    plt.subplot(2,1,1)
    plt.scatter(day,daily)
    plt.title('Renewable vs Non-renewable Production Ratio')

    plt.subplot(2, 1, 2)
    sns.boxplot(hour, y)
    plt.title('Renewable vs Non-renewable Power Production Ratio grouped by Hour')
    wd = os.path.dirname(train_data_path)
    plt.savefig(wd + '/renewable-ratio-history.png')
    #plt.show()

    # Predict future renewable energy production using Prophet
    dd = pd.DataFrame(daily)
    dd.reset_index(inplace=True)
    dd.columns = ['ds','y']
    mR = Prophet(daily_seasonality=False)
    mR.fit(dd)
    futureR=mR.make_future_dataframe(periods=365*5)
    forecastR=mR.predict(futureR)
    figR = mR.plot(forecastR,ylabel='Renewable vs Non-renewable Power Production Ratio', xlabel='Date')
    plt.title('Forecasted Renewable vs Non-renewable Power Production Ratio')
    axes = plt.gca()
    print('Future Renewable vs Non-renewable Power Production Ratio')
    figR.savefig(wd + '/renewable-ratio-forecast.png')

    # Caluclate prediction accuracy
    rmse = -1.0
    r2score = -1.0
    if len(test_data_path) > 0:
        dft = pd.read_csv(test_data_path)
        dft['TIMESTAMP'] = dft['TIMESTAMP'].astype('datetime64')
        dft.set_index('TIMESTAMP',inplace=True)
        dft_start_datetime = min(dft.index)
        dft_end_datetime = max(dft.index)
        actual_mean = dft['RENEWABLES_RATIO'].resample('24H').mean()
        predicted_mean = forecastR.loc[(forecastR['ds'] >= dft_start_datetime) & (forecastR['ds'] <= dft_end_datetime)]
        predicted_mean.set_index('ds', inplace=True)
        actual_mean = actual_mean[min(predicted_mean.index):]
        mse = metrics.mean_squared_error(actual_mean, predicted_mean.yhat)
        rmse = sqrt(mse)
        r2score = metrics.r2_score(actual_mean, predicted_mean.yhat)
        print(str.format("Prophet RMSE: {:.2f}, R2: {:.2f}", rmse, r2score))
    return rmse, r2score

def predictProphet(data_path,periods):
    print("Training prophet model with full dataset ...")
    df = pd.read_csv(data_path)
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('datetime64')
    df.set_index('TIMESTAMP',inplace=True)
    dfr = df['RENEWABLES_RATIO'].resample('24H').mean()
    dd = pd.DataFrame(dfr)
    dd.reset_index(inplace=True)
    dd.columns = ['ds','y']
    m = Prophet(daily_seasonality=False)
    m.fit(dd)
    future=m.make_future_dataframe(periods=periods)
    print(str.format("\nPredicting with prophet model for {0} days ({1} years) ...",periods, int(periods/365)))
    forecast=m.predict(future)
    fig = m.plot(forecast,ylabel='Renewable Power Production Ratio', xlabel='Date')
    plt.title('CA Forecasted Renewable vs Non-renewable Power Production Ratio')
    axes = plt.gca()
    print('CA Future Renewable vs Non-renewable Power Production Ratio')
    wd = os.path.dirname(data_path)
    fig.savefig(wd + '/renewable-ratio-forecast.png')
    forecast.rename(columns={'ds':'TIMESTAMP'}, inplace=True)
    forecast.set_index('TIMESTAMP',inplace=True)
    prediction = pd.DataFrame({'RENEWABLES_RATIO_MEAN':forecast['yhat'].resample('1Y').mean(),'RENEWABLES_RATIO_LOWER':forecast['yhat_lower'].resample('1Y').mean(),'RENEWABLES_RATIO_UPPER':forecast['yhat_upper'].resample('1Y').mean()})
    return prediction

def rmse(actual,predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score

def transformDataset(df):
    # Add ratio from one and two days ago as well as difference in yesterday-1 and yesterday-1
    renewables_ratio = df['RENEWABLES_RATIO']
    df['YESTERDAY'] = df['RENEWABLES_RATIO'].shift()
    df['YESTERDAY_DIFF'] = df['YESTERDAY'].diff()
    df['YESTERDAY-1']=df['YESTERDAY'].shift()
    df['YESTERDAY-1_DIFF'] = df['YESTERDAY-1'].diff()
    df=df.dropna()
    x_train=pd.DataFrame({'YESTERDAY':df['YESTERDAY'],'YESTERDAY_DIFF':df['YESTERDAY_DIFF'],'YESTERDAY-1':df['YESTERDAY-1'],'YESTERDAY-1_DIFF':df['YESTERDAY-1_DIFF']})
    y_train = df['RENEWABLES_RATIO']
    return x_train,y_train

def buildRandomForestRegression(train_data_path,test_data_path):    
    print("Building Random Forest Regression Model ...")
    
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
    rmse_score = metrics.make_scorer(rmse, greater_is_better = False)
    gsearch = model_selection.GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring=rmse_score)
    gsearch.fit(x_train, y_train)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    y_true = y_test.values
    print("Predicting with best regressor ...")
    y_pred = best_model.predict(x_test)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmsescore = sqrt(mse)
    r2score = metrics.r2_score(y_true, y_pred)
    print(str.format("Random Forest Regression RMSE: {:.2f}, R2: {:.2f}", rmsescore, r2score))
    return rmsescore,r2score

def predictRandomForestRegression(data_path,periods):
    print("Training Random Forest Regression model with full dataset ...")
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
    rmse_score = metrics.make_scorer(rmse, greater_is_better = False)
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
    prediction = pd.DataFrame(columns=['TIMESTAMP','RENEWABLES_RATIO'])
    l = len(x_train)
    x_pred = x_train.iloc[[l-1]]
    y_pred = best_model.predict(x_pred)
    xmin_pred = xmin_train.iloc[[l-1]]
    ymin_pred = best_model_min.predict(xmin_pred)
    xmax_pred = xmax_train.iloc[[l-1]]
    ymax_pred = best_model_max.predict(xmax_pred)
    prediction = prediction.append({'TIMESTAMP':x_pred.index[0],'RENEWABLES_RATIO':y_pred[0],'RENEWABLES_RATIO_MIN':ymin_pred[0],'RENEWABLES_RATIO_MAX':ymax_pred[0]}, ignore_index=True)
    for i in range(1,periods):
        #ti = prediction.iloc[i-1]['TIMESTAMP'] + datetime.timedelta(days=1)
        ti = prediction.iloc[i-1]['TIMESTAMP'] + pd.offsets.DateOffset(months=1)
        xi_pred = pd.DataFrame({'YESTERDAY':y_pred,'YESTERDAY_DIFF':y_pred-x_pred['YESTERDAY'],'YESTERDAY-1':x_pred['YESTERDAY'],'YESTERDAY-1_DIFF':x_pred['YESTERDAY_DIFF']})
        yi_pred = best_model.predict(xi_pred)
        xmini_pred = pd.DataFrame({'YESTERDAY':ymin_pred,'YESTERDAY_DIFF':ymin_pred-xmin_pred['YESTERDAY'],'YESTERDAY-1':xmin_pred['YESTERDAY'],'YESTERDAY-1_DIFF':xmin_pred['YESTERDAY_DIFF']})
        ymini_pred = best_model.predict(xmini_pred)
        xmaxi_pred = pd.DataFrame({'YESTERDAY':ymax_pred,'YESTERDAY_DIFF':ymax_pred-xmax_pred['YESTERDAY'],'YESTERDAY-1':xmax_pred['YESTERDAY'],'YESTERDAY-1_DIFF':xmax_pred['YESTERDAY_DIFF']})
        ymaxi_pred = best_model.predict(xmaxi_pred)
        prediction = prediction.append({'TIMESTAMP':ti,'RENEWABLES_RATIO_MEAN':yi_pred[0],'RENEWABLES_RATIO_LOWER':ymini_pred[0],'RENEWABLES_RATIO_UPPER':ymaxi_pred[0]}, ignore_index=True)
        x_pred = xi_pred
        y_pred = yi_pred
        xmin_pred = xmini_pred
        ymin_pred = ymini_pred
        xmax_pred = xmaxi_pred
        ymax_pred = ymaxi_pred

    prediction.set_index('TIMESTAMP',inplace=True)
    prediction = prediction.resample('1Y').mean()

    p = prediction.plot()
    p.set_title('CA Predicted Renewables Ratio by Random Forest Regression')
    p.set_ylabel('RATIO')
    wd = os.path.dirname(data_path)
    plt.savefig(wd + '/renewables-ratio-forecast-rf.png')

    return prediction

def predictWithBestModel(prophet_rmse, randomforest_rmse, preprocessed_data_path):
    print("Comparing models ...")
    if (prophet_rmse <= randomforest_rmse):
        print("The best model is Prophet")
        prediction = predictProphet(preprocessed_data_path,365*30)
    else:
        print("The best model is RandomForestRegression")
        prediction = predictRandomForestRegression(preprocessed_data_path,12*30)
        
#    visualizePrediction(prediction)
    return prediction

def visualizePrediction(wd, prediction):
    
    # Log output
    print("Visualizing prediction ...")
    print("Prediction:")
    print(prediction)
    
    prediction.reset_index(inplace=True)
    print("\nPrediction for CA Renewables Ratio in Key Years:")
    prediction2030 = prediction[prediction['TIMESTAMP'] == pd.to_datetime('12-31-2030')]
    prediction2045 = prediction[prediction['TIMESTAMP'] == pd.to_datetime('12-31-2045')]
    print(str.format("   Prophet prediction 12/31/2030:"))
    print(str.format("      low: {:.2f}, mean: {:.2f}, high: {:.2f}",prediction2030['RENEWABLES_RATIO_LOWER'].values[0],prediction2030['RENEWABLES_RATIO_MEAN'].values[0],prediction2030['RENEWABLES_RATIO_UPPER'].values[0]))
    print(str.format("   Prophet prediction 12/31/2045:"))
    print(str.format("      low: {:.2f}, mean: {:.2f}, high: {:.2f}",prediction2045['RENEWABLES_RATIO_LOWER'].values[0],prediction2045['RENEWABLES_RATIO_MEAN'].values[0],prediction2045['RENEWABLES_RATIO_UPPER'].values[0]))
    
    # Table output
    prediction.to_csv(wd+'/prediction.csv', index=False)
    s = io.StringIO()
    prediction.to_csv(s, index=False)
    
    # Plot output
    prediction.set_index('TIMESTAMP', inplace=True)
    plot = prediction.plot()
    plot.set_title('Predicted CA Renewables vs Non-renewables Ratio')
    plot.set_xlabel('Date')
    plot.set_ylabel('Ratio')
    plt.savefig(wd + '/prediction.png')
    with open(wd + '/prediction.png', 'rb') as image_file:
        image = base64.b64encode(image_file.read())
    
    # Metadata
    metadata = {
        'outputs': [{
            'type': 'table',
            'storage': 'inline',
            'format': 'csv',
            'header': ['TIMESTAMP','RENEWABLES_RATIO_MEAN','RENEWABLES_RATIO_LOWER','RENEWABLES_RATIO_UPPER'],
            'source': s.getvalue()
        },
        {
            'type': 'web-app',
            'storage': 'inline',
            'source': '<html><head><title>Plot</title></head><body><div><img src="data:image/png;base64, ' + image.decode('ascii') +'" /></div></body></html>'
        }]
    }
    metadata_path = wd + '/mlpipeline-ui-metadata.json'
    with open( metadata_path, 'w') as f:
        json.dump(metadata,f)
