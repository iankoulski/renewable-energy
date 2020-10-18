import pandas as pd
from fbprophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from math import sqrt

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
    plt.savefig('/tmp/renewable-ratio-history.png')
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
    figR.savefig('/tmp/renewable-ratio-forecast.png')

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
    m = Prophet(daily_seasonality=True)
    m.fit(dd)
    future=m.make_future_dataframe(periods=periods)
    print(str.format("Predicting with prophet model for {0} periods ...",periods))
    forecast=m.predict(future)
    fig = m.plot(forecast,ylabel='Renewable vs Non-renewable Power Production Ratio', xlabel='Date')
    plt.title('Forecasted Renewable vs Non-renewable Power Production Ratio')
    axes = plt.gca()
    print('Future Renewable vs Non-renewable Power Production Ratio')
    fig.savefig('/tmp/renewable-ratio-forecast.png')
    forecast.set_index('ds',inplace=True)
    prediction = pd.DataFrame({'RENEWABLES_RATIO_MEAN':forecast['yhat'].resample('1Y').mean(),'RENEWABLE_RATIO_LOWER':forecast['yhat_lower'].resample('1Y').mean(),'RENEWABLE_RATIO_UPPER':forecast['yhat_upper'].resample('1Y').mean()})
    return prediction
