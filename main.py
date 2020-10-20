from kale.sdk import pipeline
from data import wrangle, preprocess, split
from model import buildProphet, buildRandomForestRegression, predictWithBestModel, visualizePrediction 
import sys
import pandas as pd

@pipeline(name='renewable-energy-pipeline',
          experiment='renewable-energy',
          autosnapshot=True)    
def renewable_energy_pipeline(wd = '.', wrangle_cache = True, preprocess_cache = True, train_pct = 70):
    data_path = wrangle(wd, wrangle_cache)
    preprocessed_data_path = preprocess(data_path, preprocess_cache)
    train_data_path, test_data_path = split(preprocessed_data_path, train_pct)
    prophet_rmse, prophet_r2score = buildProphet(train_data_path, test_data_path)
    randomforest_rmse, randomforest_r2score = buildRandomForestRegression(train_data_path, test_data_path)
    prediction = predictWithBestModel(prophet_rmse, randomforest_rmse, preprocessed_data_path)
    visualizePrediction(wd, prediction)
    
    
def main(argv):
    print("Running renewable energy project ... ")
    
    data_path = './data.csv'
    if (len(argv)>0 and ('--wrangle' in argv or '-w' in argv)):
        data_path = data.wrangle()
    
    preprocessed_data_path = './preprocessed_data.csv'
    if (len(argv)>0 and ('--preprocess' in argv or '-p' in argv)):
        preprocessed_data_path = data.preprocess(data_path)

    train_data_path, test_data_path = data.split(preprocessed_data_path, 70, './train_data.csv', './test_data.csv')
    
    prophet_rmse, prophet_r2score = model.buildProphet(train_data_path, test_data_path)
    rf_rmse, rf_r2score = model.buildRandomForestRegression(train_data_path, test_data_path)
    
    if (prophet_rmse < rf_rmse):
        prediction = model.predictProphet(preprocessed_data_path,365*30)
    else:
        prediction = model.predictRandomForestRegression(preprocessed_data_path,12*30)
    
    print("Prediction:")
    print(prediction)
    
    prediction.reset_index(inplace=True)
    print("\nPrediction for CA Renewables Ratio in Key Years:")
    print(prediction[prediction['TIMESTAMP'] == pd.to_datetime('12-31-2030')])
    print(prediction[prediction['TIMESTAMP'] == pd.to_datetime('12-31-2045')])



if __name__ == "__main__":
    #main(sys.argv[1:])
    renewable_energy_pipeline( wd = '.', wrangle_cache = True, preprocess_cache = True, train_pct = 70)
