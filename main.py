import data
import model
import sys
import pandas as pd

def main(argv):
    print("Running renewable energy project ... ")
    
    data_path = '/tmp/data.csv'
    if (len(argv)>0 and ('--wrangle' in argv or '-w' in argv)):
        data_path = data.wrangle()
    
    preprocessed_data_path = '/tmp/preprocessed_data.csv'
    if (len(argv)>0 and ('--preprocess' in argv or '-p' in argv)):
        preprocessed_data_path = data.preprocess(data_path)

    train_data_path = '/tmp/train_data.csv'
    test_data_path = '/tmp/test_data.csv'    
    len_train, len_test = data.split(preprocessed_data_path, 70, train_data_path, test_data_path)
    
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
    main(sys.argv[1:])
