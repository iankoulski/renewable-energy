import sys
import os

def renewable_energy_pipeline(wd = '.', wrangle_cache = True, preprocess_cache = True, train_pct = 70):
    data_path = wrangle_step(wd, wrangle_cache)
    preprocessed_data_path = preprocess_step(data_path, preprocess_cache)
    train_data_path, test_data_path = split_step(preprocessed_data_path, train_pct)
    linear_rmse = linear_build_step(train_data_path, test_data_path)
    prophet_rmse = prophet_build_step(train_data_path, test_data_path)
    randomforest_rmse = randomforest_build_step(train_data_path, test_data_path)
    #xgboost_rmse = xgboost_build_step(train_data_path, test_data_path)
    #armodel_rmse = armodel_build_step(train_data_path, test_data_path)
    #lstm_rmse = lstm_build_step(train_data_path, test_data_path)
    #techniques = {'Linear': linear_rmse, 'Prophet':prophet_rmse, \
    #               'RandomForest': randomforest_rmse, 'XGBoost': xgboost_rmse, \
    #               'Autoregression': armodel_rmse, 'LSTM': lstm_rmse}
    techniques = {'Linear': linear_rmse, 'Prophet': prophet_rmse, 'RandomForest': randomforest_rmse}
    prediction = predict_with_linear_and_best_model_step(techniques, preprocessed_data_path)
    visualize_prediction_step(wd, prediction)

def wrangle_step(wd, wrangle_cache):
    from data import wrangle
    data_path = wrangle(wd, wrangle_cache)
    return data_path

def preprocess_step(data_path, preprocess_cache):
    from data import preprocess
    preprocessed_data_path = preprocess(data_path, preprocess_cache)
    return preprocessed_data_path

def split_step(preprocessed_data_path, train_pct):
    from data import split
    train_data_path, test_data_path = split(preprocessed_data_path, train_pct)
    return train_data_path, test_data_path

def linear_build_step(train_data_path, test_data_path):
    from model import buildLinearModel
    linear_rmse = buildLinearModel(train_data_path, test_data_path)
    return linear_rmse

def prophet_build_step(train_data_path, test_data_path):
    from model import buildProphet
    prophet_rmse = buildProphet(train_data_path, test_data_path)
    return prophet_rmse

def randomforest_build_step(train_data_path, test_data_path):
    from model import buildRandomForestRegression
    randomforest_rmse = buildRandomForestRegression(train_data_path, test_data_path)
    return randomforest_rmse

def lstm_build_step(train_data_path, test_data_path):
    from model import buildLSTM
    lstm_rmse = buildLSTM(train_data_path, test_data_path)
    return lstm_rmse

def xgboost_build_step(train_data_path, test_data_path):
    from model import buildXGBoostRegression
    xgboost_rmse = buildXGBoostRegression(train_data_path, test_data_path)
    return xgboost_rmse

def armodel_build_step(train_data_path, test_data_path):
    from model import buildARModel
    armodel_rmse = buildARModel(train_data_path, test_data_path)
    return armodel_rmse

def predict_with_linear_and_best_model_step(techniques, preprocessed_data_path):
    from model import predictWithLinearAndBestModel
    prediction = predictWithLinearAndBestModel(techniques, preprocessed_data_path)
    return prediction

def visualize_prediction_step(wd, prediction):
    from model import visualizePrediction
    visualizePrediction(wd, prediction)
    
def main(argv):
    print("Running renewable energy project ... ")

    wrangle_cache = True
    if len(argv)>0 and ('--wrangle' in argv or '-w' in argv):
        wrangle_cache = False
        
    preprocess_cache = True
    if len(argv)>0 and ('--preprocess' in argv or '-p' in argv):
        preprocess_cache = False

    wd = '.'
    if len(argv)>1 and ('--workdir' in argv or '-d' in argv):
        index1 = argv.index('--workdir')
        index2 = argv.index('-d')
        index = max(index1,index2)
        if (len(argv)>index):
            wd = argv[index+1]        

    train_pct = 70
    if len(argv)>1 and ('--train' in argv or '-t' in argv):
        index1 = argv.index('--train')
        index2 = argv.index('-t')
        index = max(index2,index2)
        if (len(argv)>index):
            train_pct = argv[index+1]
            
    print("Running renewable energy pipeline ... ")
    renewable_energy_pipeline( wd=wd, wrangle_cache=wrangle_cache, preprocess_cache=preprocess_cache, train_pct=train_pct)


if __name__ == "__main__":
    cwd = os.getcwd()
    print(str.format('Current working directory: {}', cwd))
    main(sys.argv[1:])
