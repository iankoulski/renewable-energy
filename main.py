import data
import model
import sys

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
    
    #prophet_rmse, prophet_r2score = model.buildProphet(train_data_path, test_data_path)

    

    prediction = model.predictProphet(preprocessed_data_path,365*30)

    print(prediction)


if __name__ == "__main__":
    main(sys.argv[1:])
