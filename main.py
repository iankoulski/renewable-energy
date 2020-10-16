import data
import sys

def main(argv):
    print("Running renewable energy project ... ")
    
    data_path='/tmp/data.csv'
    if (len(argv)>0 and ('--wrangle' in argv or '-w' in argv)):
        data_path = data.wrangle()
    
    clean_data_path = data.preprocess(data_path)

if __name__ == "__main__":
    main(sys.argv[1:])
