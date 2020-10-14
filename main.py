import data
import sys, getopt

def main(argv):
    print("Running renewable energy project ... ")
    data_path = data.wrangle()

if __name__ == "__main__":
    main(sys.argv[1:])
