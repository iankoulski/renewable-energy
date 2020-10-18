# Module data

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import timedelta,date,datetime
from kale.sdk import step

@step(name='wrangle')
def wrangle(data_path):
    print("Wrangling data ...")
    row_count,col_count = scrape_url('http://content.caiso.com',data_path)
    print(str.format("Saved {0} rows x {1} columns in file {2}",row_count,col_count,data_path))
    return data_path
    
def scrape_url(base_url,file_path):
    page = requests.get(str.format('{0}/green/renewrpt/files.html',base_url))
    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find_all('a')
    dataframe = pd.DataFrame(columns=['DATE','HOUR','RENEWABLES','NUCLEAR','THERMAL','IMPORTS','HYDRO'])
    for link in links:
        href = link.attrs['href']
        if str.endswith(href,'.txt'):
            file_url = base_url + href
            frame = wrangle_data(file_url)
            dataframe = dataframe.append(frame, ignore_index = True)
    dataframe.to_csv(file_path, index = False)
    return dataframe.shape[0], dataframe.shape[1]

def wrangle_data(file_url):
    print(str.format("Wrangling {0} ...",file_url))
    f = requests.get(file_url)
    txt = f.text
    date = txt.splitlines()[0].split()[0]
    tbl_txt = txt[str.find(txt,'Hourly Breakdown of Total Production by Resource Type'):]
    tbl_lines = tbl_txt.splitlines()
    frame = pd.DataFrame(columns=['DATE','HOUR','RENEWABLES','NUCLEAR','THERMAL','IMPORTS','HYDRO'])
    for i in range(2,26):
        tbl_line=tbl_lines[i]
        row = tbl_line.split()
        frame = frame.append({'DATE':date,'HOUR':row[0],'RENEWABLES':row[1],'NUCLEAR':row[2],'THERMAL':row[3],'IMPORTS':row[4],'HYDRO':row[5]}, ignore_index = True)
    return frame

@step(name='preprocess')
def preprocess(data_path):
    print("Preprocessiong data ...")
    dataframe = pd.read_csv(data_path)
    dataframe = df_cleanse(dataframe)
    dataframe = df_format(dataframe)
    dataframe = df_fill(dataframe)
    dataframe = df_engineer(dataframe)
    preprocessed_data_path = '/tmp/preprocessed_data.csv'
    dataframe.to_csv(preprocessed_data_path, index = False)
    print(str.format("Saved preprocessed data: {0} rows x {1} columns in file {2}",dataframe.shape[0],dataframe.shape[1],preprocessed_data_path))
    return preprocessed_data_path

def df_cleanse(df):
    print("Cleansing ...")
    # replace HOUR=2R with HOUR=24
    ids = df.loc[df['HOUR'] == '2R'].index
    for id in ids:
        df.loc[id].HOUR = '24'

    # replace #REF! with 0 in RENEWABLES and NUCLEAR
    ids = df.loc[df['RENEWABLES'] == '#REF!'].index
    for id in ids:
        df.loc[id].RENEWABLES = '0'
        df.loc[id].NUCLEAR = '0'
    
    # replace #NAME? with 0 in entire row
    ids = df.loc[df['RENEWABLES'] == '#NAME?'].index
    for id in ids:
        df.loc[id].RENEWABLES = '0'
        df.loc[id].NUCLEAR = '0'
        df.loc[id].THERMAL = '0'
        df.loc[id].IMPORTS = '0'
        df.loc[id].HYDRO = '0'

    # replace #VALUE! with 0 in RENEWABLES, THERMAL, IMPORTS and HYDRO
    ids = df.loc[df['RENEWABLES'] == '#VALUE!'].index
    for id in ids:
        df.loc[id].RENEWABLES = '0'
    ids = df.loc[df['THERMAL'] == '#VALUE!'].index
    for id in ids:
        df.loc[id].THERMAL = '0'
    ids = df.loc[df['IMPORTS'] == '#VALUE!'].index
    for id in ids:
        df.loc[id].IMPORTS = '0'
    ids = df.loc[df['HYDRO'] == '#VALUE!'].index
    for id in ids:
        df.loc[id].HYDRO = '0'

    return df

def df_format(df):
    print("Formatting ...")
    df['IMPUTED'] = '0'
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['HOUR'] = pd.to_numeric(df['HOUR'], downcast='integer')
    df['RENEWABLES'] = pd.to_numeric(df['RENEWABLES'])
    df['NUCLEAR'] = pd.to_numeric(df['NUCLEAR'])
    df['THERMAL'] = pd.to_numeric(df['THERMAL'])
    df['IMPORTS'] = pd.to_numeric(df['IMPORTS'])
    df['HYDRO'] = pd.to_numeric(df['HYDRO'])
    df['IMPUTED'] = pd.to_numeric(df['IMPUTED'])
    return df

def df_fill(df):
    df = df_fill_dates(df)
    df = df_fill_values(df)
    return df

def df_fill_dates(df):
    print("Filling missing dates ...")
    start_date = min(df.DATE).date()
    end_date = max(df.DATE).date()
    for single_date in daterange(start_date, end_date):
        print(single_date.strftime("%m/%d/%y"))
        dt = datetime(single_date.year,single_date.month,single_date.day)
        date_data = df.loc[df['DATE'] == dt]
        hrs = len(date_data)
        if (hrs < 24) :
            #fill missing hours
            for hr in range(1,25):
                hr_data = df.loc[(df['DATE'] == dt) & (df['HOUR'] == hr)]
                if len(hr_data) < 1 :
                    print(str.format("Adding DATE: {0}, HOUR: {1}", dt.strftime("%m/%d/%y"), hr))
                    new_row = {'DATE': dt, 'HOUR': hr, 'RENEWABLES': 0, 'NUCLEAR': 0, 'THERMAL': 0, 'IMPORTS': 0, 'HYDRO': 0, 'IMPUTED': 0}
                    df = df.append(new_row, ignore_index=True)
    df.sort_values(by=['DATE','HOUR'], inplace=True, ascending=True)
    df = df.reset_index()
    df = df.drop(columns=['index'])
    return df

def df_fill_values(df):
    print("Filling missing values ...")
    for col in ['RENEWABLES', 'NUCLEAR', 'THERMAL', 'IMPORTS', 'HYDRO']: 
        df = df_fill_values_col(df, col)
    return df

def df_fill_values_col(df, col):
    print(str.format("Column {0} ...",col))
    cdata = df[col]
    l = len(cdata)
    last_value=0
    i = 0
    while i < l:
        if cdata[i] == 0:            
            # find next nonzero value
            j = i
            while (j < l) and (cdata[j] == 0):
                j = j + 1
            if j == l:
                next_value = 0
            else:
                next_value = cdata[j]
            # fill values with average
            j = i
            avg = (last_value + next_value)/2
            while (j<l) and (cdata[j] == 0):
                print(str.format("   Imputing column {0} row {1} as {2} ...", col, j, avg))
                df.at[j,col] = avg
                df.at[j,'IMPUTED'] = 1
                j = j + 1
            i = j - 1
        else:
            # remember last nonzero value
            last_value = cdata[i]
        i = i + 1
    return df

def df_engineer(df):
    print("Feature engineering ...")
    
    print("  Adding timestamps ...")
    df['TIMESTAMP'] = df['DATE']
    l = len(df)
    for r in range(0,l):
        df.at[r,'TIMESTAMP'] = df['DATE'][r] + timedelta(hours=int(df['HOUR'][r])-1)
    
    print("   Adding NONRENEWABLES ...")
    df['NONRENEWABLES'] = df['NUCLEAR'] + df['THERMAL'] + df['HYDRO'] + df['IMPORTS']

    print("   Adding RATIO ...")
    df['RENEWABLES_RATIO'] = df['RENEWABLES'] / df['NONRENEWABLES']

    return df

@step(name='split')
def split(data_path, train_pct, train_data_path, test_data_path):
    print("Splitting data ...")
    print(str.format("  Training: {0}%, Testing {1}%", train_pct, 100 - train_pct))
    dataframe = pd.read_csv(data_path)
    split_idx = int(len(dataframe) * (train_pct/100))
    split_idx = split_idx - split_idx % 24
    df_train = dataframe[:split_idx]
    df_test = dataframe[split_idx+1:]
    df_train.to_csv(train_data_path, index = False)
    df_test.to_csv(test_data_path, index = False)
    return train_data_path, test_data_path    

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)