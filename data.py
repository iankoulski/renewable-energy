# Module data

import requests
from bs4 import BeautifulSoup
import pandas as pd

def wrangle():
    print("Wrangling data ...")
    data_path = '/tmp/data.csv'
    row_count,col_count = scrape_url('http://content.caiso.com',data_path)
    print(str.format("Saved {0} rows x {1} columns in file {3}",row_count,col_count,data_path))


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

