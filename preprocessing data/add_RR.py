import pandas as pd
import datetime

excel_file = 'src/4612 - 2017.xlsx'
RR_file = pd.read_excel(excel_file)
file = pd.read_excel('file.xlsx')

j = 0
i = 0
length = len(file['h_num_demo'])
while i < length:
    rr_time = RR_file['Time'][j].replace(minute=0)
    file_time = file['date'][i]
    if not pd.isna(file['date'][i]) and file['date'][i] == str(rr_time):
        file['RR'][i] = RR_file['Value'][j]
        j += 1
    elif pd.isna(file['date'][i]) and str(rr_time - datetime.timedelta(hours=1)) == str(RR_file['Time'][j-1].replace(minute=0)) :
        j += 1
    elif str(rr_time) == str(RR_file['Time'][j-1].replace(minute=0)):
        i -= 1
        j += 1
    elif file['h_num_demo'][i] != file['h_num_demo'][i-1] and RR_file['h_num_hash'][j] == RR_file['h_num_hash'][j-1]:
        j += 1
        i -= 1
    elif not pd.isna(file['date'][i]) and file['h_num_demo'][i] == RR_file['h_num_hash'][j] and str(rr_time) < str(file_time):
        j += 1
        i -= 1
    i += 1

file.to_excel('file.xlsx')
