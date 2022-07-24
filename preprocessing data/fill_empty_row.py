import datetime
import pandas as pd

# excel_file = 'weight height.xlsx'
# WeightHeight_file = pd.read_excel(excel_file)
file = pd.read_excel('file.xlsx')


file['MAP'] = file['MAP'].fillna(method='ffill')  # fill MAP
file['HR'] = file['HR'].fillna(method='ffill')  # fill HR.
file['RR'] = file['RR'].fillna(method='ffill')  # fill RR.
file['BSA'] = file['BSA'].fillna(method='ffill')  # fill RR.


# for i in range(len(file['h_num_demo'])):
#     if pd.isna(file['date'][i]):
#         file['date'][i] = str((file['date'][i] + datetime.timedelta(hours=1)).replace(minute=0))

file.to_excel('file.xlsx')
