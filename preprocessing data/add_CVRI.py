import pandas as pd

read_file = pd.read_excel('file.xlsx')

def calculateCVRI(MAP, HR, RR, BSA):
    if BSA == 0 or RR == 0:
        return 0
    result_CVRI = 18 * MAP / (HR * RR * BSA)
    return result_CVRI


read_file['MAP'] = read_file['MAP'].fillna(method='ffill')  # fill MAP
read_file['HR'] = read_file['HR'].fillna(method='ffill')  # fill HR.
read_file.to_excel('file.xlsx')

for i in range(len(read_file['h_num_demo'])):
    read_file['CVRI'][i] = calculateCVRI(read_file['MAP'][i],read_file['HR'][i],read_file['RR'][i],read_file['BSA'][i])

read_file['CVRI'] = read_file['CVRI'].fillna(method='ffill')  # fill RR.

read_file.to_excel('file.xlsx')