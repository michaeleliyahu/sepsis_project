import pandas as pd

read_file = pd.read_excel('Demo.xlsx')

def calculateCVRI(MAP, HR, RR, BSA):
    if BSA == 0 or RR == 0:
        return 0
    result_CVRI = 18 * MAP / (HR * RR * BSA)
    return result_CVRI


for i in range(len(read_file['h_num_demo'])):
    read_file['CVRI'][i] = calculateCVRI(read_file['MAP'][i],read_file['HR'][i],read_file['RR'][i],read_file['BSA'][i])

read_file['CVRI'] = read_file['CVRI'].fillna(method='ffill')  # fill RR.

read_file.to_excel('Demo.xlsx')