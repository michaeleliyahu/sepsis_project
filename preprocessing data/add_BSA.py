import pandas as pd

excel_file = 'src/weight height.xlsx'
WeightHeight_file = pd.read_excel(excel_file)
read_file = pd.read_excel('file.xlsx')



def calculate_BSA(height,weight):
    BSA = 0
    height = height * 100  # convert to cm
    if 30 <= weight <= 180 and 130 <= height <= 210:  # validation.
        BSA = ((height * weight) / 3600) * 0.5  # BSA formula.
        BSA = round(BSA,1)
        # print("here1")
    else:
        print("here1")
    if 2.5 >= BSA >= 1.2:
        return BSA
    else:
        print(BSA)
    return 0  # return 0 means the BSA is unvalid and we can't calculate CVRI.


dict_weight = {}
dict_height= {}
id = ""
i = 1
for i in range(len(WeightHeight_file['h_num_hash'])):
    if WeightHeight_file['h_num_hash'][i] != id and WeightHeight_file['ParameterID'][i] == 6393:
        id = WeightHeight_file['h_num_hash'][i]
        dict_weight[id] = WeightHeight_file['Value'][i]

id = ""
i = 1
for i in range(len(WeightHeight_file['h_num_hash'])):
    if WeightHeight_file['h_num_hash'][i] != id and WeightHeight_file['ParameterID'][i] == 6395:
        id = WeightHeight_file['h_num_hash'][i]
        dict_height[id] = WeightHeight_file['Value'][i]


key = ""
bsa = 0
exist = False
i = 1
counter_unvalid_bsa = 0
for i in range(len(read_file['h_num_demo'])):
    if read_file['h_num_demo'][i] != key:
        key = read_file['h_num_demo'][i]
        if key in dict_height and key in dict_weight:
            exist = True
            bsa = calculate_BSA(dict_height[key],dict_weight[key])
            if bsa == 0:
                counter_unvalid_bsa += 1
            read_file['BSA'][i] = bsa
        else:
            exist = False
            # counter_unvalid_bsa += 1
    elif exist and not pd.isna(read_file['date'][i]):
        read_file['BSA'][i] = bsa

print(counter_unvalid_bsa)
read_file.to_excel('file.xlsx')
