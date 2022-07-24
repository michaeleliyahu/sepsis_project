import pandas as pd
import xlsxwriter
import datetime

pd.options.mode.chained_assignment = None  # default='warn'
workbook = xlsxwriter.Workbook('file.xlsx')
worksheet = workbook.add_worksheet()

drugs = [4704, 5433]
row_excel = 0

worksheet.write('A1', 'h_num_demo')
worksheet.write('B1', 'date')
worksheet.write('C1', 'counter')
worksheet.write('D1', 'HR')
worksheet.write('E1', 'MAP')
worksheet.write('F1', 'BSA')
worksheet.write('G1', 'RR')
worksheet.write('H1', 'CVRI')
worksheet.write('I1', 'EVENT')


excel_file = 'src/4704&5433 - 2017.xlsx'
vital_file = pd.read_excel(excel_file)


def put1andColor(row):
    format = workbook.add_format()
    format.set_pattern(1)
    format.set_bg_color('red')
    worksheet.write(row, 8, 1, format)


def validation(vital_number, value, row):
    if vital_number == drugs[0]:  # HR check 40-180
        if 40 <= int(value) <= 180:
            return True
    if vital_number == drugs[1]:  # MAP check 50-180
        if 50 <= int(value) <= 180:
            if value < 60:
                put1andColor(row)
            return True
    return False


def add_MAP(start_row, end_row, row, colomn, drug):
    while start_row < end_row:
        time_next = vital_file['Time'][start_row+1].replace(minute=0)
        time = vital_file['Time'][start_row].replace(minute=0)
        time_plus = time + datetime.timedelta(hours=1)
        value = vital_file['Value'][start_row]
        if str(time) == str(time_next):
            row -= 1
        elif str(time_plus) < str(time_next):
            if validation(drug, value, row):
                worksheet.write(row, colomn, value)
            while str(time_plus) < str(time_next):
                row += 1
                time_plus = time_plus + datetime.timedelta(hours=1)
        else:
            if validation(drug, value, row):
                worksheet.write(row, colomn, value)
        row += 1
        start_row += 1
    return row


def add_id_counter_date_HR(start_row, end_row, row, colomn, drug):
    counter = 0
    while start_row < end_row:
        time_next = vital_file['Time'][start_row+1].replace(minute=0)
        time = vital_file['Time'][start_row].replace(minute=0)
        time_plus = time + datetime.timedelta(hours=1)
        value = vital_file['Value'][start_row]
        if str(time) == str(time_next):
            row -= 1
            counter -= 1

        elif str(time_plus) < str(time_next):
            worksheet.write(row, 0, id)
            worksheet.write(row, 1, str(time))
            worksheet.write(row, 2, counter)
            if validation(drug, value, row):
                worksheet.write(row, colomn, value)
            while str(time_plus) < str(time_next):
                row += 1
                counter += 1
                worksheet.write(row, 0, id)
                worksheet.write(row, 2, counter)
                time_plus = time_plus + datetime.timedelta(hours=1)
        else:
            worksheet.write(row, 0, id)
            worksheet.write(row, 1, str(time))
            worksheet.write(row, 2, counter)
            if validation(drug, value, row):
                worksheet.write(row, colomn, value)
        row += 1
        counter += 1
        start_row += 1
    return row


first_id = vital_file['h_num_hash'][0]
first_drug = vital_file['ParameterID'][0]

vital_row = 0
excel_row = 1
keep = True
length = len(vital_file['h_num_hash'])
while vital_row < length:
    id = first_id = vital_file['h_num_hash'][vital_row]
    drug = vital_file['ParameterID'][vital_row]
    start_drug1 = vital_row

    while id == vital_file['h_num_hash'][vital_row] and drug == vital_file['ParameterID'][vital_row]:
        vital_row += 1

    end_drug1 = vital_row
    drug = vital_file['ParameterID'][vital_row]
    start_drug2 = end_drug1

    while vital_row < length and id == vital_file['h_num_hash'][vital_row] and drug == vital_file['ParameterID'][vital_row]:
        vital_row += 1

    end_drug2 = vital_row-1

    keep_excel_row = excel_row
    excel_row = add_id_counter_date_HR(start_drug1, end_drug1, excel_row, 3, drugs[0])
    x = add_MAP(start_drug2, end_drug2, keep_excel_row, 4, drugs[1])

workbook.close()
