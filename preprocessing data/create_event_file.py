import xlsxwriter
workbook = xlsxwriter.Workbook('CVRI_data.xlsx')
worksheet = workbook.add_worksheet()

counter_zero = 0
total_people = 0
total_event = []
event_with_zero_val = []
not_event_with_zero_val = []
total_with_hour = []
total_delete_row = []

import pandas as pd
excel_file = 'file.xlsx'
vital_file = pd.read_excel(excel_file)


def check_befure(start, end, event,row_cvri):
    keep_event = event - 1
    null_val = True
    while keep_event > keep_event-10 and keep_event > start:
        if vital_file['CVRI'][keep_event] != 0:
            null_val = False
        keep_event -= 1
    if null_val:
        event_with_zero_val.append(vital_file['h_num_demo'][keep_event])
        return 0
    if event - start >= 20:
        hour_befure = event-20
        colomn = 1
        while hour_befure != event:
            # cvri = vital_file['CVRI'][hour_befure]
            # print(vital_file['CVRI'][hour_befure], hour_befure,event)
            worksheet.write(row_cvri, colomn, vital_file['CVRI'][hour_befure])
            hour_befure += 1
            colomn += 1
        worksheet.write(row_cvri, 0, vital_file['h_num_demo'][hour_befure])
        worksheet.write(row_cvri, colomn, vital_file['EVENT'][hour_befure])
        total_with_hour.append(vital_file['h_num_demo'][hour_befure])
        return 1
    else:
        i = event + 1
        count_row = 1
        while i < end:
            if count_row == 20:
                return 0
            elif vital_file['MAP'][i] < 60:
                count_row = 0
                i += 1
            else:
                i += 1
                count_row += 1
    return 0


def add_line_not_enough_row(start, row_cvri):
    colomn = 1
    i = 0
    while i < 20:
        worksheet.write(row_cvri, colomn, vital_file['CVRI'][start])
        start += 1
        colomn += 1
        i += 1
    worksheet.write(row_cvri, 0, vital_file['h_num_demo'][start])
    worksheet.write(row_cvri, colomn, 0)


def add_line_for_not_event(start, row_cvri):
    i = 0
    null_val = True
    while i < 20 :
        if vital_file['CVRI'][i + start] != 0:
            null_val = False
        i += 1
    if null_val:
        not_event_with_zero_val.append(vital_file['h_num_demo'][i + start])
        return 0
    colomn = 1
    i = 0
    while i < 20:
        worksheet.write(row_cvri, colomn, vital_file['CVRI'][start])
        colomn += 1
        start += 1
        i += 1
    worksheet.write(row_cvri, 0, vital_file['h_num_demo'][start])
    worksheet.write(row_cvri, colomn, 0)
    return 1


def add_event(start, end,row_cvri):
    event = False
    i = start
    while i < end:
        if vital_file['MAP'][i] < 60:
            event = True
            row_delete = check_befure(start,end,i,row_cvri)
            total_event.append(vital_file['h_num_demo'][i])
            return row_delete
        i = i +1
    if not event:
        return add_line_for_not_event(start, row_cvri)


number = -20
worksheet.write(0, 0, 'id')
for i in range(1,21):
    worksheet.write(0,i,number)
    number += 1
worksheet.write(0, 21, 'event')


row_start = 1
row_end = 0
row_cvri = 1
first_id = vital_file['h_num_demo'][1]
counter = 0
for i in range(len(vital_file['h_num_demo'])):
    if vital_file['h_num_demo'][i] != first_id:
        counter += 1
        row_end = i-1
        # if counter == 49:
        #     print(first_id,row_start,row_end,row_cvri)
        #     break
        row_delete = add_event(row_start,row_end,row_cvri)
        if row_delete == 0:
            counter_zero += 1
        if row_delete:
            total_delete_row.append(row_delete)
        row_start = i
        first_id = vital_file['h_num_demo'][i]
        row_cvri += row_delete
        total_people += 1


workbook.close()
print("total people: ", total_people,
      ", total event: ", len(total_event),
      ", total event with 20 hours: ", len(total_with_hour),
      ',delete_row: ', len(total_delete_row),
      ',not_event_with_zero_val:', len(not_event_with_zero_val),
      ',event_with_zero_val:', len(event_with_zero_val))

print(counter_zero)
#workbook.close()

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# labels = ['6', '7', '8', '9']
# men_means = [35, 33, 29, 26]
# women_means = [71, 73, 77, 80]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='exist')
# rects2 = ax.bar(x + width/2, women_means, width, label='not_exist')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Number of people')
# ax.set_xticks(x, labels)
# ax.legend()
#
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
#
# fig.tight_layout()
#
# plt.show()