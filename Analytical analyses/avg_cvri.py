import pandas as pd
import matplotlib.pyplot as plt
import np as np

read_file2 = pd.read_excel('total_cvri.xlsx')

cvri_for_graph = []
cvri_for_not_graph = []

def avg_cvri_for_event():
    cvri_for_event = {}
    cvri_for_not_event = {}
    row = -20
    avg = 0
    counter = 0
    final_avg = 0
    while row < 0:
        for i in range(len(read_file2['id'])):
            if read_file2['event'][i] == 1:
                avg += read_file2[row][i]
                counter += 1
        final_avg += avg / counter
        cvri_for_event[row] = final_avg
        cvri_for_graph.append(final_avg)
        final_avg = 0
        avg = 0
        row += 1
        counter = 0
    return cvri_for_event

def avg_cvri_for_not_event():
    cvri_for_not_event = {}
    row = -20
    avg = 0
    counter = 0
    final_avg = 0
    while row < 0:
        for i in range(len(read_file2['id'])):
            if read_file2['event'][i] == 0:
                avg += read_file2[row][i]
                counter += 1

        final_avg += avg / counter
        cvri_for_not_event[row] = final_avg
        cvri_for_not_graph.append(final_avg)
        final_avg = 0
        avg = 0
        row += 1
        counter = 0

    return cvri_for_not_event

not_event = avg_cvri_for_not_event()
event_cvri_avg = avg_cvri_for_event()

print(event_cvri_avg)

def plot_avg():
    plt.subplots(figsize=(13, 5))
    x = np.array(
        ['-20', '-19', '-18', '-17', '-16', '-15', '-14', '-13', '-12', '-11', '-10', '-9', '-8', '-7', '-6', '-5',
         '-4', '-3', '-2', '-1'])
    plt.plot(x, cvri_for_graph, label='Septic-Shock ')
    plt.plot(x, cvri_for_not_graph, label='Not Septic-Shock')
    plt.legend()
    plt.xlabel('Hours before event')
    plt.ylabel('CVRI')
    plt.title('Average CVRI')
    # ax = plt.axes(x)
    plt.show()

plot_avg()

data = []
for i in range(len(not_event)):
    data.append(np.random.normal(not_event[i], event_cvri_avg[i], 200))

print(data)
# # Import libraries
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.array(
#     ['-20', '-19', '-18', '-17', '-16', '-15', '-14', '-13', '-12', '-11', '-10', '-9', '-8', '-7', '-6', '-5',
#      '-4', '-3', '-2', '-1'])
#
# # Creating dataset
# np.random.seed(10)
# data = []
#
# for i in range(len(not_event)):
#     np.random.normal(100, 20, 200)
#     data.append(np.random.normal(not_event[i], event_cvri_avg[i], 200))
#
# print(data)
# fig = plt.figure(figsize=(10, 7))
#
# # Creating plot
# plt.boxplot(x,data)
#
# # show plot
# plt.show()