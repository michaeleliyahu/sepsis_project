import matplotlib.pyplot as plt
import numpy as np

labels = ['Logistic_Regression', 'Desicion_tree', 'Random_forest', 'XGboost']


hour_12 = [0.596, 0.569, 0.615, 0.58]
hour_16 = [0.599, 0.598, 0.630, 0.624]
hour_18 = [0.624, 0.602, 0.642, 0.615]
x = np.arange(4)  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, hour_12, width, label='8 hour before')
rects2 = ax.bar(x, hour_16, width, label='4 hour before')
rects3 = ax.bar(x + width, hour_18, width, label='2 hour before')


ax.set_ylabel('Auc roc')
ax.set_title('Algorithms results for each data set')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)


fig.tight_layout()

plt.show()
