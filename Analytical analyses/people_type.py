import matplotlib.pyplot as plt
import numpy as np
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Shock septic in first 20 hours\n(deleted)', 'shock septic after 20 hours','people without shock septic'
sizes = [1672, 822, 619]
explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

def func(pct, allvalues):
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\n{:d}".format(pct, absolute)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels,  autopct=lambda pct: func(pct, sizes),
        shadow=True, startangle=90)
ax1.set(title ="Total people 2011-2017")
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# # Import libraries
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Creating dataset
# cars = 'Shock septic in first 20 hours', 'shock septic after 20 hours','people without shock septic'
#
# data = [1672, 822, 619]
#
# # Creating explode data
# explode = (0, 0,0)
#
# # Creating color parameters
# colors = ("orange", "cyan","green")
#
# # Wedge properties
# wp = {'linewidth': 1, 'edgecolor': "green"}
#
#
# # Creating autocpt arguments
# def func(pct, allvalues):
#         absolute = int(pct / 100. * np.sum(allvalues))
#         return "{:.1f}%\n{:d}".format(pct, absolute)
#
#
# # Creating plot
# fig, ax = plt.subplots(figsize=(10, 7))
# wedges, texts, autotexts = ax.pie(data,
#                                   autopct=lambda pct: func(pct, data),
#                                   explode=explode,
#                                   labels=cars,
#                                   shadow=True,
#                                   colors=colors,
#                                   startangle=90,
#                                   wedgeprops=wp,
#                                   textprops=dict(color="magenta"))
#
# # # Adding legend
# # ax.legend(wedges, cars,
# #           title="Cars",
# #           loc="center left",
# #           bbox_to_anchor=(1, 0, 0.5, 1))
#
# plt.setp(autotexts, size=8, weight="bold")
# ax.set_title("Customizing pie chart")

# # show plot
# plt.show()