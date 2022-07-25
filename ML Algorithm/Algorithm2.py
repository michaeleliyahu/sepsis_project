import dataset as dataset
import np as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import openpyxl

read_file = pd.read_excel(r'total_cvri.xlsx')
read_file.to_csv(r'CVRI_model.csv', index=None, header=True)

pd.read_csv('CVRI_model.csv', delimiter=',').fillna(0)

dataset = pd.read_csv('CVRI_model.csv')
dataset.head()

x = dataset[['-20', '-19', '-18', '-17', '-16', '-15', '-14', '-13', '-12', '-11', '-10', '-9', '-8']]
# x = dataset[['-20', '-19', '-18', '-17', '-16', '-15', '-14', '-13', '-12', '-11', '-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2']]
y = dataset['event']

# Spliting data into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=65)

algorithms_name = []
algorithms_result = []

auc_results = []
recall_results = []

# LogisticRegression
algorithms_name.append('Logistic Regression')
model_Log = LogisticRegression(random_state=1)
model_Log.fit(X_train, Y_train)
Y_pred_logistic = model_Log.predict(X_test)
model_Log_accuracy = round(accuracy_score(Y_test, Y_pred_logistic), 4) * 100  # Accuracy
algorithms_result.append(model_Log_accuracy)
print("Logistic Regression Accuracy:", model_Log_accuracy, "%")

auc_score = roc_auc_score(Y_test, Y_pred_logistic)
print('Logistic Regression Auc roc: %.3f' % auc_score, '%')
auc_results.append(auc_score)
print('Recall: %.3f' % metrics.recall_score(Y_test, Y_pred_logistic), "%")
recall_log = metrics.recall_score(Y_test, Y_pred_logistic)
recall_results.append(recall_log)
print("Classification repot:")
print(metrics.classification_report(Y_test, Y_pred_logistic))



# model_log_auc = round(roc_auc_score(Y_test,Y_pred), 4)*100
# print(model_log_auc)
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred_logistic)
# confusion matrix to see the miscalifications.
print("Confusion matrix: ")
print(cnf_matrix)

importances = model_Log.coef_[0]
print(importances)

print("----------------------------------------------------------------------")

# Desicion tree
algorithms_name.append('Desicion tree')
model_tree = DecisionTreeClassifier(random_state=10, criterion="entropy", max_depth=100)
model_tree.fit(X_train, Y_train)
Y_pred_decision = model_tree.predict(X_test)
model_tree_accuracy = round(accuracy_score(Y_test, Y_pred_decision), 4) * 100  # Accuracy
algorithms_result.append(model_tree_accuracy)
print('Desicion tree Accuracy: %.2f' % model_tree_accuracy, "%")
auc_score_tree = roc_auc_score(Y_test, Y_pred_decision)
print('Decision Tree Auc roc: %.3f' % auc_score_tree, '%')
auc_results.append(auc_score_tree)
print("Classfification Report: ")
print(metrics.classification_report(Y_test, Y_pred_decision))

recall_decision = metrics.recall_score(Y_test, Y_pred_decision)
recall_results.append(recall_decision)
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred_decision)
# confusion matrix to see the miscalifications.
print("Confusion matrix: ")
print(cnf_matrix)
print(model_tree.feature_importances_)

print("----------------------------------------------------------------------")

# Random forest
algorithms_name.append('Random forest')
model_random = RandomForestClassifier(n_estimators=1000, criterion="gini", random_state=10)
model_random.fit(X_train, Y_train)
y_pred_Random = model_random.predict(X_test)
model_random_accuracy = round(accuracy_score(Y_test, y_pred_Random), 4) * 100  # Accuracy
algorithms_result.append(model_random_accuracy)
print('Random forest Accuracy:  %.2f' % model_random_accuracy, "%")
auc_score_random = roc_auc_score(Y_test, y_pred_Random)
print('Random Forest Auc roc: %.3f' % auc_score_random, '%')
auc_results.append(auc_score_random)
print("Classfification Report: ")
print(metrics.classification_report(Y_test, y_pred_Random))

recall_random = metrics.recall_score(Y_test, y_pred_Random)
recall_results.append(recall_random)

cnf_matrix = metrics.confusion_matrix(Y_test, y_pred_Random)
# confusion matrix to see the miscalifications.
print("Confusion matrix: ")
print(cnf_matrix)

#print(model_random.feature_importances_)
feat_importances = pd.Series(model_random.feature_importances_, index=x.columns)
print(feat_importances)

'''
feat_importances.nlargest(10).plot(kind='barh', label = "Hours")
plt.title('Feature Importance')
plt.xlabel('Score')
plt.ylabel('Hours before')
plt.legend()
plt.show()
'''
print("----------------------------------------------------------------------")
# xgboost
algorithms_name.append('xgboost')
model_boost = XGBClassifier(random_state=10)
model_boost.fit(X_train, Y_train)
y_pred_xg = model_boost.predict(X_test)
model_boost_accuracy = round(accuracy_score(Y_test, y_pred_xg), 4) * 100  # Accuracy
algorithms_result.append(model_boost_accuracy)
print('XGboost Accuracy: %.3f' % model_boost_accuracy, "%")
auc_score_XG = roc_auc_score(Y_test, y_pred_xg)
print('XGboost Auc roc: %.3f' % auc_score_XG, '%')
auc_results.append(auc_score_XG)
print("Classfification Report: ")
print(metrics.classification_report(Y_test, y_pred_xg))

recall_XG = metrics.recall_score(Y_test, y_pred_xg)
recall_results.append(recall_XG)

cnf_matrix = metrics.confusion_matrix(Y_test, y_pred_xg)
# confusion matrix to see the miscalifications.
print("Confusion matrix: ")
print(cnf_matrix)

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

X_axis = np.arange(len(algorithms_name))
plt.subplots(figsize=(8, 5))
plt.bar(X_axis - 0.2, auc_results, 0.4, label='Auc')
plt.bar(X_axis + 0.2, recall_results, 0.4, label='Recall')

plt.xticks(X_axis, algorithms_name)
plt.xlabel("Algorithms")
plt.ylabel("Score")
plt.title("Auc & Recall results")
plt.legend()
plt.show()
