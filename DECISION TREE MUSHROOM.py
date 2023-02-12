import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree



my_ds=pd.read_csv("Mobile_train.csv") 
column_list = []
for i in my_ds.columns:
    column_list.append(i)

del column_list[20]

L = my_ds[column_list]
M = my_ds['price_range']


X_train, X_test, y_train, y_test =train_test_split(L, M, test_size=0.4, random_state=1)
clf = DecisionTreeClassifier(max_leaf_nodes=5)
clf = clf.fit(X_train, y_train)
A_pred = clf.predict(X_test)
print(A_pred)
plt.figure(figsize=(5, 6))
_ = tree.plot_tree(clf,
                   feature_names=column_list,
                   class_names=['battery_power','clock_speed','dual_sim	fc'	,'four_g',	
                                'int_memory',	'mobile_wt'	,'ram',	'touch_screen','wifi'],
                   filled = True,
                   rounded =True)