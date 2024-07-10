import pandas as pd
import numpy as np

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'                #法國葡萄酒資料來源
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',      #13個特徵值
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()


from sklearn.model_selection import train_test_split                   #70% 訓練資料 ； 30%測試資料          

X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values # X為值 y為label

X_train,X_test,y_train,y_test = \
    train_test_split(X,y,test_size=0.3,
                     stratify=y,
                     random_state=0)

from sklearn.preprocessing import StandardScaler                       # 數據集需經過「標準化」處理

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)                                        # 建立「共變異數矩陣」
                                                                       # 「共變異數矩陣」中的「特徵向量(eigen vector)」代表「主成分(最大變異數的方向)」； 對應的「特徵值(eigen value)」會定義它們的「幅度(magnitude)」
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)











