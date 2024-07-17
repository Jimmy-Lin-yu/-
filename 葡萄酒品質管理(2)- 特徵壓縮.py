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

----------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split                   #70% 訓練資料 ； 30%測試資料          
X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values # X為值 y為label
X_train,X_test,y_train,y_test = \
    train_test_split(X,y,test_size=0.3,
                     stratify=y,
                     random_state=0)

----------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler                       # 數據集需經過「標準化」處理
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

----------------------------------------------------------------------------------------------------------------------------------------------
# 建立「共變異數矩陣」
cov_mat = np.cov(X_train_std.T)                                         #「共變異數矩陣」中的「特徵向量(eigen vector)」代表「主成分(最大變異數的方向)」； 對應的「特徵值(eigen value)」會定義它們的「幅度(magnitude)」
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

----------------------------------------------------------------------------------------------------------------------------------------------
#總變異數與解釋變異數
import numpy as np
import matplotlib.pyplot as plt

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------
#主成分分析

from sklearn.decomposition import PCA  #decomposition(分解)

pca=PCA()

X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_

pca=PCA(n_components=2) #帶入兩個主成分
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

plt.scatter(X_train_pca[:,0],X_train_pca[:,1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


from sklearn.linear_model import LogisticRegression
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)

lr = LogisticRegression(multi_class="ovr",random_state=1, solver='lbfgs')
lr.fit(X_train_pca, y_train)
----------------------------------------------------------------------------------------------------------------------------------------------
#計算帶入主成分後的結果
def _calc_score(self, X_train_pca, y_train, X_test_pca, y_test):
    self.estimator.fit( X_train_pca, y_train)
    y_pred = self.estimator.predict(X_test_pca)
    score = self.scoring(y_test, y_pred)  #scoring=accuracy_score
    return score

print("Training accuracy:",lr.score(X_train_pca,y_train))         #Training accuracy: 0.9838709677419355   #使用兩個主成分就能得到98% 與 92%的正確率
print("Test accuracy:",lr.score(X_test_pca,y_test))               #Test accuracy: 0.9259259259259259


----------------------------------------------------------------------------------------------------------------------------------------------
#繪製決策區域圖
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()
