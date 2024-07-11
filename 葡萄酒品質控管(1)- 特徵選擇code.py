import numpy as np
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']



from sklearn.linear_model import LogisticRegression                     #經由「正規化」，對過於LR分類器引進一個「penalty(懲罰)」
LogisticRegression(penalty="l1",solver="liblinear",multi_class="ovr")

import matplotlib.pyplot as plt                                         #觀察每個特徵在L1正規化的變化

fig=plt.figure()
ax=plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights,params=[],[]

for c in np.arange(-4.,6.):
    Ir=LogisticRegression(penalty="l1",C=10.**c,solver="liblinear",
                          multi_class="ovr",random_state=0)

    Ir.fit(X_train_std,y_train)
    weights.append(Ir.coef_[1])
    params.append(10**c)

weights=np.array(weights)

for column, color in zip(range(weights.shape[1]),colors): #weights.shape[1] 表示這個權重矩陣的第二個維度的大小，也就是特徵的數量。
    plt.plot(params,weights[:,column],
             label=df_wine.columns[column+1],
             color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
#plt.savefig('images/04_07.png', dpi=300,
#            bbox_inches='tight', pad_inches=0.2)
plt.show()



from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS():                                                           #Sequential Backward selection SBS(循序向後選擇)           
    def __init__(self, estimator, k_features, scoring=accuracy_score,  #accuracy_score 來評估模型效能，並作為該「特徵子集合」分類效能的estimator
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1] #有多少個dimension
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_] #子集
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:                          #k_features參數定義了我們想要演算法「最後保留多少個特徵」
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):                                   #我們可以用transform方法，將選定的「特徵」轉換成新的「數據陣列」
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)  #scoring=accuracy_score
        return score



import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier            #使用KNN分類器

knn = KNeighborsClassifier(n_neighbors=5)                     #最近的5個鄰近樣本

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)                                 #此資料為從Training Dataset分出來的 (validation datasets)驗證數據集

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]



plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
plt.show()


k3=list(sbs.subsets_[10])

print(k3)
print(df_wine.columns[1:])
print(df_wine.columns[1:][k3])                            #Index(['Alcohol', 'Malic acid', 'OD280/OD315 of diluted wines']) 選擇出來最重要的三個特徵



knn.fit(X_train_std,y_train) #使用全部的特徵
print("Training accuracy:",knn.score(X_train_std,y_train))
print("Test accuracy:",knn.score(X_test_std,y_test))

knn.fit(X_train_std[:,k3],y_train)                        #使用我們選定的3個特徵子集合
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))
