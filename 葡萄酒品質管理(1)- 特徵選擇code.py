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
-------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

stdsc= StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

-------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values

X_train,X_test,y_train,y_test=\
    train_test_split(X,y,test_size=0.3,random_state=0, stratify=y)
-------------------------------------------------------------------------------------------------------------------------------------------------------------
#自創一個 Sequential Backward selection SBS(循序向後選擇)    

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
-------------------------------------------------------------------------------------------------------------------------------------------------------------
#使用KNN分類器當特徵提取使用

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2,metric="minkowski" ) #最近的5個鄰近樣本

# selecting features
sbs = SBS(knn, k_features=1)  #我們保留至少1個特徵
sbs.fit(X_train_std, y_train) #此資料為從Training Dataset分出來的 (validation datasets)驗證數據集

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
print(df_wine.columns[1:][k3])   #Index(['Alcohol', 'Malic acid', 'OD280/OD315 of diluted wines'], dtype='object')

-------------------------------------------------------------------------------------------------------------------------------------------------------------
#使用LR分類器當訓練模型使用

lr.fit(X_train_std[:,k3],y_train) #使用我們選定的特徵子集合(['Alcohol', 'Malic acid', 'OD280/OD315 of diluted wines'])
train_scores = cross_val_score(lr, X_train_std[:, k3], y_train, cv=10, scoring='accuracy')
test_scores = cross_val_score(lr, X_test_std[:, k3], y_test, cv=10, scoring='accuracy')
print('Training accuracy (cross-validated):', train_scores.mean())
print('Test accuracy (cross-validated):', test_scores.mean())



#使用KNN分類器當訓練模型使用
knn.fit(X_train_std[:,k3],y_train) #使用我們選定的特徵子集合(['Alcohol', 'Malic acid', 'OD280/OD315 of diluted wines'])
train_scores = cross_val_score(knn, X_train_std[:, k3], y_train, cv=10, scoring='accuracy')
test_scores = cross_val_score(knn, X_test_std[:, k3], y_test, cv=10, scoring='accuracy')
print('Training accuracy (cross-validated):', train_scores.mean())
print('Test accuracy (cross-validated):', test_scores.mean())

------------------------------------------------------------------------------------------------------------------------------------------------------------

#使用LR分類器當特徵提取使用

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class="ovr",random_state=1, solver='lbfgs')

sbs = SBS(lr, k_features=1)  #我們保留至少1個特徵
sbs.fit(X_train_std, y_train)

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
print(df_wine.columns[1:][k3])  #Index(['Alcohol', 'Ash', 'Flavanoids'], dtype='object')

-------------------------------------------------------------------------------------------------------------------------------------------------------------
#使用LR分類器當訓練模型使用

lr.fit(X_train_std[:,k3],y_train) #使用我們選定的特徵子集合(['Alcohol', 'Ash', 'Flavanoids'])

train_scores = cross_val_score(lr, X_train_std[:, k3], y_train, cv=10, scoring='accuracy')
test_scores = cross_val_score(lr, X_test_std[:, k3], y_test, cv=10, scoring='accuracy')
print('Training accuracy (cross-validated):', train_scores.mean())
print('Test accuracy (cross-validated):', test_scores.mean())


#使用KNN分類器當訓練模型使用
knn.fit(X_train_std[:,k3],y_train) #使用我們選定的特徵子集合(['Alcohol', 'Ash', 'Flavanoids'])
train_scores = cross_val_score(knn, X_train_std[:, k3], y_train, cv=10, scoring='accuracy')
test_scores = cross_val_score(knn, X_test_std[:, k3], y_test, cv=10, scoring='accuracy')
print('Training accuracy (cross-validated):', train_scores.mean())
print('Test accuracy (cross-validated):', test_scores.mean())

-------------------------------------------------------------------------------------------------------------------------------------------------------------
#特徵值重要程度分析
from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]

forest=RandomForestClassifier(n_estimators=500,random_state=1) #500棵「決策樹」

forest.fit(X_train,y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
plt.show()

-------------------------------------------------------------------------------------------------------------------------------------------------------------
#特徵EDA

#['Alcohol', "Ash", "Flavanoids",]散佈圖矩陣圖
import matplotlib.pyplot as plt   
from mlxtend.plotting import scatterplotmatrix              

cols = ['Alcohol', "Ash", "Flavanoids",]
scatterplotmatrix(df_wine[cols].values,figsize=(10,8),
                     names=cols,alpha=0.3)
plt.tight_layout()
plt.show

#['Alcohol','OD280/OD315 of diluted wines',"Malic acid",]散佈圖矩陣圖

cols = ['Alcohol','OD280/OD315 of diluted wines',"Malic acid",]
scatterplotmatrix(df_wine[cols].values,figsize=(10,8),
                     names=cols,alpha=0.3)
plt.tight_layout()
plt.show

#熱圖
import numpy as np
from mlxtend.plotting import heatmap
cols = ['Alcohol',"Flavanoids","Ash","OD280/OD315 of diluted wines","Malic acid",'Proline' ]
cm = np.corrcoef(df_wine[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

-------------------------------------------------------------------------------------------------------------------------------------------------------------
#假說檢定


import pandas as pd
from scipy.stats import pearsonr
alcohol = df_wine['Alcohol']
od_ratio = df_wine['OD280/OD315 of diluted wines']


correlation_coefficient, p_value = pearsonr(alcohol, od_ratio) ## 計算皮爾森相關係數和p值
print("皮爾森相關係數:", correlation_coefficient)
print("p值:", p_value)

alpha = 0.05

if p_value < alpha:
    print("拒絕零假設，表示Alcohol與OD280/OD315之間存在顯著相關性。")    ## 檢查p值與顯著性水平比較
else:
    print("不拒絕零假設，表示Alcohol與OD280/OD315之間不存在顯著相關性。")

-------------------------------------------------------------------------------------------------------------------------------------------------------------
# 模型評估和超參數調校(網格搜索與k折交叉驗證)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler  #


pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1, solver='liblinear'))  


param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]      # 定義超參數範圍


param_grid = [{'logisticregression__C': param_range,                    # 設置超參數網格
               'logisticregression__penalty': ['l1', 'l2']}] 

                                                 
gs = GridSearchCV(estimator=pipe_lr, #設置 GridSearchCV
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=10,
                  n_jobs=-1)


gs.fit(X_train_std[:, k3], y_train)


print("Best cross-validated score:", gs.best_score_)
print("Best parameters found by GridSearchCV:", gs.best_params_)


best_lr = LogisticRegression(C=gs.best_params_['logisticregression__C'], # 使用最佳參數進行多次評估
                             penalty=gs.best_params_['logisticregression__penalty'],
                             random_state=1,
                             solver='liblinear')

# 交叉驗證評估
scores = cross_val_score(best_lr, X_train_std[:, k3], y_train, cv=10, scoring='accuracy')
print("Cross-validated scores with best parameters:", scores)
print("Mean accuracy with best parameters:", scores.mean())


manual_lr = LogisticRegression(C=10.0, penalty='l2', random_state=1, solver='liblinear')
manual_scores = cross_val_score(manual_lr, X_train_std[:, k3], y_train, cv=10, scoring='accuracy')
print("Cross-validated scores with manual parameters (C=10.0, penalty='l2'):", manual_scores)
print("Mean accuracy with manual parameters:", manual_scores.mean())
















