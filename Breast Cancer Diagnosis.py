import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/'                    #威斯康辛乳癌數據集:  569個樣本(惡性與良性)，32個特徵
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)

df.head()

--------------------------------------------------------------------------------------------------
#將字串轉為整數
from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

--------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \    #訓練數據集80%；測試數據集20%
    train_test_split(X, y,
                     test_size=0.20,
                     stratify=y,
                     random_state=1)

--------------------------------------------------------------------------------------------------
#結合轉換器與估計器到管線中

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver="lbfgs"))

pipe_lr.fit(X_train,y_train)
y_pred = pipe_lr.predict(X_test)
print("Test Accuracy: %3f" % pipe_lr.score(X_test,y_test))

--------------------------------------------------------------------------------------------------
#使用k折交叉驗證法

import numpy as np
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10).split(X_train,y_train) #k=10

scores=[]

for k , (train,test) in enumerate(kfold):
    pipe_lr.fit(X_train[train],y_train[train])           #traing set
    score = pipe_lr.score(X_train[test],y_train[test])   # validation set
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))           #np.bincount 計算每個類別的樣本數


print("\n CV accuracy: % .3f +/- %.3f " %(np.mean(scores),np.std(scores)))

--------------------------------------------------------------------------------------------------
#運用學習曲線診斷偏誤與變異數

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1,
                                           solver='lbfgs', max_iter=10000))


train_sizes, train_scores, test_scores =\
                learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)


train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
# plt.savefig('images/06_05.png', dpi=300)
plt.show()

--------------------------------------------------------------------------------------------------
#運用驗證曲線討論低度適合與過度適合
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr,
                X=X_train,
                y=y_train,
                param_name="logisticregression__C",
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi=300)
plt.show()

--------------------------------------------------------------------------------------------------
#使用Grid Search

#SVM模型
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]


gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring="accuracy",
                  refit=True,
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)                                    #SVM模型中，當svc__C = 100.0 且 svc_gamma = 0.001時，正確率最高

clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))  #Test accuracy: 0.974



#決策樹模型

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_range=[1,2,3,4,5,6,7,8,9,10,11,12,13]

param_grid = [{"max_depth":[1,2,3,4,5,6,7,None]}]

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)                                     #決策樹模型中，當max_depth = 7 時，正確率最高

clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(X_test, y_test))   #Test accuracy: 0.939

--------------------------------------------------------------------------------------------------
#以巢狀交叉驗證選擇演算法

#SVM模型
from sklearn.model_selection import cross_val_score
gs = GridSearchCV(estimator=pipe_svc,    #SVM模型
                  param_grid=param_grid,
                  scoring="accuracy",
                  cv=2)

scores = cross_val_score(gs,X_train,y_train,
                         scoring="accuracy",cv=5)

print('CV accuracy: %.3f +/- %.3f' %(np.mean(scores),  #CV accuracy: 0.974 +/- 0.015
                                     np.std(scores))) 


#決策樹模型
from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), #決策樹模型
                  param_grid=[{"max_depth":[1,2,3,4,5,6,7,None]}],
                  scoring="accuracy",
                  cv=2)

scores = cross_val_score(gs,X_train,y_train,
                         scoring="accuracy",cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),             #CV accuracy: 0.934 +/- 0.016
                                      np.std(scores)))

#可以發現SVM模型效能(97.4%)>決策樹模型效能(93.4%%)。前者的分類效能較好

--------------------------------------------------------------------------------------------------
#混沌矩陣指標
from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train,y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test,y_pred=y_pred) #confmat confused matrix
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
plt.show()
#y_test中有114筆資料，71筆(TP)；40筆(TN)；1筆(FN)；2筆(FP)
--------------------------------------------------------------------------------------------------
#最佳化分類模型的精確度與召回率

from sklearn.metrics import precision_score,recall_score,f1_score

print("Precision: %.3f" % precision_score(y_true=y_test,y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

from sklearn.metrics import make_scorer

scorer = make_scorer(f1_score,pos_label=0)

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_gamma_range,
               'svc__gamma': c_gamma_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)                     #0.9861994953378878
print(gs.best_params_)                    #{'svc__C': 10.0, 'svc__gamma': 0.01, 'svc__kernel': 'rbf'}

-------------------------------------------------------------------------------------------------
#製作接收操作特徵

from concurrent.futures import thread
from sklearn.metrics import roc_curve,auc
from distutils.version import LooseVersion as Version
from scipy import __version__ as scipy_version
from sklearn.model_selection import StratifiedKFold


if scipy_version >= Version('1.4.1'):
    from numpy import interp
else:
    from scipy import interp

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty="l2",
                                           random_state=1,
                                           solver="lbfgs",
                                           C=100.0))

X_train2 = X_train[:,[4,14]]  #我們只使用兩個feature

cv = list(StratifiedKFold(n_splits=3).split(X_train,y_train)) #折數減為3個

fig = plt.figure(figsize=(7, 5))

mean_tpr =0.0
mean_fpr = np.linspace(0,1,100)
all_tpr = []

for i , (train,test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test], #fpr(False Positive Rate 假陽性) ；  tpr(True Positive Rate 真陽性)
                                     probas[:,1], #為label 1 的機率
                                     pos_label=1
                                     )

    mean_tpr +=interp(mean_fpr, fpr ,tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,
             tpr,
             label='ROC fold %d (area = %0.2f)'
                   % (i+1, roc_auc))

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='Perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
# plt.savefig('images/06_10.png', dpi=300)
plt.show()

