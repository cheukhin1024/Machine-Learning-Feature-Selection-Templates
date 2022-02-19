#MDA works as follows:
#first, it fits a model and computes its cross-validated performance; 
#second, it computes the cross-validated performance of the same fitted model, with the only difference that it shuffles the observations associated with one of the features.

import numpy as np
import pandas as pd

def featImpMDA(clf,X,y,n_splits=10):
   # feat importance based on OOS score reduction
   
   from sklearn.metrics import log_loss
   from sklearn.model_selection._split import KFold
   
   cvGen=KFold(n_splits=n_splits)
   scr0,scr1=pd.Series(),pd.DataFrame(columns=X.columns)
   for i,(train,test) in enumerate(cvGen.split(X=X)):
     X0,y0=X.iloc[train,:],y.iloc[train]
     X1,y1=X.iloc[test,:],y.iloc[test]
     fit=clf.fit(X=X0,y=y0) # the fit occurs here
     prob=fit.predict_proba(X1) # prediction before shuffling
     scr0.loc[i]=-log_loss(y1,prob,labels=clf.classes_)
     for j in X.columns:
        X1_=X1.copy(deep=True)
        np.random.shuffle(X1_[j].values) # shuffle one column
        prob=fit.predict_proba(X1_) # prediction after shuffling
        scr1.loc[i,j]=-log_loss(y1,prob,labels=clf.classes_)
  imp=(-1*scr1).add(scr0,axis=0)
  imp=imp/(-1*scr1)
  imp=pd.concat({‘mean’:imp.mean(),
    ‘std’:imp.std()*imp.shape[0]**-.5},axis=1) # CLT
 return imp

X,y=getTestData(40,5,30,10000,sigmaStd=.1)
clf=DecisionTreeClassifier(criterion=‘entropy’,max_features=1,
  class_weight=‘balanced’,min_weight_fraction_leaf=0)
clf=BaggingClassifier(base_estimator=clf,n_estimators=1000,
  max_features=1.,max_samples=1.,oob_score=False)
imp=featImpMDA(clf,X,y,10)
