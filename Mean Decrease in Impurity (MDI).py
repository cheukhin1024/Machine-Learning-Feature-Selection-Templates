#GINI importance measures the average gain of purity by splits of a given variable. If the variable is useful, it tends to split mixed labeled nodes into pure single class nodes. 
#Splitting by a permuted variables tend neither to increase nor decrease node purities. Permuting a useful variable, tend to give relatively large decrease in mean gini-gain. 
#GINI importance is closely related to the local decision function, that random forest uses to select the best available split. 
#Therefore, it does not take much extra time to compute. 
#On the other hand, mean gini-gain in local splits, is not necessarily what is most useful to measure, in contrary to change of overall model performance.
#Gini importance is overall inferior to (permutation based) variable importance as it is relatively more biased, more unstable and tend to answer a more indirect question.

import numpy as np
import pandas as pd

def featImpMDI(fit,featNames):
  # feat importance based on IS mean impurity reduction
  df0={i:tree.feature_importances_ for i,tree in \
    enumerate(fit.estimators_)}
  df0=pd.DataFrame.from_dict(df0,orient=‘index’)
  df0.columns=featNames
  df0=df0.replace(0,np.nan) # because max_features=1
  imp=pd.concat({‘mean’:df0.mean(),
    ‘std’:df0.std()*df0.shape[0]**-.5},axis=1) # CLT
  imp/=imp[‘mean’].sum()
  return imp


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
X,y=getTestData(40,5,30,10000,sigmaStd=.1)
clf=DecisionTreeClassifier(criterion=‘entropy’,max_features=1,
  class_weight=‘balanced’,min_weight_fraction_leaf=0)
clf=BaggingClassifier(base_estimator=clf,n_estimators=1000,
  max_features=1.,max_samples=1.,oob_score=False)
fit=clf.fit(X,y)
imp=featImpMDI(fit,featNames=X.columns)
