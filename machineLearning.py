'''
Descripttion: Say something
version: 0.1
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-08 13:35:53
LastEditTime: 2022-06-22 10:50:21
Copyright (c) 2022 by ziyang-W (ziyangw@yeah.net), All Rights Reserved. 
'''
# 用于保存评估机器学习相关函数
from pyexpat import model
import pandas as pd
import numpy as np
import torch
import random

import modelEvaluation as me
import dataPrep as dp


def set_seed(seed=42):
    '''
    description: Set random seed. eg: random_state = ml.set_seed(42)
    param {int} seed: Random seed to use
    return {int} seed
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed



def rf_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,logInfo=False,fold=5)->pd.DataFrame:
    '''
    description: 
    param {pd} xDF: 
    param {pd} yDF: 
    param {int} random_state: 
    param {None|dict} logInfo: <- wzyFunc.dataPrep.make_logInfo(), 传入表示保存结果
    param {int} fold: 
    return {pd.DataFrame} PRF1kfoldDF
    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve

    d = {'model':'RF','suffix':'resultRF','l':'Random Forest'}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    prf1List=[]
    
    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = RandomForestClassifier(n_estimators=100,
                                    random_state=random_state).fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        prf1Dict = me.PRF1(np.array(ytest), ypre, yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)

        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List), logInfo=logInfo, suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List)


def lr_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,logInfo=False,fold=5)->pd.DataFrame:
    '''
    description: 
    param {pd} xDF: 
    param {pd} yDF: 
    param {int} random_state: 
    param {None|dict} logInfo: <- wzyFunc.dataPrep.make_logInfo(), 传入表示保存结果
    param {int} fold: 
    return {pd.DataFrame} PRF1kfoldDF
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve

    d = {'model':'LR','suffix':'resultLR','l':'Logistic Regression'}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    prf1List=[]
    
    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = LogisticRegression(max_iter=1000,class_weight='auto').fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        prf1Dict = me.PRF1(np.array(ytest), ypre, yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)

        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List), logInfo=logInfo, suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List)


def lightGBM_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,logInfo=False,fold=5)->pd.DataFrame:
    '''
    description: 
    param {pd} xDF: 
    param {pd} yDF: 
    param {int} random_state: 
    param {None|dict} logInfo: <- wzyFunc.dataPrep.make_logInfo(), 传入表示保存结果
    param {int} fold: 
    return {pd.DataFrame} PRF1kfoldDF
    '''
    from lightgbm import LGBMClassifier as LGBMC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve

    d = {'model':'LGBM','suffix':'resultLGBM','l':'LightGBM'}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    prf1List=[]
    
    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = LGBMC(num_leaves=31,
                      learning_rate=0.05,
                      n_estimators=100).fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        prf1Dict = me.PRF1(np.array(ytest), ypre, yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)

        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List), logInfo=logInfo, suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List)


def XGBoost_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,logInfo=False,fold=5)->pd.DataFrame:
    '''
    description: 
    param {pd} xDF: 
    param {pd} yDF: 
    param {int} random_state: 
    param {None|dict} logInfo: <- wzyFunc.dataPrep.make_logInfo(), 传入表示保存结果
    param {int} fold: 
    return {pd.DataFrame} PRF1kfoldDF
    '''
    from xgboost import XGBRFClassifier as XGBC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve

    d = {'model':'XGB','suffix':'resultXGB','l':'XGBoost'}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    prf1List=[]

    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = XGBC(n_estimators =100,
                         random_state=random_state,
                         learning_rate=0.1,
                         booster='gbtree',
                         objective='reg:logistic',
                         #silent=False
                    ).fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        prf1Dict = me.PRF1(np.array(ytest), ypre, yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)
        
        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List), logInfo=logInfo, suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List)


def svm_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,logInfo=False,fold=5)->pd.DataFrame:
    '''
    description: 
    param {pd} xDF: 
    param {pd} yDF: 
    param {int} random_state: 
    param {None|dict} logInfo: <- wzyFunc.dataPrep.make_logInfo(), 传入表示保存结果
    param {int} fold: 
    return {pd.DataFrame} PRF1kfoldDF
    '''
    from sklearn import svm
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve

    d = {'model':'SVM','suffix':'resultSVM','l':'Sopport Vector Machine'}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    prf1List=[]
    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = svm.SVC(kernel='rbf',
                        gamma='auto',
                        C=1,
                        probability=True).fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]
        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        prf1Dict = me.PRF1(np.array(ytest), ypre, yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)
        
        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List),logInfo=logInfo,suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List)
