'''
Descripttion: Say something
version: 0.1
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-08 13:35:53
LastEditTime: 2022-06-25 12:25:49
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
                    #   is_unbalance=True,
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


def svm_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,tag=False,logInfo=False,fold=5)->pd.DataFrame:
    '''
    description: 
    param {pd} xDF: 
    param {pd} yDF: 
    param {int} random_state: 
    param {int} tag: 自定义标签  # 好像暂时没有用到自定义tag的需求
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
        if bool(tag): # 将自定义标签添加到prf1Dict中
            for k,v in zip(tag.keys(),tag.values()):
                prf1Dict[k]=v
        
        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List),logInfo=logInfo,suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List)


def rf(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, tag=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve
    d = {'model':'RF','suffix':'resultRF','l':'Random Forest'}
    model = RandomForestClassifier(n_estimators=100,
                                   class_weight='balanced',
                                   random_state=random_state).fit(xtrain, ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     me.plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    prf1Dict = me.PRF1(np.array(ytest), ypre, yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['modle'],prf1Dict['AUC'],prf1Dict['AUPR'],))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
    
    return prf1Dict

def lgbm(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, tag=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from lightgbm import LGBMClassifier as LGBMC
    from sklearn.metrics import roc_curve
    d = {'model':'LGBM','suffix':'resultLGBM','l':'LightGBM'}
    model = LGBMC(num_leaves=60,
                      learning_rate=0.05,
                      n_estimators=100,
                      class_weight='auto',
                      random_state=random_state).fit(xtrain, ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     me.plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    prf1Dict = me.PRF1(ytest, ypre, yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['modle'],prf1Dict['AUC'],prf1Dict['AUPR'],))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
    
    return prf1Dict

def xgb(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, tag=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from xgboost import XGBRFClassifier as XGBC
    from sklearn.metrics import roc_curve
    d = {'model':'XGB','suffix':'resultXGB','l':'XGBoost'}
    model = XGBC(n_estimators =100,
                         random_state=random_state,
                         learning_rate=0.1,
                         booster='gbtree',
                         objective='reg:logistic',
                         is_unbalance=True,
                         scale_pos_weight=len(ytest)/sum(ytrain)
                         #silent=False
                    ).fit(xtrain,ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     me.plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    prf1Dict = me.PRF1(np.array(ytest), ypre, yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['modle'],prf1Dict['AUC'],prf1Dict['AUPR'],))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
    
    return prf1Dict

def lr(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, tag=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    d = {'model':'XGB','suffix':'resultXGB','l':'XGBoost'}
    model = LogisticRegression(max_iter=1000, class_weight='auto').fit(xtrain,ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     me.plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    prf1Dict = me.PRF1(np.array(ytest), ypre, yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['modle'],prf1Dict['AUC'],prf1Dict['AUPR'],))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
    
    return prf1Dict


def muti_model(xtrain:np.array,
                ytrain:np.array,
                xtest:np.array,
                ytest:np.array,
                tag:dict,
                random_state:int) -> dict:
    '''
    description: 通过传入划分好的测试集和训练集, 对数据进行多模型建模和验证
    param {np.array} xtrain, ytrain, xtest, ytest <- skearn.model_selection.KFold().split(X,Y) !! 注意顺序不同 !!
    param {dict | None} tag: 自定义传入结果字典的标签
    param {dict | None} logInfo: <- wzyFunc.dataPrep.make_logInfo()
    param {int} random_state: <- wzyFunc.machineLearning.set_seed()
    return {dict} prf1List

    --------example:-----------
    random_state = ml.set_seed(42)
    nfold=5
    # skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
    fold=1
    prf1List=[]
    for xtrain,xtest,ytrain,ytest in zip(kf.split(X),kf.split(Y)):
        print('=========={}-Cross Validation: Fold {}==========='.format(nfold,fold))
        # OTHER CODE

        prf1Dict = muti_model(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=random_state)

        print('AUC: {:.4}, AUPR: {:.4f}'.format(prf1Dict['AUC'],prf1Dict['AUPR']))
        prf1List.append(prf1Dict)
    prf1DF = pd.DataFrame(prf1Dict)
    group = prf1DF.groupby('model')
    kmodel = {
        'TPR_MEAN': group.mean()['R(Sen)(TPR)'],
        'TPR_STD': group.std()['R(Sen)(TPR)'],
        'L': list(group.mean().index)
    }
    me.plot_ROC_kmodel(kmodel,logInfo)
    '''
    prf1List=[]
    prf1List.append(rf(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state))
    prf1List.append(lgbm(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state))
    prf1List.append(xgb(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state))
    return prf1List

def model_kfold(X:np.array,Y:np.array,random_state:int,tag=False,nfold=5,logInfo=False):
    '''
    description: 调用多个模型的n折交叉验证结果, 并将ROC和结果保存, 单独调用可见muti_model
    param {np} X: 
    param {np} Y: 
    param {int} random_state: 
    param {dict | None} tag: 自定义标签, 写入到prf1Dict中, 注意，只支持传入
    param {int} nfold: 
    param {dict | None} logInfo: <- wzyFunc.dataPrep.make_logInfo()
    return {DF} prf1DF
    '''

    from sklearn.model_selection import KFold
    # skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
    fold=1
    prf1List=[]

    for xtrain,xtest,ytrain,ytest in zip(kf.split(X),kf.split(Y)):
        print('=========={}-Cross Validation: Fold {}==========='.format(nfold,fold))
        # OTHER CODE

        prf1Dict = muti_model(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=random_state)
        if bool(tag): # 将自定义标签添加到prf1Dict中
            for k,v in zip(tag.keys(),tag.values()):
                prf1Dict[k]=v
        # print('AUC: {:.4}, AUPR: {:.4f}'.format(prf1Dict['AUC'],prf1Dict['AUPR']))
        prf1List.append(prf1Dict)
        fold +=1
    prf1DF = pd.DataFrame(prf1Dict)
    group = prf1DF.groupby('model')
    kmodel = {
        'TPR_MEAN': group.mean()['R(Sen)(TPR)'],
        'TPR_STD': group.std()['R(Sen)(TPR)'],
        'L': list(group.mean().index)
    }
    me.plot_ROC_kmodel(kmodel,logInfo)

    return prf1DF