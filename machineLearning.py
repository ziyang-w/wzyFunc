'''
Descripttion: Say something
version: 0.1
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-08 13:35:53
LastEditTime: 2023-07-01 19:38:14
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

#======================= 单模型 ===========================
def rf(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='',tag=False,logInfo=False,
       best_params:dict=None)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve
    d = {'model':'RF','suffix':'resultRF','l':'Random Forest'}
    if best_params is not None:
        model = RandomForestClassifier(**best_params).fit(xtrain, ytrain)
    else:
        model = RandomForestClassifier(n_estimators=100,
                                       class_weight='balanced',
                                       random_state=random_state).fit(xtrain, ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     me.plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = me.PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = me.PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest.reshape(-1), 'ypre':ypre.reshape(-1), 'yprob':yprob.reshape(-1)}
            dp.save_csv(df=pd.DataFrame(dataDict), suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo,subFolder='foldResult')
    
    return prf1Dict,model

def lgbm(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='',tag=False, logInfo=False,
       best_params:dict=None)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from lightgbm import LGBMClassifier as LGBMC
    from sklearn.metrics import roc_curve
    d = {'model':'LGBM','suffix':'resultLGBM','l':'LightGBM'}
    if best_params is not None:
        model = LGBMC(**best_params).fit(xtrain, ytrain)
    else:
        model = LGBMC(num_leaves=60,
                        learning_rate=0.05,
                        n_estimators=100,
                        class_weight='balanced',
                        random_state=random_state).fit(xtrain, ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     me.plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = me.PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = me.PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest.reshape(-1), 'ypre':ypre.reshape(-1), 'yprob':yprob.reshape(-1)}
            dp.save_csv(df=pd.DataFrame(dataDict), suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo,subFolder='foldResult')
    return prf1Dict,model

def xgb(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='', tag=False,logInfo=False,
       best_params:dict=None)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    只有在tag字段中传入包含fold键值对和logInfo, 才会保存模型的预测数据, 即ytest, ypre, yprob
    '''
    from xgboost import XGBRFClassifier as XGBC
    from sklearn.metrics import roc_curve
    d = {'model':'XGB','suffix':'resultXGB','l':'XGBoost'}
    if best_params is not None:
        model = XGBC(**best_params).fit(xtrain, ytrain)
    else:
        model = XGBC(n_estimators =100,
                            random_state=random_state,
                            learning_rate=0.1,
                            booster='gbtree',
                            objective='reg:logistic'
                            #  is_unbalance=True,
                            #  scale_pos_weight=len(ytest)/sum(ytrain)
                            #silent=False
                        ).fit(xtrain,ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     me.plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = me.PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = me.PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest.reshape(-1), 'ypre':ypre.reshape(-1), 'yprob':yprob.reshape(-1)}
            dp.save_csv(df=pd.DataFrame(dataDict), suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo,subFolder='foldResult')
    return prf1Dict,model

def lr(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='',tag=False,logInfo=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    d = {'model':'LR','suffix':'resultLR','l':'Logistic Regression'}
    model = LogisticRegression(max_iter=1000, class_weight='auto').fit(xtrain,ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     me.plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = me.PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = me.PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest.reshape(-1), 'ypre':ypre.reshape(-1), 'yprob':yprob.reshape(-1)}
            dp.save_csv(df=pd.DataFrame(dataDict), suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo,subFolder='foldResult')
        
    return prf1Dict,model

def nb(xtrain:np.array,
       ytrain:np.array,
       xtest:np.array,
       ytest:np.array,
       random_state:int, suffix='',tag=False,logInfo=False)->dict:
    '''
    用于单次调用机器学习模型, 主要用于kfold_model调用,
    '''
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import roc_curve
    d = {'model':'NB','suffix':'resultNB','l':'Naive Bayes'}
    model = GaussianNB().fit(xtrain, ytrain)
    ypre = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
#     plot_ROC(yprob, ytest, l = d['l'],logInfo=False)
    # prf1Dict = PRF1(np.array(ytest), ypre, yprob)
    prf1Dict = me.PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dataDict = {'ytest':ytest.reshape(-1), 'ypre':ypre.reshape(-1), 'yprob':yprob.reshape(-1)}
            dp.save_csv(df=pd.DataFrame(dataDict), suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo,subFolder='foldResult')
    
    return prf1Dict

def model_voting(xtrain:np.array, ytrain:np.array, xtest:np.array, ytest:np.array,bestPara:dict,
                random_state:int=42, modelList:list=['RF','LGBM','XGB'],voting:str='soft', suffix='',tag=False,logInfo=False)->dict:
    '''
    bestPara:{'RF': RandomForestClassifier{'max_depth': 8, 'n_estimators': 100}} 直接保存的是模型
    '''
    # modelList = [model+'Vote' for model in modelList] # 因为后续会用到globals()函数，防止模型名称冲突，所以需要为变量名加上Vote后缀
    d = {'model':'_'.join(modelList),
          'suffix':'result_'+'_'.join(modelList),
          'l':'Voting_'+'_'.join(modelList)}
    from sklearn.linear_model import LogisticRegression
    lrVote = LogisticRegression(max_iter=1000, class_weight='auto')

    from sklearn.naive_bayes import GaussianNB
    nbVote = GaussianNB()

    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    if 'RF' in bestPara.keys():
        rfVote=bestPara['RF']
    else:
        rfVote = RandomForestClassifier(n_estimators=100,
                                class_weight='balanced',
                                random_state=random_state)


    from lightgbm import LGBMClassifier as LGBMC
    if 'LGBM' in bestPara.keys():
        lgbmVote=bestPara['LGBM']
    else:
        lgbmVote = LGBMC(num_leaves=60,
                        learning_rate=0.05,
                        n_estimators=100,
                        class_weight='balanced',
                        random_state=random_state)    

    from xgboost import XGBClassifier as XGBMC
    if 'XBG' in bestPara.keys():
        xgbVote=bestPara['XGB']
    else:
        xgbVote = XGBMC(n_estimators =150,
                        random_state=random_state,
                        learning_rate=0.1,
                        booster='gbtree',
                        objective='reg:logistic',
                        max_depth=4,
                        n_jobs=-1,
                        scale_pos_weight=(len(ytrain)/sum(ytrain)).item()
                )
    # TODO: 应用ensemble.named_estimators_['lrVote'].predict_proba(xtest)或者predict获取模型预测结果，并将结果拼起来
    # TODO: 应用combinations函数取modelList的组合，并将结果拼起来
    # for model in combinations(estList,2): # 将模型列表中的模型组合成两两组合
    estList = {'LR':('LR',lrVote),'RF':('RF',rfVote),'XGB':('XGB',xgbVote),'LGBM':('LGBM',lgbmVote),'NB':('NB',nbVote)}
    estList = [estList.get(model) for model in modelList] # 选择模型列表中的模型
    # print(estList)

    ensembleModel = VotingClassifier(estimators=estList,voting=voting).fit(xtrain,ytrain)
    ypre = ensembleModel.predict(xtest)
    yprob = ensembleModel.predict_proba(xtest)[:, 1]

    prf1Dict = me.PRF1(np.array(ytest), yprob)
    prf1Dict['model']=d['model']
    dataDict = {'ytest':ytest.reshape(-1), 'ypre':ypre.reshape(-1), 'yprob':yprob.reshape(-1)}
    print('{} : AUC={:.4f}, AUPR={:.4f}'.format(prf1Dict['model'],prf1Dict['AUC'],prf1Dict['AUPR']))
    if bool(tag): # 将自定义标签添加到prf1Dict中
        for k,v in zip(tag.keys(),tag.values()):
            prf1Dict[k]=v
            dataDict[k]=v
        if 'fold' in tag.keys() and bool(logInfo): # 如果传入fold字段, 则将保存模型的预测数据
            dp.save_csv(df=pd.DataFrame(dataDict), suffix=suffix+'_{}_fold{}'.format(d['model'],tag['fold']),logInfo=logInfo,subFolder='foldResult')
    
    return prf1Dict,dataDict,ensembleModel

### =======================单模型 kfold======================
def rf_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,logInfo=False,fold=5,gridSearch=False,bestPara=None)->pd.DataFrame:
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

    #gridSearch
    if gridSearch and bestPara is None:
        print('Grid Searching, please wait...')
        from sklearn.model_selection import GridSearchCV 
        param_grid = {'n_estimators': range(50,150,20),'max_depth':range(4,8)}

        GCV = GridSearchCV(RandomForestClassifier(random_state=random_state), param_grid, scoring='roc_auc', n_jobs=-1, cv=fold, verbose=0)
        GCV.fit(np.array(xDF), np.array(yDF))
        bestModel = GCV.best_estimator_

        print('Best parameters found by grid search are:', GCV.best_estimator_)
    elif bestPara is not None:
        bestModel = RandomForestClassifier(**bestPara)
        print('Got best para from bestPara without Grid Search',bestModel)
    else:
        bestModel = RandomForestClassifier(n_estimators=100,
                                       class_weight='balanced',
                                       random_state=random_state)
        print('using default parameters: ',bestModel) 

    d = {'model':'RF','suffix':'resultRF','l':'Random Forest'}
    skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    KFold = skfolds.split(xDF, yDF)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    opts = []
    prf1List=[]
    logDF = pd.DataFrame()
    i = 1
    for train_index, test_index in KFold:
        xtrain = xDF.iloc[train_index, :]
        ytrain = yDF.iloc[train_index]
        xtest = xDF.iloc[test_index, :]
        ytest = yDF.iloc[test_index]

        model = bestModel.fit(np.array(xtrain), np.array(ytrain))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        prf1Dict = me.PRF1(np.array(ytest), yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)

        # 保存每一折的ytest和yprob，便于后续绘图
        #TODO: test
        r = pd.DataFrame([ytest,yprob]).T
        r.columns=['ytest','yprob']
        r['fold']=i
        logDF = pd.concat([logDF,r],axis=0)
        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List), logInfo=logInfo, suffix=d['suffix'],fileName=False)
        dp.save_csv(logDF, logInfo=logInfo, suffix='log_'+d['suffix'],fileName=False) #TODO: test

    return pd.DataFrame(prf1List),model


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
        prf1Dict = me.PRF1(np.array(ytest), yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)

        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List), logInfo=logInfo, suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List),model


def lgbm_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,logInfo=False,fold=5,gridSearch=False,bestPara:dict=None)->pd.DataFrame:
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
    
    #gridSearch
    if gridSearch:
        print('Grid Searching, please wait...')
        from sklearn.model_selection import GridSearchCV 
        param_grid = {'n_estimators': range(20,150,30),'learning_rate':[0.1,0.05,0.02],
                    'max_depth':range(4,8,2),'num_leaves':range(10,50,15),'min_child_samples':range(10,50,15) }

        GCV = GridSearchCV(LGBMC(random_state=random_state,is_unbalance=True), param_grid, scoring='roc_auc', n_jobs=-1, cv=fold, verbose=0)
        GCV.fit(np.array(xDF), np.array(yDF))
        bestModel = GCV.best_estimator_
        print('Best parameters found by grid search are:', GCV.best_estimator_)
    elif bestPara is not None:
        bestModel = LGBMC(**bestPara)
        print('Got best para from bestPara without Grid Search',bestModel)
    else:
        bestModel = LGBMC(num_leaves=30,max_depth=6,learning_rate=0.05,n_estimators=150)
        print('using default parameters: ',bestModel) 

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
        
        model = bestModel.fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        prf1Dict = me.PRF1(np.array(ytest), yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)

        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List), logInfo=logInfo, suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List),model


def xgb_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,logInfo=False,fold=5,gridSearch=False,bestPara:dict=None)->pd.DataFrame:
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

    #gridSearch
    if gridSearch:
        print('Grid Searching, please wait...')
        from sklearn.model_selection import GridSearchCV 
        param_grid = {'n_estimators': range(50,150,20),'max_depth':range(4,8),'learning_rate':[0.02,0.05,0.08]}

        GCV = GridSearchCV(XGBC(random_state=random_state,booster='gbtree',objective='reg:logistic'), 
                           param_grid, scoring='roc_auc', n_jobs=-1, cv=fold, verbose=0)
        GCV.fit(np.array(xDF), np.array(yDF))
        bestModel = GCV.best_estimator_
        print('Best parameters found by grid search are:', GCV.best_estimator_)
    elif bestPara is not None:
        bestModel = XGBC(**bestPara)
        print('Got best para from bestPara without Grid Search',bestModel)
    else:
        bestModel = XGBC(n_estimators =100,learning_rate=0.1,booster='gbtree',objective='reg:logistic',
                     random_state=random_state,
                    )
        print('using default parameters: ',bestModel) 

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

        model = bestModel.fit(np.array(xtrain), np.array(ytrain))
        ypre = model.predict(np.array(xtest))
        yprob = model.predict_proba(np.array(xtest))[:, 1]

        fpr, tpr, thr = roc_curve(ytest, yprob)
        opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
        opts.append(opt)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        prf1Dict = me.PRF1(np.array(ytest), yprob, threshold=opt[0])
        prf1Dict['model']=d['model']
        prf1List.append(prf1Dict)
        
        i += 1
    me.plot_ROC_kfold(tprs, opts, l=d['l'],logInfo=logInfo)
    if bool(logInfo):
        dp.save_csv(pd.DataFrame(prf1List), logInfo=logInfo, suffix=d['suffix'],fileName=False)
    return pd.DataFrame(prf1List),model


def svm_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,tag=False,logInfo=False,fold=5)->pd.DataFrame:
    '''TODO: save ytest, yprob to pickle
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



# 统一调用多模型融合比较的接口函数
def muti_model(xtrain:np.array,
                ytrain:np.array,
                xtest:np.array,
                ytest:np.array,
                tag:dict,
                random_state:int,suffix='',logInfo=False,best_para=None) -> dict:
    '''
    description: 通过传入划分好的测试集和训练集, 对数据进行多模型建模和验证
    param {np.array} xtrain, ytrain, xtest, ytest <- skearn.model_selection.KFold().split(X,Y) !! 注意顺序不同 !!
    param {dict | None} tag: 自定义传入结果字典的标签
                             若要在交叉验证中调用并且保存ytest, ypre, yprob, 则需要传入tag={'fold':k}
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

        prf1Dict = muti_model(xtrain,ytrain,xtest,ytest,tag={'fold':fold},random_state=random_state,logInfo=logInfo)

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
    if best_para is None:
        prf1List.append(rf(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix)[0])
        prf1List.append(lgbm(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix)[0])
        prf1List.append(xgb(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix)[0])
        prf1List.append(lr(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix)[0])
        prf1List.append(model_voting(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix)[0])
    else:
        prf1List.append(rf(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix,best_params=best_para['RF'].get_params())[0])
        prf1List.append(lgbm(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix,best_params=best_para['LGBM'].get_params())[0])
        prf1List.append(xgb(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix,best_params=best_para['XGB'].get_params())[0])
        prf1List.append(lr(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix)[0])
        prf1List.append(model_voting(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo,suffix=suffix,bestPara=best_para)[0])
    return prf1List

def get_best_model(xtrain:np.array,
                    ytrain:np.array,
                    random_state:int=0,
                    ) -> dict:
    
    '''
    description: 通过传入划分好的测试集和训练集, 对数据进行多模型建模和验证
    param {np.array} xtrain, ytrain, xtest, ytest <- skearn.model_selection.KFold().split(X,Y) !! 注意顺序不同 !!
    param {dict | None} tag: 自定义传入结果字典的标签

    --------example:-----------
    '''
    from sklearn.model_selection import GridSearchCV 

    bestPara={}
    print('Grid Searching, please wait...')

    # RF
    from sklearn.ensemble import RandomForestClassifier
    param_grid = {'n_estimators': range(50,150,20),'max_depth':range(4,8)}
    GCV = GridSearchCV(RandomForestClassifier(random_state=random_state), param_grid, scoring='roc_auc', n_jobs=-1, cv=5, verbose=0)
    GCV.fit(np.array(xtrain), np.array(ytrain))
    print('RF：Best parameters found by grid search are:', GCV.best_params_,'AUC',GCV.best_score_)
    bestPara['RF']=GCV.best_estimator_

    # LGBM
    from lightgbm import LGBMClassifier as LGBMC
    param_grid = {'n_estimators': range(20,150,30),'learning_rate':[0.1,0.05,0.02],
                    'max_depth':range(4,8,2),'num_leaves':range(10,50,15),'min_child_samples':range(10,50,15) }
    GCV = GridSearchCV(LGBMC(random_state=random_state), param_grid, scoring='roc_auc', n_jobs=-1, cv=5, verbose=0)
    GCV.fit(np.array(xtrain), np.array(ytrain))
    print('LGBM：Best parameters found by grid search are:', GCV.best_params_,'AUC:',GCV.best_score_)
    bestPara['LGBM']=GCV.best_estimator_
    
    # XGB
    from xgboost import XGBClassifier as XGBMC
    param_grid = {'n_estimators': range(50,150,20),'max_depth':range(4,8),'learning_rate':[0.02,0.05,0.08]}
    GCV = GridSearchCV(XGBMC(random_state=random_state), param_grid, scoring='roc_auc', n_jobs=-1, cv=5, verbose=0)
    GCV.fit(np.array(xtrain), np.array(ytrain))
    print('XGB：Best parameters found by grid search are:', GCV.best_params_,'AUC:',GCV.best_score_)
    bestPara['XGB']=GCV.best_estimator_

    return bestPara




# =============================最终封装函数==========================
def model_kfold(xDF:pd.DataFrame,yDF:pd.DataFrame,random_state:int,tag=False,kfold=5,logInfo=False,suffix='',gridSearch=False):
    '''
    description: 调用多个模型的n折交叉验证结果, 并将ROC和结果保存, 单独调用可见muti_model
    param {np} xDF: 
    param {np} yDF: 
    param {int} random_state: 
    param {dict | None} tag: 自定义标签, 写入到prf1Dict中, 注意，只支持传入
    param {int} nfold: 
    param {dict | None} logInfo: <- wzyFunc.dataPrep.make_logInfo()
    return {DF} prf1DF
    '''

    import os 
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_curve
    # skfolds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state).split(xDF, yDF)
    fold=1
    prf1DF=pd.DataFrame()
    got_best_para=False


    for train_index, test_index in kf:
        xtrain = np.array(xDF.iloc[train_index, :])
        ytrain = np.array(yDF.iloc[train_index]).reshape(-1,1)
        xtest = np.array(xDF.iloc[test_index, :])
        ytest = np.array(yDF.iloc[test_index]).reshape(-1,1)
        print('=========={}-Cross Validation: Fold {}==========='.format(kfold,fold))
        
        # 如果放在循环外面，相当于看了所有的数据，会数据泄露。
        if gridSearch==False:
            bestPara = None
        elif got_best_para==False:
            bestPara = get_best_model(xtrain,ytrain,random_state=random_state)
            got_best_para=True
        else:
            pass
            
        # OTHER CODE
        # TODO: 1. 传入的参数可以是一个字典，包含所有的参数，然后在函数内部进行解包

        prf1Dict = muti_model(xtrain,ytrain.reshape(-1,1),xtest,ytest.reshape(-1,1),
                                tag={'fold':fold},random_state=random_state,logInfo=logInfo,best_para=bestPara)

        # 便于后续调试和添加内容
        # prf1_rf   = ml.rf(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo)
        # prf1_lgbm = ml.lgbm(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo)
        # prf1_xgb  = ml.xgb(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo)
        # prf1_lr   = ml.lr(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo)
        # prf1_vote = ml.model_voting(xtrain,ytrain,xtest,ytest,tag=tag,random_state=random_state,logInfo=logInfo)
        
        if bool(tag): # 将自定义标签添加到prf1Dict中
            for k,v in zip(tag.keys(),tag.values()):
                prf1Dict[k]=v
        # print('AUC: {:.4}, AUPR: {:.4f}'.format(prf1Dict['AUC'],prf1Dict['AUPR']))
        prf1DF = pd.concat([prf1DF,pd.DataFrame(prf1Dict)])
        fold +=1


    # group = prf1DF.groupby('model')

    # 直接从log中读取文件，并用该数据绘制多模型ROC
    tPath = os.path.join(logInfo['logPath'], 'foldResult', str(logInfo['hour']))

    kmodel = {'TPR_MEAN': [], 'TPR_STD': [], 'OPTS': [], 'L': []}
    for model in ['LGBM', 'XGB', 'RF', 'RF_LGBM_XGB','LR']:
        tprs = []
        for i in range(kfold):
            f = pd.read_csv('{}_{}_fold{}.csv'.format(tPath,model,i+1))
            me
            fpr, tpr, thr = roc_curve(f.ytest, f.yprob)
            opt = me.find_optimal_cutoff(tpr, fpr, threshold=thr)
            tprs.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            tprs[-1][0] = 0.0

        kmodel['TPR_MEAN'].append(np.mean(tprs, axis=0))
        kmodel['TPR_STD'].append(np.std(tprs, axis=0))
        kmodel['L'].append(model)

    me.plot_ROC_kmodel(kmodel,logInfo,suffix)
    dp.save_csv(prf1DF,logInfo,'kmodelPRF1'+suffix)
    return prf1DF