'''
Descripttion: 保存评估机器学习模型的效果
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-18 22:15:18
LastEditTime: 2022-06-22 11:22:38
Copyright (c) 2022 by ziyang-W (ziyangw@yeah.net), All Rights Reserved. 
'''
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import calStats as cs
import calPlot as cp
import dataPrep as dp

cp.set_plot_chinese()

def PRF1(ytest:np.array, ypre:np.array, yprob:np.array,threshold=0.5)->dict:
    '''
    description: 警告: 如果两个模型不能输出prob和decision_function的话, 
                 模型无法计算AUC分数, 同时计算之后也没有任何意义! 
    param {np} ytest: 真实标签
    param {np} ypre: 模型预测标签
    param {np} yprob: 模型输出的每个样本的预测概率
    param {int} threshold: 阈值, 默认为0.5
    return prf1Dict: {'A','P(PPV)','R(Sen)(TPR)','Spec(TNR)','F1','AUC','YI','threshold'}
    '''
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, average_precision_score, confusion_matrix as CM
    cm={}
    if threshold == 0.5:
        cm['A']= accuracy_score(ytest, ypre)
        cm['P(PPV)'] = precision_score(ytest, ypre) #PPV
        cm['R(Sen)(TPR)'] = recall_score(ytest, ypre)
        cmTemp= CM(ytest,ypre,labels=[1,0]) 
        cm['Spec(TNR)'] = cmTemp[1,1]/cmTemp[1,:].sum() #TNR
        cm['F1']= f1_score(ytest, ypre)
        cm['AUC'] = roc_auc_score(ytest, yprob)
        cm['AUPR'] = average_precision_score(ytest, yprob)
        cm['YI'] = cm['R(Sen)(TPR)'] + cm['Spec(TNR)'] -1
        cm['threshold'] = threshold
        return  cm
    else:
        cm['A'] = accuracy_score(ytest, [1 if x > threshold else 0 for x in yprob])
        cm['P(PPV)'] = precision_score(ytest, [1 if x > threshold else 0 for x in yprob])
        cm['R(Sen)(TPR)'] = recall_score(ytest, [1 if x > threshold else 0 for x in yprob])
        cmTemp= CM(ytest,[1 if x > threshold else 0 for x in yprob],labels=[1,0]) 
        cm['Spec(TNR)'] = cmTemp[1,1]/cmTemp[1,:].sum() #TNR
        cm['F1'] = f1_score(ytest, [1 if x > threshold else 0 for x in yprob])
        cm['AUC'] = roc_auc_score(ytest, yprob)
        cm['AUPR'] = average_precision_score(ytest, yprob)
        cm['YI'] = cm['R(Sen)(TPR)'] + cm['Spec(TNR)'] -1
        cm['threshold'] = threshold
        return cm

def plot_ROC(yprob, ytest, l,logInfo=False):
    '''
    yprob: 模型预测的概率
    ytest: y的真实值
    l: 绘制的标签
    log_info: 传入log信息的字典, {'logPath','hour'}
    '''
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    FPR, recall, _ = roc_curve(ytest.ravel(), yprob.ravel())

    plt.plot(FPR, recall, 'r-', label=l + ', AUC=%.2f' % auc(FPR, recall))
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(l)
    plt.legend()
    if bool(logInfo):
        plt.savefig(os.path.join(logInfo['logPath'],'%s_%s.pdf'%(logInfo['hour'],l)),dpi=200)
    plt.show()


def find_optimal_cutoff(TPR:np.array, FPR:np.array, threshold:np.array):
    '''
    description: 获取最佳阈值点,并返回阈值和点的位置, 
                 一般被各种机器学习模型的kfold()函数调用,
                 用于为 wzyFunc.machineLearning.plot_ROC_kfold() 函数创建opts参数
    param {np} TPR: <- sklearn.metrics.roc_curve(ytest, yprob)
    param {np} FPR: <- sklearn.metrics.roc_curve(ytest, yprob)
    param {np} threshold: <- sklearn.metrics.roc_curve(ytest, yprob)
    return  optimal_threshold: int
            point: 点的坐标, 用于wzyFunc.machineLearning.plot_ROC_kfold()绘制
    '''
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def plot_ROC_kfold(tprs:np.array, opts:tuple, l:str,logInfo=False):
    '''
    description: 绘制k折交叉验证的ROC函数图
    param {np} tprs: 
    param {tuple} opts: <- wzyFunc.machineLearning.find_optimal_cutoff()
    param {str} l: 用于设置标题和文件名, plt.title, plotFileName
    param {None|dict} logInfo: 保存图片时传入
    return None
    '''
    from sklearn.metrics import auc
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # mean_fpr = max()
    # fig = plt.figure(figsize=(12, 9), dpi=150)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    i = 0
    for tpr, opt in zip(tprs,opts):
        roc_auc = auc(mean_fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(mean_fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f, th = %0.2f)' % (i + 1, roc_auc,opt[0]))
        plt.plot(opt[1][0], opt[1][1], marker='o', color='r', alpha=0.3)
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    # 绘制均值蓝线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)
    # 绘制阴影线
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # 最后美化图像
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: %s' % l)
    plt.legend(loc="lower right")
    
    # todo 设置标签字体稍微小一点以防盖住曲线文字
    
    if bool(logInfo):
        plt.savefig(os.path.join(logInfo['plotPath'], logInfo['hour'] +'ROC_' + l + '.pdf'), dpi=300)
    else: plt.show()


def kfold_general(DF,logInfo=False)->pd.DataFrame:
    '''
    description: 从多折交叉验证的结果中计算交叉验证的方差和均值, 
                 并且与最好的模型进行t检验, 看模型是否更好
    eg:
        kfoldGeneralDF,kfoldTtestDF = wzyFunc.me.kfold_general(DF)
    or:

    param {dict} DF: <- wzyFunc.machineLearning.rf_kfold(xDF,yDF), 传入表示保存结果
                        同时可以将多个结果竖着concat起来, 一并计算, 并且与最佳模型比较统计学差异
    param {dict} logInfo: <- wzyFunc.dataPrep.make_logInfo(), 传入表示保存结果
    return  resultTab 多折交叉验证的PRF1的mean(std)
            ttestResultDF ttest结果
    '''
    resultTab=pd.DataFrame()
    modelList=pd.DataFrame()
    bestModel={'model':'','AUC':0}
    fileList=DF['model'].unique()
    for f in fileList:
        model = DF[DF['model']==f]
        # model['model'] = f
        if model['AUC'].mean()>bestModel['AUC']: # 保存AUC最好的模型
            bestModel['model'] = f
            bestModel['AUC'] = model['AUC'].mean()
        # 保存包含数据的列名
        cols = list(model.select_dtypes(np.number).columns)
        for col in cols:
            model[col+'(std)'] = str(np.round(model[col].mean(),3)) + '±' + str(np.round(model[col].std(),2))
        resultTab = pd.concat([resultTab,model.loc[2,'model':]],axis=1,ignore_index=True)
        # 保存所有prf1DF
        modelList = pd.concat([modelList,model],axis=0)
    ttestResultDF = cs.cal_ttest_result(modelList,tcol=['model'],trow=cols,CLS=bestModel['model'])
    if bool(logInfo):
        dp.save_csv(resultTab,logInfo,suffix='kfoldGeneral')
        dp.save_csv(ttestResultDF,logInfo,suffix='kfoldTtest')
    return resultTab.T,ttestResultDF


def kfold_general_fromFile(fileList:list,logInfo:dict):
    '''
    description: 从多折交叉验证的结果中计算交叉验证的方差和均值, 
                 并且与最好的模型进行t检验, 看模型是否更好
    eg:
        fileList=['LGBM','LR','RF','XGB','BN']
        logInfo={'logPath':r'D:\\2021共享杯\\02-25','hour':'08_'}
        rT,ttestDF = kfold_general_result(fileList,logInfo)

    param {list} fileList: 用于构造路径, 同时还将构造标签, 作为筛选PFR1DF结果值的标签
    param {dict} logInfo: 当时的logInfo,用于构造路径
    return  resultTab 多折交叉验证的PRF1的mean(std)
            ttestResultDF ttest结果
    '''
    resultTab=pd.DataFrame()
    modelList=pd.DataFrame()
    bestModel={'model':'','AUC':0}
    for f in fileList:
        model = pd.read_csv(os.path.join(logInfo['logPath'],logInfo['hour']+'result_'+f+'.csv'))
        model['model'] = f
        if model['AUC'].mean()>bestModel['AUC']: # 保存AUC最好的模型
            bestModel['model'] = f
            bestModel['AUC'] = model['AUC'].mean()
        # 保存包含数据的列名
        cols = list(model.select_dtypes(np.number).columns)
        for col in cols:
            model[col+'(std)'] = str(np.round(model[col].mean(),3)) + '±' + str(np.round(model[col].std(),2))
        resultTab = pd.concat([resultTab,model.loc[2,'model':]],axis=1,ignore_index=True)
        # 保存所有prf1DF
        modelList = pd.concat([modelList,model],axis=0)
    ttestResultDF = cs.cal_ttest_result(modelList,tcol=['model'],trow=cols,CLS=bestModel['model'])
    return resultTab,ttestResultDF