'''
Descripttion: 保存评估机器学习模型的效果
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-18 22:15:18
LastEditTime: 2023-01-08 19:47:17
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

def PRF1(ytest:np.array, yprob:np.array,threshold:float=None)->dict:
    '''
    description: 警告: 如果两个模型不能输出prob和decision_function的话, 
                 模型无法计算AUC分数, 同时计算之后也没有任何意义! 
    param {np} ytest: 真实标签
    param {np} ypre: 模型预测标签
    param {np} yprob: 模型输出的每个样本的预测概率
    param {int} threshold: 阈值, 默认为0.5
    return prf1Dict: {'A','P(PPV)','R(Sen)(TPR)','Spec(TNR)','F1','AUC','AUPR','YI','threshold'}
    '''
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, average_precision_score, confusion_matrix as CM,roc_curve,matthews_corrcoef
    cm={}   
    if threshold == None:
        fpr, tpr, thr = roc_curve(ytest, yprob)
        threshold, _ = find_optimal_cutoff(tpr, fpr, threshold=thr)
    ypre = (yprob>threshold).astype(int)
    cm['AUC'] = roc_auc_score(ytest, yprob)
    cm['AUPR'] = average_precision_score(ytest, yprob)
    cm['F1']= f1_score(ytest, ypre)
    cm['MCC'] = matthews_corrcoef(ytest, ypre)
    cm['A']= accuracy_score(ytest, ypre)
    cm['R(Sen)(TPR)'] = recall_score(ytest, ypre)
    cmTemp= CM(ytest,ypre,labels=[1,0]) 
    cm['Spec(TNR)'] = cmTemp[1,1]/cmTemp[1,:].sum() #TNR
    cm['P(PPV)'] = precision_score(ytest, ypre) #PPV
    cm['YI'] = cm['R(Sen)(TPR)'] + cm['Spec(TNR)'] -1
    cm['threshold'] = threshold
    return  cm

# def PRF1(real_score:np.array, predict_score:np.array)->dict:
#     """Calculate the performance metrics.
#     Resource code is acquired from:
#     Yu Z, Huang F, Zhao X et al.
#      Predicting drug-disease associations through layer attention graph convolutional network,
#      Brief Bioinform 2021;22.

#     Parameters
#     ----------
#     real_score: true labels, ytest, shape(n,), means a row vector.
#     predict_score: model predictions, yprob, shape(n,)

#     Return
#     ---------
#     AUC, AUPR, Accuracy, F1-Score, Precision, Recall, Specificity
#     """
#     sorted_predict_score = np.array(
#         sorted(list(set(np.array(predict_score).flatten()))))
#     sorted_predict_score_num = len(sorted_predict_score)
#     thresholds = sorted_predict_score[np.int32(
#         sorted_predict_score_num * np.arange(1, 1000) / 1000)]
#     thresholds = np.mat(thresholds)
#     thresholds_num = thresholds.shape[1]

#     predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
#     negative_index = np.where(predict_score_matrix < thresholds.T)
#     positive_index = np.where(predict_score_matrix >= thresholds.T)
#     predict_score_matrix[negative_index] = 0
#     predict_score_matrix[positive_index] = 1
#     TP = predict_score_matrix.dot(real_score.T)
#     FP = predict_score_matrix.sum(axis=1) - TP
#     FN = real_score.sum() - TP
#     TN = len(real_score.T) - TP - FP - FN

#     fpr = FP / (FP + TN)
#     tpr = TP / (TP + FN)
#     ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
#     ROC_dot_matrix.T[0] = [0, 0]
#     ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
#     x_ROC = ROC_dot_matrix[0].T
#     y_ROC = ROC_dot_matrix[1].T
#     auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

#     recall_list = tpr
#     precision_list = TP / (TP + FP)
#     PR_dot_matrix = np.mat(sorted(np.column_stack(
#         (recall_list, precision_list)).tolist())).T
#     PR_dot_matrix.T[0] = [0, 1]
#     PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
#     x_PR = PR_dot_matrix[0].T
#     y_PR = PR_dot_matrix[1].T
#     aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

#     f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
#     accuracy_list = (TP + TN) / len(real_score.T)
#     specificity_list = TN / (TN + FP)

#     max_index = np.argmax(f1_score_list)
#     f1_score = f1_score_list[max_index]
#     accuracy = accuracy_list[max_index]
#     specificity = specificity_list[max_index]
#     recall = recall_list[max_index]
#     precision = precision_list[max_index]

#     cm = {
#     'AUC':auc[0, 0],
#     'AUPR':aupr[0, 0],
#     'F1':f1_score,
#     'A':accuracy,
#     'R(Sen)(TPR)':recall,
#     'Spec(TNR)':specificity,
#     'P(PPV)':precision,
#     'YI':recall + specificity - 1,
#     'threshold':thresholds.T[max_index][0][0].item()}
#     return cm
def figure_setting():
    return {
        'axisFont' : {'family':'Times New Roman', 'weight':'normal','size': 15},
        'titleFont' : {'family' : 'Times New Roman','size': 18},
        'legendFont' : {'family' : 'Times New Roman','size': 12},
        'figureSize':(12,9),
        'fileType':'png'}
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


    figSetting = figure_setting()
    plt.figure(figsize=figSetting['figureSize'])
    plt.plot(FPR, recall, 'r-', label=l + ', AUC=%.2f' % auc(FPR, recall))
    plt.plot([0, 1], [0, 1], 'b--')
    plt.title(l,fontdict=figSetting['titleFont'])
    plt.xlabel('False Positive Rate',fontdict=figSetting['axisFont'])
    plt.ylabel('True Positive Rate',fontdict=figSetting['axisFont'])
    plt.legend(frozenset(loc='lower right'),prop=figSetting['legendFont'])
    if bool(logInfo):
        plt.savefig(os.path.join(logInfo['logPath'],'%s_%s.pdf'%(logInfo['hour'],l)),dpi=400,bbox_inches = 'tight',pad_inches = 0.1)
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

    figSetting = figure_setting()
    plt.figure(figsize=figSetting['figureSize'])
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

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
    plt.xlabel('False Positive Rate', fontproperties=figSetting['axisFont'])
    plt.ylabel('True Positive Rate', fontproperties=figSetting['axisFont'])
    plt.title('ROC: %s' % l, fontproperties=figSetting['titleFont'])
    plt.legend(loc="lower right", prop=figSetting['legendFont'])
    
    # TODO: 设置标签字体稍微小一点以防盖住曲线文字
    
    if bool(logInfo):
        plt.savefig(os.path.join(logInfo['plotPath'], logInfo['hour'] +'ROC_' + l + '.pdf'), dpi=400,bbox_inches = 'tight',pad_inches = 0.1)
    else: plt.show()

def plot_ROC_kmodel(kmodel:dict,logInfo=False,suffix='')->None:
    '''
    description: 用于绘制多模型的ROC曲线
    param {dict} kmodel: 
                 eg: {'TPR_MEAN': [float],'TPR_STD': [float], 'L': [str]}
    param {*} logInfo: <- wzyFunc.dataPrep.make_logInfo()
    return {None}
    '''
    from sklearn.metrics import auc

    aucs = []
    tprs_mean = kmodel['TPR_MEAN']
    tprs_std = kmodel['TPR_STD']
    labels = kmodel['L']
    mean_fpr = np.linspace(0, 1, 100)
    
    figSetting = figure_setting()
    plt.figure(figsize=figSetting['figureSize'])
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    i = 0

    for mean_tpr, std_tpr, l in zip(tprs_mean, tprs_std, labels):
        roc_auc = auc(mean_fpr, mean_tpr)
        aucs.append(roc_auc)
        plt.plot(mean_fpr, mean_tpr, lw=1, label=l + '(AUC = {:.2f})'.format(roc_auc))

        # 在一个标准差内绘制阴影线 
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.15)
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontdict=figSetting['axisFont'])
    plt.ylabel('True Positive Rate',fontdict=figSetting['axisFont'])
    plt.title('ROC' ,fontdict=figSetting['titleFont'])
    plt.legend(loc="lower right",prop=figSetting['legendFont'])
    
    if bool(logInfo):
        plt.savefig(os.path.join(logInfo['plotPath'], logInfo['hour'] + 'ROC_kmodel_{}.pdf'.format(suffix)), dpi=300)
    plt.show()


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
        resultTab = pd.concat([resultTab,model.loc['2','model':]],axis=1,ignore_index=True)
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