'''
Descripttion: 主要包含计算统计相关的函数, 并将结果写入到log中
version: 0.2
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-09 22:52:23
LastEditTime: 2023-09-09 15:15:12
Copyright (c) 2022 by ziyang-W (ziyangw@yeah.net), All Rights Reserved. 
'''

import pandas as pd
import numpy as np

# 下面这个是Chi2分箱的的代码，绘制的图片横轴为record分类，纵轴为y的发病率，附加一个bar
# sns.catplot(data=df[df['BMI_class'].isin([2,3])],x='record',y='妊娠期糖尿病',kind='bar')

def cal_ttest_result(dataSet:pd.DataFrame, tcol:list, trow:list, showP=False, CLS=False, general=False):
    '''
    v3, 添加了统计量以及用±替代()
    description: 返回一系列的t检验结果表, 输入为【1, 0数据集】,
                 此方法默认采用方差不齐的t检验, 并且只用于做【两独立样本t检验】
    param {pd} dataSet: 
    param {list} tcol: 表示要检验的结果, 最后结果显示为标题列
    param {list} trow: 表示要检验的特征, 最后结果显示为行
    param {None|str} CLS: None-> 默认, 为tr列中的所有unique标签作为分组, 进行两两t检验
                          str -> 只以tr中的str分类, 与其他标签做t检验,用于标签中有多种分类结果的计算
    param {Bool} general: [True] -> 返回的结果表中不展示每组的mean(std), 只展示特征和p值
                          [False]-> 返回每组的mean(std), 并且附带最后的p值
    return TtestDF: 检验结果

    eg:
    r = cs.cal_ttest_result(dataSet=data,tcol=[targetY[0]],trow=tdAnova,general=True,showP=True)
    '''
    from scipy import stats
    from itertools import combinations
    TtestDF = pd.DataFrame()
    calParticialTrLabel=False # 默认计算目标列全部unique值两两t检验结果
    if bool(CLS): #TODO:改成向量，形式，与tcol长度相同能够实现对不同列进行不同的设置
        calParticialTrLabel=True # 如果传入了CLS参数, 那么就计算目标列中 CLS与其他label的结果，否则计算目标列中所有unique值的两两t检验
    for i, r in enumerate(tcol):
        # 计算全人群
        l = []
        for j, tda in enumerate(trow):
            tableDict = {}
            genDict = {}
            if i == 0:
                tableDict['feature'] = tda
                genDict['feature'] = tda
                tableDict['mean ± std'] = str(
                    np.round(dataSet[tda].mean(), 2)) + ' ± %.2f' % np.round(dataSet[tda].std(), 2)
#             print(CLS)
            if calParticialTrLabel:
                otherLabel = list(dataSet[r].unique())
                otherLabel.remove(CLS)
                # 遍历r列中的每一个label，计算不同分组下的均值和方差
                for cls in dataSet[r].unique():
                    sub0 = dataSet[dataSet[r] == cls]
                    tableDict['(%s)mean ± std' % cls] = str(
                        np.round(sub0[tda].mean(), 2))+' ± %.2f' % np.round(sub0[tda].std(), 2)
                # 便利otherLabel, 计算label与CLS的t检验结果
                for cls in otherLabel:
                    sub0 = dataSet[dataSet[r] == CLS]
                    sub1 = dataSet[dataSet[r] == cls]
                    pvalue = stats.ttest_ind(
                        sub0[tda], sub1[tda], equal_var=False)[1]
                    tvalue = stats.ttest_ind(
                        sub0[tda], sub1[tda], equal_var=False)[0]
                    tvalue = round(tvalue, 2)
                    
                    if showP:
                        beautyP = pvalue if pvalue < 0.001 else round(pvalue, 4)
                    else:
                        beautyP = 'p<0.001' if pvalue < 0.001 else round(pvalue, 4)
                    tableDict[r+'-%s-%s-t' % (CLS, cls)] = tvalue
                    tableDict[r+'-%s-%s-p' % (CLS, cls)] = beautyP
                    genDict[r+'-%s-%s-t' %(CLS, cls)] = tableDict[r+'-%s-%s-t' % (CLS, cls)]
                    genDict[r+'-%s-%s-p' %(CLS, cls)] = tableDict[r+'-%s-%s-p' % (CLS, cls)]
                if general:
                    l.append(genDict)
                else:
                    l.append(tableDict)
            else: # 代表默认对r列内所有label做两两ttest
                CLS = dataSet[r].unique()
                for cls in CLS:
                    sub0 = dataSet[dataSet[r] == cls]
                    tableDict['(%s)mean(std)' % cls] = str(
                        np.round(sub0[tda].mean(), 2))+' ± %.2f' % np.round(sub0[tda].std(), 2)
                # 遍历排列组合结果, 计算label与CLS的t检验结果
                for cls in combinations(CLS, 2):
                    sub0 = dataSet[dataSet[r] == cls[0]]
                    sub1 = dataSet[dataSet[r] == cls[1]]
                    pvalue = stats.ttest_ind(
                        sub0[tda], sub1[tda], equal_var=False)[1]
                    tvalue = stats.ttest_ind(
                        sub0[tda], sub1[tda], equal_var=False)[0]
                    tvalue = round(tvalue, 2)

                    if showP:
                        beautyP = pvalue if pvalue < 0.001 else round(pvalue, 4)
                    else:
                        beautyP = 'p<0.001' if pvalue < 0.001 else round(pvalue, 4)
                    tableDict[r+'-%s-%s-t' % (cls[0], cls[1])] = tvalue
                    tableDict[r+'-%s-%s-p' % (cls[0], cls[1])] = beautyP
                    genDict[r+'-%s-%s-t' %
                            (cls[0], cls[1])] = tableDict[r+'-%s-%s-t' % (cls[0], cls[1])]
                    genDict[r+'-%s-%s-p' %
                            (cls[0], cls[1])] = tableDict[r+'-%s-%s-p' % (cls[0], cls[1])]
                if general:
                    l.append(genDict)
                else:
                    l.append(tableDict)
        TtestDF = pd.concat([TtestDF, (pd.DataFrame(l))], axis=1)
    return TtestDF

def cal_chi_result(dataSet:pd.DataFrame, tcol:list, trow:list, general=False, beauty=True,showP=True,fisher=False)->pd.DataFrame:
    '''
    计算卡方检验结果, 并返回结果表格
    :param dataSet: 数据集
    :param tcol: 目标列
    :param trow: 目标行
    :param general: 是否只返回卡方值和p值
    :param beauty: 是否美化表格
    :param showP: 是否显示p值; 如果不显示p值, 则显示p<0.001


    eg:
    ChiDF = cal_chi_result(dataSet=singleData, tcol=tr, trow=tdChi+mutiFeature)
    ChiDF.to_csv(os.path.join(log_path,
                            '%sChi_Result.csv' % hour), encoding='utf-8-sig')

    '''
    from scipy import stats
    spDF = dataSet.astype('category')
    ChiDF = pd.DataFrame()
    for r in tcol:
        l = pd.DataFrame()
        for tda in trow:
            tableDict = pd.crosstab(
                spDF[tda], spDF[r], margins=True, margins_name='Total')
            percent = pd.crosstab(
                spDF[tda], spDF[r], margins=True, normalize='index', margins_name='Total') # index横着求和，计算得到的是分间的百分比，无意义
                # spDF[tda], spDF[r], margins=True, normalize='columns', margins_name='Total') # columns竖着求和，计算得到的是分组内的百分比

            if fisher:
                chi2 = round(stats.fisher_exact(tableDict.iloc[:-1,:-1])[0],2) #[1]:chi2,[2]:df
                pvalue = stats.fisher_exact(tableDict.iloc[:-1,:-1])[1] #[1]:chi2,[2]:df
            else:
                chi2 = round(stats.chi2_contingency(tableDict.iloc[:-1,:-1])[0],2) #[1]:chi2,[2]:df
                pvalue = stats.chi2_contingency(tableDict.iloc[:-1,:-1])[1] #[1]:chi2,[2]:df

            if showP:
                beautyP = pvalue if pvalue < 0.001 else round(pvalue, 4)
            else:
                beautyP = 'p<0.001' if pvalue < 0.001 else round(pvalue, 4)
                
            tableDict[r+'-p'] = beautyP
            tableDict[r+'-chi'] = chi2
            

            if beauty:
                ###====下面内容为优化表格====####
                #               #将百分数添加到字段后面
                # 按照index求和计算的方式
                tableDict.iloc[:, :-3] = tableDict.iloc[:, :-3]\
                        .astype(str)+'('+(percent.iloc[:, :]*100).round(1).astype(str)+'%)'

                # 按照columns求和计算的方式
                # tableDict.iloc[:-1, :-2] = tableDict.iloc[:-1, :-2]\
                #         .astype(str)+'('+(percent.iloc[:, :]*100).round(1).astype(str)+'%)'
                # 在为小表格添加一个新行
#                 pd.concat([pd.DataFrame(['随便写点儿什么'],index=[tda],columns=['rp']), tableDict]).drop('rp',axis=1)
                # 不能删掉赋值，不然pandas会默认将全部为空的行删掉
                tableDict.loc[tda, r+'-chi'] = chi2
                tableDict.loc[tda, r+'-p'] = beautyP
        

            if general:
                genDF = pd.DataFrame([chi2,beautyP], index=[r+'-chi',r+'-p'], columns=[tda]).T
                l = pd.concat([l, genDF], axis=0)
            else:
                l = pd.concat([l, tableDict], axis=0)

        ChiDF = pd.concat([ChiDF, l], axis=1)
    return ChiDF


def cal_stratify_chi_result(dataSet:pd.DataFrame,tcol:list,trow:list,stratify:str,fisher=False,general =False,beauty=True,showP=True):
    '''
    计算分组卡方,对stratify计算整体的卡方检验,然后在每个分组内再进行卡方检验
    tcol:行列表的C,最后生成该列的卡方p值,一般为targetY
    trow:行列表的行R,一般为待研究的变量
    stratify:分组的列,是根据该列将样本划分为对应的子集,并在自己内分别进行trow列关于tcol的卡方检验

    fisher: 是否采用fihser精确检验
    '''
    from scipy import stats
    spDF = dataSet.astype('category')
    ChiDF=pd.DataFrame()

    for r in tcol:
        l=pd.DataFrame()
        for tda in trow:
            sTableDict = pd.crosstab(index=[spDF[stratify], spDF[tda]],columns = spDF[r],margins=True,margins_name='Total')
            sPercent = pd.crosstab(index=[spDF[stratify], spDF[tda]],columns = spDF[r],margins=True,normalize='index',margins_name='Total')
            
#             print(sTableDict.index)
            S=pd.DataFrame()
            for s in spDF[stratify].unique():
                
                tableDict = sTableDict.loc[s]
                percent = sPercent.loc[s]

                if fisher:
                    chi2 = round(stats.fisher_exact(tableDict.iloc[:-1,:-1])[0],2) #[1]:chi2,[2]:df
                    pvalue = stats.fisher_exact(tableDict.iloc[:-1,:-1])[1] #[1]:chi2,[2]:df
                else:
                    chi2 = round(stats.chi2_contingency(tableDict.iloc[:-1,:-1])[0],2) #[1]:chi2,[2]:df
                    pvalue = stats.chi2_contingency(tableDict.iloc[:-1,:-1])[1] #[1]:chi2,[2]:df

                if showP:
                    beautyP = pvalue if pvalue < 0.001 else round(pvalue, 4)
                else:
                    beautyP = 'p<0.001' if pvalue < 0.001 else round(pvalue, 4)

                
                tableDict[r+'-chi'] = chi2
                tableDict[r+'-p'] = beautyP

                if beauty:
                    ###====下面内容为优化表格====####
    #               #将百分数添加到字段后面
                    tableDict.iloc[:,:-3] = tableDict.iloc[:,:-3].astype(str)+'('+(percent.iloc[:,:]*100).round(1).astype(str)+'%)'

                    # 按照index求和计算的方式
                    # tableDict.iloc[:, :-3] = tableDict.iloc[:, :-3]\
                    #         .astype(str)+'('+(percent.iloc[:, :]*100).round(1).astype(str)+'%)'

                    # 按照columns求和计算的方式
                    # tableDict.iloc[:, :-2] = tableDict.iloc[:, :-2]\
                    #         .astype(str)+'('+(percent.iloc[:, :]*100).round(1).astype(str)+'%)'
                    # 在为小表格添加一个新行
    #                 pd.concat([pd.DataFrame(['随便写点儿什么'],index=[tda],columns=['rp']), tableDict]).drop('rp',axis=1)
                    tableDict.loc[s,r+'-p'] = beautyP # 不能删掉赋值，不然pandas会默认将全部为空的行删掉

                if general:
                    genDF = pd.DataFrame([beautyP],columns=[r+'-p'],index=[s])
                    l = pd.concat([l,genDF],axis=0)
                else:
                    l = pd.concat([l,tableDict],axis=0)

            S = pd.concat([S,l],axis=0)
        ChiDF = pd.concat([ChiDF,S],axis=1)
    return ChiDF

    
