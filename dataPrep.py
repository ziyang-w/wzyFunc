'''
Descripttion: 用于保存数据处理相关的函数代码
version: 0.1
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-08 13:35:05
LastEditTime: 2023-01-08 17:16:34
Copyright (c) 2022 by ziyang-W (ziyangw@yeah.net), All Rights Reserved. 
'''

import pandas as pd
import numpy as np
import os
import pickle

def make_logInfo(fileName:str, filePath:str) -> dict:
    '''
    description: 根据数据集fileName, 数据集路径filePath, 构造LogInfo字典
                 主要目的为了后续数据分析文件的保存
    param {str} fileName: 数据集文件名, 包含后缀
    param {str} filePath: 数据集路径
    return  logInfo: {'logPath','plotPath','date','hour','fileName','filePath'}
    '''
    from datetime import datetime
    startDate = datetime.now().strftime('%m-%d')
    hour = datetime.now().strftime('%H_')
    logPath = os.path.join(filePath, 'log', fileName.split('.')[0], startDate)

    if not os.path.exists(logPath):
        os.makedirs(logPath)
        os.makedirs(os.path.join(logPath, 'plot'))
        os.makedirs(os.path.join(logPath, 'pickle'))
    logInfo = {'logPath': logPath,
               'plotPath':os.path.join(logPath, 'plot'),
               'picklePath':os.path.join(logPath, 'pickle'),
               'date':startDate,
               'hour': hour,
               'fileName': fileName,
               'filePath': filePath}
    return logInfo

# TODO : save_pickle
def save_pickle(variable:any, logInfo:dict, suffix:str, fileName=False):
    '''
    description:  将结果保存到对应的log目录下, 
                  eg: 'filePath\\log\\fileName\\mm-dd\\hh_fileName_suffix.csv'
                  Tips: 在调用时, 一般只加一次后缀, 即在suffix参数中尽量用驼峰法命名, 而不包含'_', 方便后续查找
    param {pd} df: 要保存的DataFrame
    param {dict} logInfo: <- wzyFunc.make_logInfo()
    param {str} suffix: 想要添加的后缀名, 一般应用驼峰法命名, 而不使用'_'来进行分隔
    param { None | True } fileName: 在保存的文件名中是否加入当前分析数据集文件名后缀logInfo['fileName']
    return None
    '''
    suffix += '.pkl'
    if bool(fileName):
        tPath = os.path.join(logInfo['logPath'],
                             str(logInfo['hour'])+logInfo['fileName'].split('.')[0]+'_'+suffix)
    else:
        tPath = os.path.join(logInfo['picklePath'], str(logInfo['hour'])+suffix)
    pickle.dump(variable, open(tPath, 'wb'))
    print('file has been saved in : %s' % tPath)

def save_csv(df:pd.DataFrame, logInfo:dict, suffix:str, fileName=False,subFolder=''):
    '''
    description:  将结果保存到对应的log目录下, 
                  eg: 'filePath\\log\\fileName\\mm-dd\\hh_fileName_suffix.csv'
                  Tips: 在调用时, 一般只加一次后缀, 即在suffix参数中尽量用驼峰法命名, 而不包含'_', 方便后续查找
    param {pd} df: 要保存的DataFrame
    param {dict} logInfo: <- wzyFunc.make_logInfo()
    param {str} suffix: 想要添加的后缀名, 一般应用驼峰法命名, 而不使用'_'来进行分隔
    param { None | True } fileName: 在保存的文件名中是否加入当前分析数据集文件名后缀logInfo['fileName']
    return None
    '''
    suffix += '.csv'
    if bool(subFolder):
        if not os.path.exists(os.path.join(logInfo['logPath'], subFolder)):
            os.makedirs(os.path.join(logInfo['logPath'], subFolder))
        
    if bool(fileName):
        tPath = os.path.join(logInfo['logPath'], subFolder,
                             str(logInfo['hour'])+logInfo['fileName'].split('.')[0]+'_'+suffix)
    else:
        tPath = os.path.join(logInfo['logPath'], subFolder, str(logInfo['hour'])+suffix)
    df.to_csv(tPath,
              index=False,
              encoding='utf-8-sig')
    print('file has been saved in : %s' % tPath)


def bigfile_readlines(f, separator:str):
    '''
    description: 按照规定的切分符号去读取大文件方法
    param {*} f: 文件流
    param {str} separator: 每一行的分隔符
    yield: 一行数据，以字符串形式返回
    eg: with open('bigFile.txt','r',encoding='utf-8') as f:
            for line in readlines(f,'file_separator'):
                processing code hear
                
    REFERENCE: https://www.jb51.net/article/182539.htm
    '''
    buf = ''
    while True:
        while separator in buf:
            position = buf.index(separator) # 分隔符的位置
            yield buf[:position] # 切片, 从开始位置到分隔符位置
            buf = buf[position + len(separator):] # 再切片,将yield的数据切掉,保留剩下的数据
        chunk = f.read(4096) # 一次读取4096的数据到buf中
        if not chunk: # 如果没有读到数据
            yield buf # 返回buf中的数据
            break # 结束
        buf += chunk # 如果read有数据 ,将read到的数据加入到buf中

def missing_dect(df)->pd.DataFrame:
    '''统计每个变量的缺失数据的数量, 变量名称的DataFrame'''
    cols = {'feature_no': [], 'Variables': [], 'dtypes': [],
            'unique_shape': [], 'null_num': [], 'null_rate': []}
    for e, c in enumerate(df.columns):
        #         if sum(pd.isnull(df[c]))!=0:
        cols['feature_no'].append(e)
        cols['Variables'].append(c)
        cols['dtypes'].append(df[c].dtypes)
        cols['unique_shape'].append(len(df[c].unique()))
        cols['null_num'].append(sum(pd.isnull(df[c])))
        cols['null_rate'].append(sum(pd.isnull(df[c]))/len(df[c]))
    return pd.DataFrame(cols).sort_values(ascending=False, by='null_num')


def del_missing_feature(df:pd.DataFrame, missingRate:float) -> pd.DataFrame:
    '''
    description: 筛选确实比例<missingRate的数据集, 
                 eg:
                    0表示保留确实比例小于0的列==删除有缺失值的列, 
                    0.1为保留缺失值<0.1的缺失列
    param {pd} df: 
    param {float} missingRate: 
    return  DF: 删除之后的df
            delDF: 被删除的df
    '''
    missingDF = missing_dect(df)
    missing_features = df.loc[:, missingDF[missingDF.null_rate >
                                           missingRate].Variables].columns.values
    features = df.loc[:, missingDF[missingDF.null_rate <
                                   missingRate].Variables].columns.values
    DF = df[features]
    delDF = df[missing_features]
    print('delete number of features is : %d, missing rate : %.2f' %
          (len(missing_features), missingRate))
    return DF, delDF


def get3sigma(df:pd.DataFrame, filterList=[], ignoreList=[], sigma=3)->pd.DataFrame:
    '''
    description: 筛选异常值, 根据3sigma原则, 默认为3
    返回
    param {pd} df: 
    param {*} filterList: 要筛选的特征列名, 如果不填, 默认对全部列进行筛选
    param {list} ignoreList: 有些特征不需要进行筛选, 如分类特征, ID号, 日期等
    param {*} sigma: 按照 mean +- sigma * std筛选, 默认为 3
    return  normalDF: 在sigma个方差内的数据
            unnormalDF: 在sigma个方差之外的数据, 视为离群样本
    '''
    if len(filterList) == 0:
        filterList = df.columns.values
    myFilter = np.ones(df.shape[0]).astype(bool)
    for f in filterList:
        if df[f].dtype != object and f not in ignoreList:
            low = df[f].mean() - sigma * df[f].std()
            up = df[f].mean() + sigma * df[f].std()
            myFilter = myFilter & (low < df[f]) & (df[f] < up)
    return df[myFilter].reset_index(drop=True), df[~(myFilter)].reset_index(drop=True)


def has_feature(dataSet:pd.Series, ruleList:list) -> list:
    '''
    description: 适用于apply()函数, 将分类特征中的"是、有"等文字全部转换成"是", 其他转换成"否"
    param {pd} dataSet: 
    param {list} ruleList: 
    return {*}
    '''
    result = []
    if set(ruleList).issubset(set(dataSet.columns.tolist())):
        for i in range(len(dataSet)):
            hasdis = False
            item = dataSet.iloc[i, :]
            for test in ruleList:
                if item[test] in ['是', '有', 'POS']:
                    hasdis = True
            if hasdis:
                result.append('是')
            else:
                result.append('否')
    return result


# TODO : finish
def get_SMOTE():
    ''' 
    应用SMOTE对样本不均衡的模型进行插补
    '''

def MIC_selection(df:pd.DataFrame,targetY:str,logInfo=False,k=20)->pd.DataFrame:
    '''
    description: 计算互信息值, 并且筛选后的特征集df
    param {pd} df: 数据集
    param {str} targetY: 与名为targetY的特征计算互信息值
    param {None|dict} logInfo: <- wzyFunc.dataPrep.make_logInof(), 传入代表保存结果
    param {int} k: 保存互信息结果前k个特征
    return  xDF: features are feltered by MIC xDF.shape -> (n_sample, k)
            yDF: yDF.shape -> (n_sample, 1)
            result: MIC value
    '''
    from sklearn.feature_selection import mutual_info_classif as MIC
    xDF = df[df.columns.difference([targetY])]
    yDF = df[targetY]
    result=pd.DataFrame({'var':xDF.columns.tolist(),'MIC':MIC(xDF, yDF)}).sort_values(by='MIC',ascending=False)
    if bool(logInfo):
        save_csv(result,logInfo,suffix='MIC',fileName=True)
        save_csv(xDF[result.iloc[:k, 0].tolist()].join(yDF),logInfo,suffix='MICSelection',fileName=True)
    return xDF[result.iloc[:k,0].tolist()], yDF, result


def multi_insert(df):
    '''
    多重插补的方法对缺失数据值进行填补
    '''
    insertDF=pd.DataFrame()
    return insertDF 


def str2encoding_auto(df:pd.DataFrame,catDict=False,logInfo=False)->pd.DataFrame:
    '''
    description: 自动识别文本特征, 并将object编码转变成0123, 需要在对数据进行清洗之后才能进行引用
    param {pd} df: 目标数据集, 应该包含有结果变量
    param {None|dict} catDict: 传入有 固定编码 或者 顺序 的特征的编码,函数中采用构造字典, 
                               函数内采用df.map(dict)方式进行映射, 字典中缺失的值将被赋值为None!
                               eg: {'feature1':[value1,value2]},
                                    'feature2':[value1,value2,value3,....]}
    param {None|dict} logInfo: <- wzyFunc.dataPrep.make_logInfo, 传入代表保存结果
    return  xDF: 经过编码之后的整个数据集
            codeDF: 编码结果表
    '''
    from sklearn.preprocessing import OrdinalEncoder
    
    codeList=[]
    catList = list(df.select_dtypes(object).columns.values)
    numList = df[df.columns.difference(catList)].columns.values
    print('对这些分类变量特征进行了编码：',catList)
    if bool(catDict): # 从object列表中删除，并且对数据进行编码
        for item in catDict.keys():
            catList.remove(item) #删除传入的自定义特征进行编码 
            myDict = dict(zip(catDict[item],range(len(catDict[item])))) # 构造成值:编码字典
            # 编码
            df[item] = df[item].map(myDict).copy()
            codeList.append(dict(zip(myDict.values(),myDict.keys()))) #将编码:值字典放入
        # 对剩余内容的编码
        ode = OrdinalEncoder().fit(df[catList])
        X = ode.transform(df[catList])
        xDF = pd.DataFrame(X,columns=catList,index=df.index)
        xDF = pd.concat([df[numList],df[list(catDict.keys())],xDF],axis=1)
        for featureUnique in ode.categories_:
            codeDict={}
            for i in range(len(featureUnique)):
                codeDict[i] = featureUnique[i]
            codeList.append(codeDict)
        codeDF = pd.DataFrame(codeList,index=list(catDict.keys())+catList).reset_index().rename(columns={'index':'feature'})
        if bool(logInfo):
            save_csv(codeDF,logInfo,suffix='encoding',fileName=True)
            save_csv(xDF,logInfo,suffix='numeric',fileName=True)
        return xDF,codeDF      
    else: 
        catList = df.select_dtypes(object).columns.values
        numList = df[df.columns.difference(catList)].columns.values
        ode = OrdinalEncoder().fit(df[catList])
        X = ode.transform(df[catList])
        xDF = pd.DataFrame(X,columns=catList)
        xDF = pd.concat([df[numList],xDF],axis=1)
        for featureUnique in ode.categories_:
            codeDict={}
            for i in range(len(featureUnique)):
                codeDict[i] = featureUnique[i]
            codeList.append(codeDict)
        codeDF = pd.DataFrame(codeList,index=catList).reset_index().rename(columns={'index':'feature'})
        if bool(logInfo):
            save_csv(codeDF,logInfo,suffix='encoding',fileName=True)
            save_csv(xDF,logInfo,suffix='numeric',fileName=True)
        return xDF,codeDF

def str2encoding(df:pd.DataFrame,catFeature:list,numFeature:list,logInfo=False)->pd.DataFrame:
    '''
     description: 不会自动识别文本特征, 将对指定列返回编码表以及编码后的df
    '''
    from sklearn.preprocessing import OrdinalEncoder

#     mutiStr = pd.get_dummies(df[mutiFeature].astype('category'))

    ode = OrdinalEncoder().fit(df[catFeature])
    X = ode.transform(df[catFeature])
    XDF = pd.DataFrame(X,columns=catFeature)
    # dummyDF = pd.concat([XDF,mutiStr],axis=1)
    df = pd.concat([df[numFeature],XDF],axis=1)
    
    encodingDF = pd.DataFrame()
    for i in ode.categories_:
        encodingDF = pd.concat([encodingDF,pd.DataFrame(i)],axis=1)
    encodingDF.columns=catFeature
    encodingDF = encodingDF.T.reset_index(drop=False).rename(columns={'index':'feature'})
    
    if bool(logInfo):
        save_csv(encodingDF.T,logInfo,suffix='encoding',fileName=True)
        
    return df,encodingDF