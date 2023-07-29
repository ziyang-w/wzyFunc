'''
Descripttion: 保存一些需要计算之后才能绘制的图，
version: 0.1
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-10 13:51:24
LastEditTime: 2023-05-29 21:42:22
Copyright (c) 2022 by ziyang-W (ziyangw@yeah.net), All Rights Reserved. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns


def set_plot_chinese():
    sns.set_style('darkgrid')
    plt.rcParams['axes.unicode_minus'] = False

    pltform = sys.platform
    if 'win' in pltform: # pltform.index('win)>-1; pltform.find('win')>-1
        plt.rcParams['font.sans-serif']=['SimHei'] # windows
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'normal'
    if 'darwin' in pltform: # pltform.index('win)>-1; pltform.find('win')>-1
        plt.rcParams['font.family'] = 'PingFang HK' # Mac
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'normal'
    # if 'linux' in pltform: # pltform.index('win)>-1; pltform.find('win')>-1
    #     plt.rcParams['font.family'] = 'PingFang HK' # Mac


def ratio_heatmap(df,y:str,x:str,tr:list,labels=False,bins='auto',log_info=False,sigma=3):
    '''
    用于绘制风险百分比的热力图函数
    df:数据集
    x:绘制的x列名
    y:绘制的y列名
    tr:风险度列表，要计数的列表，需要转变为【数值型】的变量
    labels: []
        plt.xlabel(labels[0])
        plt.ylabel(labels[1]) 
        plt.title(labels[2])
    log_info: 传入log信息的字典, {'logPath','hour'}, 不传该参数, 默认为绘图
    TODO: auto bins，将auto设置为间隔，进行等距分箱
    '''
    
    # 排除异常值
#     xmin = df[x].mean()-sigma*df[x].std()
#     xmax = df[x].mean()+sigma*df[x].std()
#     ymax = df[y].mean()+sigma*df[y].std()
#     ymin = df[y].mean()-sigma*df[y].std()

    if bins =='auto':
        xbins = min(20,len(df[x].unique()))
        ybins = min(20,len(df[y].unique()))

    xmin = df[x].min()
    xmax = df[x].max()
    ymax = df[y].max()
    ymin = df[y].min()
    testSet = pd.concat([df[tr],
                pd.cut(df[x],bins=xbins,
                       labels=np.linspace(xmin,xmax,xbins))],axis=1).join(
                pd.cut(df[y],bins=ybins,
                       labels=np.linspace(ymin,ymax,ybins)))
    for br in tr:
        heatDFtotal = pd.pivot_table(testSet,values=br,index=y,columns=x,aggfunc='count',fill_value=0)
        heatDFpos = pd.pivot_table(testSet,values=br,index=y,columns=x,aggfunc='sum',fill_value=0)

        heatDF = heatDFpos/(heatDFtotal+0.01)
        # 对坐标进行省略小数位数
        heatDF.columns = np.round(heatDF.columns.values.tolist(),1)
        heatDF.index = np.round(heatDF.index.values.tolist(),1)
        heatDF['order'] = list(range(len(heatDF)))
        heatDF = heatDF.sort_values(by='order',ascending=False)
        heatDF.drop('order',inplace=True,axis=1)

        plt.figure(dpi=150)
        #设置字体大小和旋转
        plt.tick_params(labelsize=8)
        plt.xticks(rotation=0)
        
        sns.heatmap(heatDF,robust=True,cmap='Blues',)
        
        if bool(labels):
            plt.xlabel(labels[0])
            plt.ylabel(labels[1]) 
            plt.title(labels[2])
        else:
            plt.xlabel(x)
            plt.ylabel(y)  
            plt.title(br+'发生百分比')  

        
        if bool(log_info):
            log_path = log_info['logPath']
            if not os.path.exists(os.path.join(log_path,'plot','heatmap')):
                os.makedirs(os.path.join(log_path,'plot','heatmap'))
            plot_path = os.path.join(log_path,'plot','heatmap')
                
            plt.savefig(os.path.join(plot_path,'%s_HeatMap_'%log_info['hour']+br+'_'+x+'_'+y+'.pdf'),dpi=200)
            # print('plot has been saved in %s'%os.path.join(plot_path,'%s_HeatMap_'%log_info['hour']+br+'_'+x+'_'+y+'.pdf'))
            plt.close()
        else:
            plt.show()

def tsne_visualization(X:np.ndarray,y:np.array,title='',logInfo=False):
    '''
    description: 用于绘制tsne进行降维可视化
    param {np} matrix: 
    param {None | str} title: 作为图标的title, 同时也会作为结果保存的标签
    param {None | dict} logInfo: <- wzyFunc.dataPrep.make_logInfo()
    '''

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
#     from matplotlib import colors  # 注意！为了调整“色盘”，需要导入colors
    
    plt.figure(dpi=300)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0,n_iter=1000)
    tsne_results = tsne.fit_transform(X)
    
    # 可视化
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1],alpha=0.3,c=y,cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y')
    if bool(title): # False for title == ''
        plt.title(title)
        title += '_' # 加上下划线，为了后续保存的时候更方便查阅

    plt.show()

    if bool(logInfo):
        log_path = logInfo['logPath']
        if not os.path.exists(os.path.join(log_path,'plot','tsne')):
            os.makedirs(os.path.join(log_path,'plot','tsne'))
        plot_path = os.path.join(log_path,'plot','tsne')
        plt.savefig(os.path.join(plot_path,'%s%stsne.pdf'%(logInfo['hour'],title)),dpi=200)
        print('plot has been saved in %s'%os.path.join(plot_path,'%s%stsne.pdf'%(logInfo['hour'],title)))
        plt.close()