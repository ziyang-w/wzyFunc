# 存放批处理绘图的代码

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
# 该命令主要用于生成图片，设置之后不会在前端显示图片。对于大量绘图时启用，不然会导致内存泄漏，导致代码崩溃
matplotlib.use('agg')
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns

import calPlot as cp

cp.set_plot_chinese()

# 保存图片防止图片被裁切
# plt.savefig(os.path.join(logInfo['plotPath'],'./weight.png'),dpi=300,bbox_inches='tight')
# 

def plot_hist(df,tdAnova,log_path, suffix=''):
    '''
    循环绘制单变量的直方图
    df:数据集
    tdAnova: 要绘制的数值型变量的列表, ['col_name'], 作为图的y
    log_path: 日志路径, 将图片保存在对应的日志路径下
    # sns.displot(data=df,x=tdAnova[0],kind = 'hist',kde=True,bins=24)
    '''
    plotType='hist'+suffix
    if not os.path.exists(os.path.join(log_path,'plot',plotType)):
        os.makedirs(os.path.join(log_path,'plot',plotType))
    plot_path = os.path.join(log_path,'plot',plotType)

    for h in tqdm(tdAnova):
        try:
            sns.displot(data=df,x=h,kind = 'hist',kde=True,bins=24)
        except Exception as e:
            print('error in plot_hist: {}'.format(h))
            print(e)
            continue
        plt.savefig(os.path.join(plot_path,'%s_%s.pdf'%(plotType,h)),dpi=200)
        plt.close() 
    print('plot save in {}'.format(plot_path))

def plot_classHist(df,tdAnova,tdChi,log_path,suffix=''):
    '''
    循环绘制单变量的直方图
    df:数据集
    tdAnova: 要绘制的数值型变量的列表, ['col_name'], 作为图的y
    tdChi: 传入分类列表
    log_path: 日志路径, 将图片保存在对应的日志路径下
    # sns.displot(data=df,x=tdAnova[0],kind = 'hist',kde=True,bins=24)
    '''  
    plotType='classHist'+suffix  
    if not os.path.exists(os.path.join(log_path,'plot',plotType)):
        os.makedirs(os.path.join(log_path,'plot',plotType))
    plot_path = os.path.join(log_path,'plot',plotType)

    for ydata in tqdm(tdAnova):
        for cls in tdChi:  
            try:  
                sns.displot(data=df,hue=cls,x=ydata,kind='hist',kde=True)
            except Exception as e:
                print('error in plot_classHist: {}_{}'.format(cls,ydata))
                print(e)
                continue
            plt.savefig(os.path.join(plot_path,'%s_%s_%s.pdf'%(plotType,ydata,cls)),dpi=200)
            plt.close()
    print('plot save in {}'.format(plot_path))

def plot_joint(df,tdAnova,log_path, suffix=''):
    '''
    循环绘制两个变量的联合直方图
    df:数据集(连续性变量)
    tdAnova: 要绘制的 数值型 变量的列表, ['col_name'], 作为图的y
    log_path: 日志路径, 将图片保存在对应的日志路径下
    # sns.jointplot(data=df,x=['var1'],y=['var2'],kind='hex')
    '''
    plotType='joint'+suffix
    if not os.path.exists(os.path.join(log_path,'plot',plotType)):
        os.makedirs(os.path.join(log_path,'plot',plotType))
    plot_path = os.path.join(log_path,'plot',plotType)
                                
    from itertools import combinations
    for item in tqdm(list(combinations(tdAnova,2))):
        try:
            sns.jointplot(data=df,x=item[0],y=item[1],kind='hex')
        except Exception as e:
            print('error in plot_joint: {}'.format(item))
            print(e)
            continue
        plt.savefig(os.path.join(plot_path,'%s_%s_%s.pdf'%(plotType,item[0],item[1])),dpi=200)
        plt.close()
    print('plot save in {}'.format(plot_path))

def plot_count(df,tdChi,tr,log_path, suffix=''):
    '''
    循环绘制两个变量的联合直方图
    df:数据集(分类数据, 文本, 以便于hue中以字符显示)
    tdChi: 要绘制的 分类型 变量的列表, ['col_name'],作为图的y
    tr: 结果变量,作为图的x
    log_path: 日志路径, 将图片保存在对应的日志路径下
    # sns.jointplot(data=df,x=['var1'],y=['var2'],kind='hex')
    '''
    plotType='count'+suffix
    if not os.path.exists(os.path.join(log_path,'plot',plotType)):
        os.makedirs(os.path.join(log_path,'plot',plotType))
    plot_path = os.path.join(log_path,'plot',plotType)
                                
    for targetResult in tqdm(tr):
        for cls in tdChi:
            try:
                sns.catplot(data=df,hue=cls,x=targetResult,kind='count')
            except Exception as e:
                print('error in plot_count: {}_{}'.format(targetResult,cls))
                print(e)
                continue
            plt.savefig(os.path.join(plot_path,'%s_%s_%s.pdf'%(plotType,targetResult,cls)),dpi=200)
            plt.close()
    print('plot save in {}'.format(plot_path))

def plot_violin(df,tdAnova,tr,log_path,suffix='',tdChi=False):
    '''
    循环绘制两个变量的联合直方图
    df:数据集(分类数据, 文本, 以便于hue中以字符显示)
    tdAnova: 要绘制的数值型变量的列表, ['col_name'], 作为图的y
    tr: 结果变量,作为图的x
    tdChi: 如果需要分类的话,给tdChi传入分类列表, 此时bool(tdChi)=true, 否则不传此参数
    log_path: 日志路径, 将图片保存在对应的日志路径下
    # sns.jointplot(data=df,x=['var1'],y=['var2'],kind='hex')
    '''
    if bool(tdChi):  # 如果需要分类的话,给tdChi传入分类列表, 此时bool(tdChi)=true
        plotType='classViolin'+suffix
        if not os.path.exists(os.path.join(log_path,'plot',plotType)):
            os.makedirs(os.path.join(log_path,'plot',plotType))
        plot_path = os.path.join(log_path,'plot',plotType) 

        for targetResult in tqdm(tr):
            for cls in tdChi:
                for ydata in tdAnova:
                    try:
                        sns.catplot(data=df,hue=cls,x=targetResult,y=ydata,kind='violin',palette="pastel")
                    except Exception as e:
                        print('erroe in plot_violin: {}_{}_{}'.format(cls,targetResult,ydata))
                        print(e)
                        continue
                    plt.savefig(os.path.join(plot_path,'%s_%s_%s_%s.pdf'%(plotType,targetResult,ydata,cls)),dpi=200)
                    plt.close() # 很重要，不然会导致内存泄漏
    else:
        plotType='violin'+suffix
        if not os.path.exists(os.path.join(log_path,'plot',plotType)):
            os.makedirs(os.path.join(log_path,'plot',plotType))
        plot_path = os.path.join(log_path,'plot',plotType)

        for targetResult in tqdm(tr):
            for ydata in tdAnova:
                sns.catplot(data=df,hue=cls,x=targetResult,y=ydata,kind='violin',palette="pastel")
                plt.savefig(os.path.join(plot_path,'%s_%s_%s.pdf'%(plotType,targetResult,ydata)),dpi=200)
                plt.close() # 很重要，不然会导致内存泄漏
    print('plot save in {}'.format(plot_path))

def plot_ratio_heatmap(df,tdAnova,tr,log_info,sigma=3,suffix=''):
    from itertools import combinations
    from calPlot import ratio_heatmap
    
    for item in tqdm(combinations(tdAnova,2)):
        try:
            ratio_heatmap(df,y=item[0],x=item[1],tr=tr,bins='auto',log_info=log_info,sigma=sigma)
        except Exception as e:
            print('error in plot_joint: {}_{}'.format(item,tr))
            print(e)
            continue
    print('plot save in {}'.format(os.path.join(log_info['logPath'],'plot','heatmap'+suffix)))