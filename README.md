<!--
 * @Descripttion: Say something
 * @version: 0.1
 * @Author: ziyang-W, ziyangw@yeah.net
 * @Co.: IMICAMS
 * @Date: 2022-06-20 01:36:45
 * @LastEditTime: 2023-08-08 16:18:18
 * Copyright (c) 2022 by ziyang-W (ziyangw@yeah.net), All Rights Reserved. 
-->
# 介绍

整个代码库主要存放本人在硕士期间编写的多种函数，目前仅主要包括以下几个方面：

1. 数据预处理
2. 统计学检验
3. 机器学习
4. 深度学习
5. 绘图
6. 办公自动化

其中，`dataPrep.py`是最常用到的一个模块，其中的`make_logInfo`函数会根据传入的数据路径和文件名，以及当前日期和时间，创建日志模块，方便保存后续数据分析和处理的结果。`make_logInfo`的返回结果`logInfo`会作为其他函数的参数传入，用于判断是否保存相关的数据。

代码会及时进行更新和完善，并且每个函数都附带了详细的注释。

# TODO：
- [ ] 介绍各个模块内函数的作用和调用逻辑
- [ ] 完善深度学习的模块
- [ ] 补充模块之间的摘要图

## Python 包提交流程

* **编写setup.py**

    1. 包名称中亿数字+字母为主，最多出现`-`，其他的字符都可能会导致安装失败的问题

```python
from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1',
    author='Your Name',
    author_email='your_email@example.com',
    description='My Python Package',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0'
    ]
)
```

* **添加亿点点细节**

    1. LICENSE.txt，可以从[license网站中下载](https://choosealicense.com/)

    1. 其他的细节，可以[点击查看CSDN笔记](https://blog.csdn.net/qq_27884799/article/details/96664812)


* **打包发布**

```SH 
python setup.py dist bdist_wheel
```

该命令将会生成`.tar.gz`和`.whl`两个格式的包文件，分别存放在`dist`目录下。

* **在[PyPI注册账号](https://pypi.org/)**

创建PyPI账号后，在`Account Settings`中按照要求设置`Recover codes`与2FA验证，iOS设备可以从APP Store中下载`Duo`软件进行扫码验证。TestPyPI与PyPI，可以仅注册PyPI即可，直接将包上传到PyPI中，但建议在上传前在本地进行测试。

设置了2FA验证后，提交包的密码不知道是哪个，一会儿要求原密码，一会儿要求2FA密码。很容易报错`Invalid or non-existent authentication information`，所以推荐设置API tokens的方式进行提交。

* **添加API tokens**

在`Account Settings`中，找到`Add API token`，按照要求在当前操作系统的用户目录下新建`.pypirc`文件。

Windows可以在cmd中运行`echo. > .pypirc`，然后在用记事本打开。

Linux 直接`vim ~/.pypirc`，将生成的Token复制进去，`:wq`结束

`.pypirc`样例如下：
```
[distutils]
index-servers=pypi

[pypi]
  username = __token__
  password = pypi-GENERATED_API_TOKEN
  
```





# REFERENCE

[Python极简打包流程](https://zhuanlan.zhihu.com/p/609180587)
