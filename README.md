# TensorFlowTest

## 使用Pin安装Python
```
# 仅使用 CPU 的版本
$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# 开启 GPU 支持的版本 (安装该版本的前提是已经安装了 CUDA sdk)
$ pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```
## 运行 TensorFlow

打开一个 python 终端:
```python
$ python

>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a+b))
42
>>>
```
## 克隆 TensorFlow 仓库
```
$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
--recurse-submodules 参数是必须得, 用于获取 TesorFlow 依赖的 protobuf 库.
```
# Python环境搭建
下载地址： http://www.python.org/download/

## 新建main.py文件
```python
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
**执行**
```python
python main.py
```
**自动下载数据训练文件**
```
|--__pycache__
    |--input_data.cpython-36
```
## 查看版本和路径
```python
root@:~/user/jupyter# python
Python 3.6.0 (default, May 16 2017, 22:05:15) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.__path__
['/usr/local/python36/lib/python3.6/site-packages/tensorflow']
>>> exit()
root@:~/user/jupyter# 

```
