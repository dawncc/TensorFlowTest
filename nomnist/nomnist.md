

```python

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline
```

- 从指定地址下载训练数据集
- 数据集为28*28的图片
-  A到J 10种字母
- 500K的训练数据和19000的测试数据


```python
url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
```

    Attempting to download: notMNIST_large.tar.gz
    0%....5%....10%....15%....20%....25%....30%....35%....40%....45%....50%....55%....60%....65%....70%....75%....80%....85%....90%....95%....100%
    Download Complete!
    Found and verified .\notMNIST_large.tar.gz
    Attempting to download: notMNIST_small.tar.gz
    0%....5%....10%....15%....20%....25%....30%....35%....40%....45%....50%....55%....60%....65%....70%....75%....80%....85%....90%....95%....100%
    Download Complete!
    Found and verified .\notMNIST_small.tar.gz
    

- 从tar.gz中提取数据集


```python
num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
```

    .\notMNIST_large already present - Skipping extraction of .\notMNIST_large.tar.gz.
    ['.\\notMNIST_large\\A', '.\\notMNIST_large\\B', '.\\notMNIST_large\\C', '.\\notMNIST_large\\D', '.\\notMNIST_large\\E', '.\\notMNIST_large\\F', '.\\notMNIST_large\\G', '.\\notMNIST_large\\H', '.\\notMNIST_large\\I', '.\\notMNIST_large\\J']
    .\notMNIST_small already present - Skipping extraction of .\notMNIST_small.tar.gz.
    ['.\\notMNIST_small\\A', '.\\notMNIST_small\\B', '.\\notMNIST_small\\C', '.\\notMNIST_small\\D', '.\\notMNIST_small\\E', '.\\notMNIST_small\\F', '.\\notMNIST_small\\G', '.\\notMNIST_small\\H', '.\\notMNIST_small\\I', '.\\notMNIST_small\\J']
    

- 显示解压后的图片


```python
#Problem1: Display a sample of the images that we just download
nums_image_show = 2#显示的图像张数
for index_class in range(num_classes):
    #i from 0 to 9
    imagename_list = os.listdir(train_folders[index_class])
    imagename_list_indice = imagename_list[0:nums_image_show]
    for index_image in range(nums_image_show):
        path = train_folders[index_class] +'\\' + imagename_list_indice[index_image]
        display(Image(filename = path))
```


![png](output_6_0.png)



![png](output_6_1.png)



![png](output_6_2.png)



![png](output_6_3.png)



![png](output_6_4.png)



![png](output_6_5.png)



![png](output_6_6.png)



![png](output_6_7.png)



![png](output_6_8.png)



![png](output_6_9.png)



![png](output_6_10.png)



![png](output_6_11.png)



![png](output_6_12.png)



![png](output_6_13.png)



![png](output_6_14.png)



![png](output_6_15.png)



![png](output_6_16.png)



![png](output_6_17.png)



![png](output_6_18.png)



![png](output_6_19.png)


## 加载和归一化图像数据
- 将硬盘中图片转换为三维的向量(image index, x, y)，存储到dataset对象
- 将读取的图像数据做归一化处理，求和，平均值，方差：比较简单，分别是np.sum(), np.mean(), np.var(), np.std()
- 将ndarray对象打包为pickle格式并存储在工作目录下，每个类别有一个.pickle文件

注：将数据打包为.pickle文件更便于数据的调用与处理。因为，图像的原始数据是使用循环打入到对象中的，如果每次使用图像数据均需要循环来加载，这样加大了代码量。而对.pickle文件只需要读取一次即可，而无需使用循环。


```python
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)
```

    Pickling .\notMNIST_large\A.pickle.
    .\notMNIST_large\A
    Could not read: .\notMNIST_large\A\RnJlaWdodERpc3BCb29rSXRhbGljLnR0Zg==.png : cannot identify image file '.\\notMNIST_large\\A\\RnJlaWdodERpc3BCb29rSXRhbGljLnR0Zg==.png' - it's ok, skipping.
    Could not read: .\notMNIST_large\A\SG90IE11c3RhcmQgQlROIFBvc3Rlci50dGY=.png : cannot identify image file '.\\notMNIST_large\\A\\SG90IE11c3RhcmQgQlROIFBvc3Rlci50dGY=.png' - it's ok, skipping.
    Could not read: .\notMNIST_large\A\Um9tYW5hIEJvbGQucGZi.png : cannot identify image file '.\\notMNIST_large\\A\\Um9tYW5hIEJvbGQucGZi.png' - it's ok, skipping.
    Full dataset tensor: (52909, 28, 28)
    Mean: -0.12825
    Standard deviation: 0.443121
    Pickling .\notMNIST_large\B.pickle.
    .\notMNIST_large\B
    Could not read: .\notMNIST_large\B\TmlraXNFRi1TZW1pQm9sZEl0YWxpYy5vdGY=.png : cannot identify image file '.\\notMNIST_large\\B\\TmlraXNFRi1TZW1pQm9sZEl0YWxpYy5vdGY=.png' - it's ok, skipping.
    Full dataset tensor: (52911, 28, 28)
    Mean: -0.00756303
    Standard deviation: 0.454491
    Pickling .\notMNIST_large\C.pickle.
    .\notMNIST_large\C
    Full dataset tensor: (52912, 28, 28)
    Mean: -0.142258
    Standard deviation: 0.439806
    Pickling .\notMNIST_large\D.pickle.
    .\notMNIST_large\D
    Could not read: .\notMNIST_large\D\VHJhbnNpdCBCb2xkLnR0Zg==.png : cannot identify image file '.\\notMNIST_large\\D\\VHJhbnNpdCBCb2xkLnR0Zg==.png' - it's ok, skipping.
    Full dataset tensor: (52911, 28, 28)
    Mean: -0.0573678
    Standard deviation: 0.455648
    Pickling .\notMNIST_large\E.pickle.
    .\notMNIST_large\E
    Full dataset tensor: (52912, 28, 28)
    Mean: -0.069899
    Standard deviation: 0.452942
    Pickling .\notMNIST_large\F.pickle.
    .\notMNIST_large\F
    Full dataset tensor: (52912, 28, 28)
    Mean: -0.125583
    Standard deviation: 0.44709
    Pickling .\notMNIST_large\G.pickle.
    .\notMNIST_large\G
    Full dataset tensor: (52912, 28, 28)
    Mean: -0.0945814
    Standard deviation: 0.44624
    Pickling .\notMNIST_large\H.pickle.
    .\notMNIST_large\H
    Full dataset tensor: (52912, 28, 28)
    Mean: -0.0685221
    Standard deviation: 0.454232
    Pickling .\notMNIST_large\I.pickle.
    .\notMNIST_large\I
    Full dataset tensor: (52912, 28, 28)
    Mean: 0.0307862
    Standard deviation: 0.468899
    Pickling .\notMNIST_large\J.pickle.
    .\notMNIST_large\J
    Full dataset tensor: (52911, 28, 28)
    Mean: -0.153358
    Standard deviation: 0.443656
    Pickling .\notMNIST_small\A.pickle.
    .\notMNIST_small\A
    Could not read: .\notMNIST_small\A\RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png : cannot identify image file '.\\notMNIST_small\\A\\RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png' - it's ok, skipping.
    Full dataset tensor: (1872, 28, 28)
    Mean: -0.132626
    Standard deviation: 0.445128
    Pickling .\notMNIST_small\B.pickle.
    .\notMNIST_small\B
    Full dataset tensor: (1873, 28, 28)
    Mean: 0.00535609
    Standard deviation: 0.457115
    Pickling .\notMNIST_small\C.pickle.
    .\notMNIST_small\C
    Full dataset tensor: (1873, 28, 28)
    Mean: -0.141521
    Standard deviation: 0.44269
    Pickling .\notMNIST_small\D.pickle.
    .\notMNIST_small\D
    Full dataset tensor: (1873, 28, 28)
    Mean: -0.0492167
    Standard deviation: 0.459759
    Pickling .\notMNIST_small\E.pickle.
    .\notMNIST_small\E
    Full dataset tensor: (1873, 28, 28)
    Mean: -0.0599148
    Standard deviation: 0.45735
    Pickling .\notMNIST_small\F.pickle.
    .\notMNIST_small\F
    Could not read: .\notMNIST_small\F\Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png : cannot identify image file '.\\notMNIST_small\\F\\Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png' - it's ok, skipping.
    Full dataset tensor: (1872, 28, 28)
    Mean: -0.118185
    Standard deviation: 0.452279
    Pickling .\notMNIST_small\G.pickle.
    .\notMNIST_small\G
    Full dataset tensor: (1872, 28, 28)
    Mean: -0.0925503
    Standard deviation: 0.449006
    Pickling .\notMNIST_small\H.pickle.
    .\notMNIST_small\H
    Full dataset tensor: (1872, 28, 28)
    Mean: -0.0586893
    Standard deviation: 0.458759
    Pickling .\notMNIST_small\I.pickle.
    .\notMNIST_small\I
    Full dataset tensor: (1872, 28, 28)
    Mean: 0.0526451
    Standard deviation: 0.471894
    Pickling .\notMNIST_small\J.pickle.
    .\notMNIST_small\J
    Full dataset tensor: (1872, 28, 28)
    Mean: -0.151689
    Standard deviation: 0.448014
    

## 显示从pickle文件中读取的图像


```python
#Problem2 Displaying a sample of the labels and images from the ndarray

# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline
import matplotlib.pyplot as plt
def load_and_displayImage_from_pickle(data_filename_set,NumClass,NumImage):
    if(NumImage <= 0):
        print('NumImage <= 0')
        return
    plt.figure('subplot')
    for index,pickle_file in enumerate(data_filename_set):
        with open(pickle_file,'rb') as f:
            data = pickle.load(f)
            ImageList = data[0:NumImage,:,:]
            for i,Image in enumerate(ImageList):
                #NumClass代表类别，每个类别一行;NumImage代表每个类显示的图像张数
                plt.subplot(NumClass, NumImage, index*NumImage+i+1)
                plt.imshow(Image)
            index = index+1        
#显示10类，每类显示5张图片        
load_and_displayImage_from_pickle(train_datasets,10,5)    
load_and_displayImage_from_pickle(test_datasets,10,5) 
```


![png](output_10_0.png)


## 问题3-检测数据是否平衡

数据是否平衡的意思是各类样本的大小是否相当。


```python
def show_sum_of_different_class(data_filename_set):
    plt.figure(1)
    #read .pickle file
    sumofdifferentclass = []
    for pickle_file in data_filename_set:
        with open(pickle_file,'rb') as f:
            data = pickle.load(f)
            print(len(data))
            sumofdifferentclass.append(len(data))

    #show the data
    x = range(10)
    plt.bar(x,sumofdifferentclass)    
    plt.show()

print('train_datasets:\n')    
show_sum_of_different_class(train_datasets)  
print('test_datasets:\n')    
show_sum_of_different_class(test_datasets) 
```

    train_datasets:
    
    52909
    52911
    52912
    52911
    52912
    52912
    52912
    52912
    52912
    52911
    


![png](output_12_1.png)


    test_datasets:
    
    1872
    1873
    1873
    1873
    1873
    1872
    1872
    1872
    1872
    1872
    


![png](output_12_3.png)


## 将不同类别的数据混合并将得到验证集
- 将不同类别的数据进行混合。之前是每个类别一个数据对象。现在，为了便于后续的训练，需将不同类别的数据存储为一个大的数据对象，即该对象同时包含A、B…J共个类别的样本。
- 从训练集中提取一部分作为验证集。


```python
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
```

    Training: (200000, 28, 28) (200000,)
    Validation: (10000, 28, 28) (10000,)
    Testing: (10000, 28, 28) (10000,)
    

## 将混合后的数据进行随机化
上一步只是将数据进行和混合并存储为一个大的数据对象，此步则将混合后的数据对象中的数据进行了随机化处理。只有随机化后的数据训练模型时才会有较为稳定的效果。


```python
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
```

## 问题4 从验证混合后的数据


```python
"""LProblem4 Convince yourself that the data is still good after shuffling!
"""
#data_set是数据集，NumImage是显示的图像张数
def displayImage_from_dataset(data_set,NumImage):
    if(NumImage <= 0):
        print('NumImage <= 0')
        return
    plt.figure('subplot')
    ImageList = data_set[0:NumImage,:,:]
    for index,Image in enumerate(ImageList):
        #NumClass代表类别，每个类别一行;NumImage代表每个类显示的图像张数
        plt.subplot(NumImage//5+1, 5, index+1)
        plt.imshow(Image)
        index = index+1    
    plt.show()
displayImage_from_dataset(train_dataset,50)   
```


![png](output_18_0.png)


## 将不同的样本及存为.pickle文件


```python
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
```


```python
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
```

    Compressed pickle size: 690800506
    

## 问题5-数据清洗
一般来说，训练集、验证集和测试集中会有数据的重合，但是，如果重合的数据太多则会影响到测试结果的准确程度。因此，需要对数据进行清洗，使彼此之间步存在交集。

注：ndarray数据无法使用set的方式来求取交集。但如果使用循环对比的方式在数据量大的情况下会非常慢，因此，下文的做法使先将数据哈希化，再通过哈希的键值来判断数据是否相等。由于哈希的键值是字符串，因此比对起来效率会高很多。


```python
#先使用hash
import hashlib

#使用sha的作用是将二维数据和哈希值之间进行一一对应，这样，通过比较哈希值就能将二维数组是否相等比较出来
def extract_overlap_hash_where(dataset_1,dataset_2):

    dataset_hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_1])
    dataset_hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_2])
    overlap = {}
    for i, hash1 in enumerate(dataset_hash_1):
        duplicates = np.where(dataset_hash_2 == hash1)
        if len(duplicates[0]):
            overlap[i] = duplicates[0]
    return overlap

#display the overlap
def display_overlap(overlap,source_dataset,target_dataset):
    overlap = {k: v for k,v in overlap.items() if len(v) >= 3}
    item = np.random.choice(list(overlap.keys()))
    imgs = np.concatenate(([source_dataset[item]],target_dataset[overlap[item][0:7]]))
    plt.suptitle(item)
    for i,img in enumerate(imgs):
        plt.subplot(2,4,i+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()

#数据清洗
def sanitize(dataset_1,dataset_2,labels_1):
    dataset_hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_1])
    dataset_hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in dataset_2])
    overlap = []
    for i,hash1 in enumerate(dataset_hash_1):
        duplictes = np.where(dataset_hash_2 == hash1)
        if len(duplictes[0]):
            overlap.append(i)
    return np.delete(dataset_1,overlap,0),np.delete(labels_1, overlap, None)


overlap_test_train = extract_overlap_hash_where(test_dataset,train_dataset)
print('Number of overlaps:', len(overlap_test_train.keys()))
display_overlap(overlap_test_train, test_dataset, train_dataset)

test_dataset_sanit,test_labels_sanit = sanitize(test_dataset,train_dataset,test_labels)
print('Overlapping images removed from test_dataset: ', len(test_dataset) - len(test_dataset_sanit))

valid_dataset_sanit, valid_labels_sanit = sanitize(valid_dataset, train_dataset, valid_labels)
print('Overlapping images removed from valid_dataset: ', len(valid_dataset) - len(valid_dataset_sanit))

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_labels_sanit.shape, valid_labels_sanit.shape)
print('Testing:', test_dataset_sanit.shape, test_labels_sanit.shape)

pickle_file_sanit = 'notMNIST_sanit.pickle'
try:
    f = open(pickle_file_sanit,'wb')
    save = {
        'train_dataset':train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file_sanit)
print('Compressed pickle size:', statinfo.st_size)
```

    Number of overlaps: 1324
    


![png](output_23_1.png)


    Overlapping images removed from test_dataset:  1324
    Overlapping images removed from valid_dataset:  1067
    Training: (200000, 28, 28) (200000,)
    Validation: (8933,) (8933,)
    Testing: (8676, 28, 28) (8676,)
    Unable to save data to .\notMNIST.pickle : [Errno 28] No space left on device
    


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-24-bf6fbb596a23> in <module>()
         63         'test_labels': test_labels,
         64     }
    ---> 65     pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
         66     f.close()
         67 except Exception as e:
    

    OSError: [Errno 28] No space left on device


## 问题6-模型训练
该模型是使用逻辑回归模型进行的训练。


```python
def train_and_predict(sample_size):
    regr = LogisticRegression()
    X_train = train_dataset[:sample_size].reshape(sample_size,784)
    y_train = train_labels[:sample_size]
    regr.fit(X_train,y_train)
    X_test = test_dataset.reshape(test_dataset.shape[0],28*28)
    y_test = test_labels

    pred_labels = regr.predict(X_test)
    print('Accuracy:', regr.score(X_test, y_test), 'when sample_size=', sample_size)

for sample_size in [50,100,1000,5000,len(train_dataset)]:
    train_and_predict(sample_size)
```

http://www.hankcs.com/ml/notmnist.html


```python

```
