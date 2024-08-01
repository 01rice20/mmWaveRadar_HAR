from random import shuffle
import glob

shuffle_data = True  # shuffle the addresses

hdf5_path = 'real_data_77.hdf5'  # file path for the created .hdf5 file

train_path = '/home/dataset/real_data/77GHz/*/*.png' # the original data path
print(train_path)

# get all the image paths 
addrs = glob.glob(train_path)
print(len(addrs))

labels = []
for i in addrs:
    if 'Towards' in i:
        labels.append(0)
    elif 'Away' in i:
        labels.append(1)
    elif 'Pick' in i:
        labels.append(2)
    elif 'Bend' in i:
        labels.append(3)
    elif 'Sit' in i:
        labels.append(4)
    elif 'Kneel' in i:
        labels.append(5)
    elif 'Crawl' in i:
        labels.append(6)
    elif 'Toes' in i:
        labels.append(7)
    elif 'Limp' in i:
        labels.append(8)
    elif 'Short' in i:
        labels.append(9)
    elif 'Scissor' in i:
        labels.append(10)
   
"""
# label the data as 0=cat, 1=dog

labels = [0 if 'towards' in else 1 for addr in addrs] # extra is included in non-vehicles name 
          
    
"""

# shuffle data
if shuffle_data:
    c = list(zip(addrs, labels)) # use zip() to bind the images and labels together
    shuffle(c)
 
    (addrs, labels) = zip(*c)  # *c is used to separate all the tuples in the list c,  
                               # "addrs" then contains all the shuffled paths and 
                               # "labels" contains all the shuffled labels.
                               
# Divide the data into 80% for train and 20% for test
train_addrs = addrs[0:int(.8*len(addrs))]
train_labels = labels[0:int(.8*len(labels))]

valid_addrs = addrs[int(0.7*len(addrs)):int(1.0*len(addrs))]
valid_labels = labels[int(0.7*len(labels)):int(1.0*len(addrs))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

#test_addrs = addrs[0:int(1.0*len(addrs))]
#test_labels = labels[0:int(1.0*len(labels))]
print(len(train_addrs))
print(len(valid_addrs))
print(len(test_addrs))
print(len(train_labels))
print(len(valid_labels))
print(len(test_labels))
##################### second part: create the h5py object #####################
import numpy as np
import h5py

train_shape = (len(train_addrs), 128, 128,3)
valid_shape = (len(valid_addrs), 128, 128,3)
test_shape = (len(test_addrs), 128, 128,3)
labels_order=np.array([9,19,4,14,12,8,15,0,16,2,1,7,18,6,10,11,17,3,13,5]) 
# open a hdf5 file and create earrays 
f = h5py.File(hdf5_path, mode='w')

# PIL.Image: the pixels range is 0-255,dtype is uint.
# matplotlib: the pixels range is 0-1,dtype is float.
f.create_dataset("train_img", train_shape, np.uint8)
f.create_dataset("valid_img", valid_shape, np.uint8)  
f.create_dataset("test_img", test_shape, np.uint8)  

# the ".create_dataset" object is like a dictionary, the "train_labels" is the key. 
f.create_dataset("train_labels", (len(train_addrs),), np.uint8)
f["train_labels"][...] = train_labels

f.create_dataset("valid_labels", (len(valid_addrs),), np.uint8)
f["valid_labels"][...] = valid_labels

f.create_dataset("test_labels", (len(test_addrs),), np.uint8)
f["test_labels"][...] = test_labels

f.create_dataset("labels_order", (len(labels_order),), np.uint8)
f["labels_order"][...] = labels_order

######################## third part: write the images #########################
import cv2

# loop over train paths
for i in range(len(train_addrs)):
  
    if i % 50 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)) )

    addr = train_addrs[i]
    #img = cv2.imread(addr,cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(addr)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)# resize to (128,128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 load images as BGR, convert it to RGB
    f["train_img"][i, ...] = img[None] 

    # loop over valid paths
for i in range(len(valid_addrs)):

    if i % 50 == 0 and i > 1:
        print ('Valid data: {}/{}'.format(i, len(valid_addrs)) )
        
    addr = valid_addrs[i]
    #img = cv2.imread(addr,cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(addr)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f["valid_img"][i, ...] = img[None]
    
# loop over test paths
for i in range(len(test_addrs)):

    if i % 50 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_addrs)) )
        
    addr = test_addrs[i]
    #img = cv2.imread(addr,cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(addr)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f["test_img"][i, ...] = img[None]

f.close()

dataset = h5py.File(hdf5_path, "r")
import matplotlib.pyplot as plt
test_labels = np.array(dataset["test_labels"])
valid_labels = np.array(dataset["valid_labels"])
train_labels = np.array(dataset["train_labels"])
train_img = np.array(dataset["train_img"])
valid_img = np.array(dataset["valid_img"])
test_img = np.array(dataset["test_img"])
labels_order=np.array(dataset["labels_order"])
dataset.close()
print(len(train_labels))
print(len(valid_labels))
print(len(test_img))
print(train_labels[1])

# plt.imshow(train_img[100])
# plt.show()