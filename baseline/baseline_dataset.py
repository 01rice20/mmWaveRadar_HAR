from random import shuffle
import glob
import numpy as np
import h5py
import cv2

shuffle_data = True

hdf5_path = '../../dataset/baseline/all.hdf5'

train_path = '../../dataset/spectrogram/spectrogram_111/*/*.png' # the original data path
print(train_path)

addrs = glob.glob(train_path)
print("addrs shape: ", len(addrs))

labels = []
for i in addrs:
    if '05_Walking_Towards_radar' in i:
        labels.append(0)
    elif '06_walking_away_from_Radar' in i:
        labels.append(1)
    elif '07_Picking_up_an_object' in i:
        labels.append(2)
    elif '08_Bending' in i:
        labels.append(3)
    elif '09_Sitting' in i:
        labels.append(4)
    elif '10_Kneeling' in i:
        labels.append(5)
    elif '11_Crawling' in i:
        labels.append(6)
    elif '16_Walking_on_both_toes' in i:
        labels.append(7)
    elif '17_limping_with_RL_Stiff' in i:
        labels.append(8)
    elif '18_Short_steps' in i:
        labels.append(9)
    elif '19_Scissors_gait' in i:
        labels.append(10)

print("labels shape: ", len(labels))

# shuffle data
if shuffle_data:
    c = list(zip(addrs, labels)) # use zip() to bind the images and labels together
    shuffle(c)
 
    (addrs, labels) = zip(*c)  # *c is used to separate all the tuples in the list c,  
                               # "addrs" then contains all the shuffled paths and 
                               # "labels" contains all the shuffled labels.
                               
# Divide the data into 80% for train and 20% for test
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

print("train_addrs: ", len(train_addrs))
print("test_addrs: ", len(test_addrs))
print("train_labels: ", len(train_labels))
print("test_labels: ", len(test_labels))

im_width = 128
im_height = 128
train_shape = (len(train_addrs), im_height, im_width, 3)
test_shape = (len(test_addrs), im_height, im_width, 3)

# open a hdf5 file and create earrays 
f = h5py.File(hdf5_path, mode='w')
f.create_dataset("train_img", train_shape, np.uint8)
f.create_dataset("test_img", test_shape, np.uint8)  
f.create_dataset("train_labels", (len(train_addrs),), np.uint8)
f["train_labels"][...] = train_labels
f.create_dataset("test_labels", (len(test_addrs),), np.uint8)
f["test_labels"][...] = test_labels

# loop over train paths
for i in range(len(train_addrs)):
  
    if i % 50 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)) )

    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC)# resize to (128,128)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 load images as BGR, convert it to RGB
    f["train_img"][i, ...] = img[None] 
    
# loop over test paths
for i in range(len(test_addrs)):

    if i % 50 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_addrs)) )
        
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f["test_img"][i, ...] = img[None]

f.close()

dataset = h5py.File(hdf5_path, "r")

test_labels = np.array(dataset["test_labels"])
train_labels = np.array(dataset["train_labels"])
train_img = np.array(dataset["train_img"])
test_img = np.array(dataset["test_img"])
dataset.close()
print(len(train_labels))
print(len(test_labels))
