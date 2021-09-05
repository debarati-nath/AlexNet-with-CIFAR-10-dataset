#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from sklearn.metrics import confusion_matrix


#Load dataset
def load_cifar10_data(filename):
with open('drive/My Drive/cifar-10-batches-py/'+ filename, 'rb') as file:
batch = pickle.load(file, encoding='latin1')
features = batch['data']
labels = batch['labels']
return features, labels
batch_1, labels_1 = load_cifar10_data('data_batch_1')
batch_2, labels_2 = load_cifar10_data('data_batch_2')
batch_3, labels_3 = load_cifar10_data('data_batch_3')
batch_4, labels_4 = load_cifar10_data('data_batch_4')
batch_5, labels_5 = load_cifar10_data('data_batch_5')
test, label_test = load_cifar10_data('test_batch')
X_train = np.concatenate([batch_1,batch_2,batch_3,batch_4,batch_5], 0)
Y_train = np.concatenate([labels_1,labels_2,labels_3,labels_4,labels_5], 0)
classes = ('airplane', ' car', ' bird', ' cat',' deer', ' dog', ' frog', ' horse', ' ship', ' truck')

def return_photo(batch_file):
assert batch_file.shape[1] == 3072
dim = np.sqrt(1024).astype(int)
r = batch_file[:, 0:1024].reshape(batch_file.shape[0], dim, dim, 1)
g = batch_file[:, 1024:2048].reshape(batch_file.shape[0], dim, dim, 1)
b = batch_file[:, 2048:3072].reshape(batch_file.shape[0], dim, dim, 1)
photo = np.concatenate([r,g,b], -1)
return photo
X_train = return_photo(X_train)
X_test = return_photo(test)
Y_test = np.array(label_test)
def plot_image(number, file, label, pred=None):
   fig = plt.figure(figsize = (3,2))
   #img = return_photo(batch_file)
   plt.imshow(file[number])
   if pred is None:
       plt.title(classes[label[number]])
   else:
       plt.title('Label_true: ' + classes[label[number]] + ' \nLabel_pred: ' + classes[pred[number]])
       plot_image(12325, X_train, Y_train)
       print('X_train shape:', X_train.shape)
       print('Y_train shape:', Y_train.shape)
       print('X_test shape:', X_test.shape)
       print('Y_test shape:', Y_test.shape)

#Split the dataset
X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# define the random seed for reproducible result
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
class CIFAR10_from_array(Dataset):
   def _init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.img_shape = data.shape
   def _getitem__(self, index):
        img = Image.fromarray(self.data[index])
        label = self.label[index]
        if self.transform is not None:
                img = self.transform(img)
         else:
                img_to_tensor = transforms.ToTensor()
                img = img_to_tensor(img)
                    #label = torch.from_numpy(label).long()
         return img, label
   def _len__(self):
          return len(self.data)
   def plot_image(self, number):
          file = self.data
          label = self.label
          fig = plt.figure(figsize = (3,2))
          #img = return_photo(batch_file)
          plt.imshow(file[number])
          plt.title(classes[label[number]])
class CIFAR10_from_url(Dataset):
   pass
#Normalization
def normalize_dataset(data):
   mean = data.mean(axis=(0,1,2)) / 255.0
   std = data.std(axis=(0,1,2)) / 255.0
   normalize = transforms.Normalize(mean=mean, std=std)
   return normalize

train_transform_aug = transforms.Compose([
transforms.Resize((227,227)),
transforms.RandomCrop((227, 227)),
transforms.RandomVerticalFlip(),
#transforms.RandomRotation(15),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([
transforms.Resize((227,227)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
transforms.Resize((227,227)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

trainset = CIFAR10_from_array(data=X_train_split, label=Y_train_split, transform=train_transform_aug)
valset = CIFAR10_from_array(data=X_val_split, label=Y_val_split, transform=train_transform_aug)
testset = CIFAR10_from_array(data=X_test, label=Y_test, transform=test_transform)

print('data shape check')
print('training set:'.ljust(20) + ' {}'.format(trainset.img_shape))
print('validation set:'.ljust(20) + ' {}'.format(valset.img_shape))
print('testing set:'.ljust(20) + ' {}'.format(testset.img_shape))
print('label numbers:'.ljust(20) + ' {}'.format(len(set(trainset.label))))

#Load data
batch_size = 64
num_workers = 1
train_loader = DataLoader(dataset=trainset,
batch_size=batch_size,
shuffle=True,
num_workers=num_workers)
val_loader = DataLoader(dataset=valset,
batch_size=batch_size,
shuffle=False,
num_workers=num_workers)
test_loader = DataLoader(dataset=testset,
batch_size=batch_size,
shuffle=False,
num_workers=num_workers)

imgs, lbls = iter(train_loader).next()
print ('Size of image:', imgs.size())
print ('Type of image:', imgs.dtype)
print ('Size of label:', lbls.size())
print ('Type of label:', lbls.dtype)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
class AlexNet(nn.Module):
def _init__(self):
super(AlexNet, self).__init__()
self.features=nn.Sequential(
nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2),
nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2),
nn.Conv2d(256,384, kernel_size=3, stride=1, padding=1),
nn.ReLU(inplace=True),
#nn.Dropout(p=0.2),
nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
nn.ReLU(inplace=True),
#nn.Dropout(p=0.2),
nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2))
self.fc_layers=nn.Sequential(
nn.Dropout(p=0.2),
nn.Linear(256*6*6,4096),
nn.ReLU(inplace=True),
nn.Dropout(p=0.2),
nn.Linear(4096,4096),
nn.ReLU(inplace=True),
nn.Linear(4096,10))
def forward(self, x):
conv_features=self.features(x)
flatten=conv_features.view(conv_features.size(0),256*6*6)
fc=self.fc_layers(flatten)
return fc

def update_info(idx, length, epoch_loss, acc, mode):
if length >= 250:
update_size = int(length/250)
else:
update_size =0
if idx % update_size == 0 and idx != 0:
finish_rate = idx/length * 100
print ("\r {} progress: {:.2f}% . ..... loss: {:.4f} , acc: {:.4f}".
format(mode, finish_rate, epoch_loss/idx, acc), end="", flush=True)
def val_per_epoch(model, loss_fn, dataloader, verbose):
model.eval()
epoch_loss = 0.0
acc = 0.0
val_size = 0
with torch.no_grad():
for i, (feature, target) in enumerate(dataload):
if torch.cuda.is_available():
feature = feature.cuda()
target = target.cuda()
output = model(feature)
, pred = torch.max(output.data, dim=1)
correct = (pred == target).sum().item() #convert to number
val_size += target.size(0)
acc += correct
loss = loss_fn(output, target)
epoch_loss += loss.item()
idx = i
length = len(dataloader)
if verbose:
update_info(idx, length, epoch_loss, acc/val_size, ' validating')
acc = acc/val_size
print('')
return epoch_loss/len(dataloader), acc

def train_per_epoch(model, loss_fn, dataloader, optimizer, verbose):
model.train()
epoch_loss = 0.0
acc = 0.0
train_size = 0
for i, (feature, target) in enumerate(dataloader):
if torch.cuda.is_available():
feature = feature.cuda()
target = target.cuda()
optimizer.zero_grad()
output = model(feature)
loss = loss_fn(output, target)
, pred = torch.max(output.data, dim=1)
correct = (pred == target).sum().item()
train_size += target.size(0)
acc += correct
epoch_loss += loss.item()
loss.backward()
optimizer.step()
idx = i
length = len(dataloader)
if verbose:
 update_info(idx, length, epoch_loss, acc/train_size, ' training')
acc = acc/train_size
print('')
return epoch_loss/len(dataloader), acc
def model_training(num_epochs, model, loss_fn, train_loader, optimizer, val_loader=None,
verbose=True)
train_batch_num = len(train_loader)
history = {}
history['train_loss'] = []
history['val_loss'] = []
history['train_acc'] = []

history['val_acc'] = []
if val_loader is not None:
val_batch_num = len(val_loader)
print('Total Sample: Train on {} samples, validate on {} samples.'.
format(trainset.img_shape[0], valset.img_shape[0]))
print(' Total Batch: Train on {} batches, validate on {} batches. {} samples/minibatch \n'.
format(train_batch_num, val_batch_num, batch_size))
else:
print('Total Sample: Train on {} samples.'.
format(train_batch_num*batch_size)
print(' Total Batch: Train on {} batches, {} samples/minibatch \n'.
format(train_batch_num, batch_size))
for epoch in range(num_epochs):
print('Epoch {}/{}'.format(epoch+1, num_epochs))
train_loss, train_acc = train_per_epoch(model, loss_fn, train_loader, optimizer, verbose=verbose)
history['train_loss'].append(train_loss)
history['train_acc'].append(train_acc)
if val_loader is not None:
val_loss, val_acc = val_per_epoch(model, loss_fn, val_loader, verbose=verbose)
print('\n Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(train_loss,val_loss))
print(' Training acc: {:.4f}, Validation acc: {:.4f}\n'.format(train_acc,val_acc))
history['val_loss'].append(val_loss)
history['val_acc'].append(val_acc)
else:
print('\n Training Loss: {:.4f}\n'.format(train_loss))
print('\n Training acc: {:.4f}\n'.format(train_acc))
return history
classes = ('airplane', ' car', ' bird', ' cat',' deer', ' dog', ' frog', ' horse', ' ship', ' truck')

if _name__ == ' __main__':
num_epochs = 100
learning_rate = 0.001
net = AlexNet()
if torch.cuda.is_available():
net = net.cuda()
print('=================================================================')
criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-5)
hist1 = model_training(num_epochs, net, criterion, train_loader, optimizer, val_loader, verbose=True)
def imshow(img):
img = img
npimg = img.numpy()
print(np.transpose(npimg, (1, 2, 0)).shape)
plt.imshow(np.transpose(npimg, (1, 2, 0)))

plt.show()
if _name__ == ' __main__':
dataiter = iter(test_loader)
images, labels = dataiter.next()
for i in range(len(images)):
plot_image(i, images.permute(0, 2, 3, 1).numpy(), labels.numpy())
if torch.cuda.is_available():
images = images.cuda()
outputs = net(images)
, predicted = torch.max(outputs, 1)
print('Predicted: ' , ' ' .join('%5s' % classes[predicted[j]]
for j in range(5)))
def model_testing(model, loss_fn, dataloader, verbose=True):
Y_pred = []
correct = 0
total = 0
epoch_loss = 0.0
acc = 0.0
test_size = 0
with torch.no_grad():
for i, (feature, target) in enumerate(dataloader):
if torch.cuda.is_available():
feature = feature.cuda()
target = target.cuda()
outputs = model(feature) #outputs.data.shape= batches_num * num_class
, pred = torch.max(outputs.data, 1)
correct = (pred == target).sum().item() #convert to number
test_size += target.size(0)
acc += correct
loss = loss_fn(outputs, target)
epoch_loss += loss.item()
idx = i
length = len(dataloader)
Y_pred += pred.cpu().numpy().tolist()
if verbose:
update_info(idx, length, epoch_loss, acc/test_size, ' testing')
acc = acc/test_size
print('\n\n Accuracy of the network on the {} test images: {}%'.format(test_size, 100*acc))
return Y_pred

if _name__ == ' __main__':
Y_pred1 = model_testing(net, criterion, test_loader, True)
def loss_acc_plt(history):
fig, ax = plt.subplots(2,1)
ax[0].plot(history['train_loss'], color='b', label="Training loss")
ax[0].plot(history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
ax[1].plot(history['train_acc'], color='b', label="Training accuracy")
ax[1].plot(history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
if _name__ == ' __main__':
    loss_acc_plt(hist1)
if _name__ == ' __main__':
    for i in range(10):
        plot_image(i, test_loader.dataset.data, test_loader.dataset.label, Y_pred1)

if _name__ == ' __main__':
      cm = confusion_matrix(Y_test, Y_pred1)
plt.figure(figsize = (10,8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=14)
plt.colorbar()
n_classes = cm.shape[0]
range_class = range(n_classes)
tick_marks = np.arange(len(range_class))
plt.xticks(tick_marks, range_class, rotation=-45, fontsize=14)
plt.yticks(tick_marks, range_class, fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)
for i in range_class:
    for j in range_class:
       plt.text(j, i, cm[i,j], horizontalalignment="center", fontsize=14,color="white" if i==j else "black")
plt.plot
for i in range(len(classes)):
    correct = ((Y_test == i)*1) * ((np.array(Y_pred1) == Y_test)*1)
    print('{}, {}: ' .rjust(10).format(i, classes[i]) + ' {}%'.format(100*correct.sum()/Y_test[Y_test == i].shape[0]))
for name, param in net.named_parameters():
    print('name',name)
    print(type(param))
    print('param.shape:', param.shape)
    print('param.requires_grad:', param.requires_grad)
    print('=====')
torch.save(hist1,'entire_model.pth')
print(net.features)
torch.save(net.features[0].weight.data,'weight5.pth')
a=net.features[0].weight.data.cpu().numpy()
print(a.shape)
a_min, a_max=b.min(), b.max()
a1=(b-a_min)/(a_min-a_max)
n_filters, ix= 3,1
for i in range(n_filters):
   f=a1[:,:,:,i]
   for j in range(3):
     ax=plt.subplot(n_filters, 3, ix)
     ax.set_xticks([])
     ax.set_yticks([])
     plt.imshow(f[:,:,j], cmap='gray')
     ix+=1
plt.show()
