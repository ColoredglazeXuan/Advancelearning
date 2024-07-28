from datetime import datetime

# 获取当前系统时间
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print(current_time)
_exp_name = f"sample{current_time}"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random

# 6666\1314\520\438
myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data *wrapping* as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.25),  # 以 20% 的概率将图像转换为灰度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    transforms.RandomRotation(degrees=(-90, 90)),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),

])


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, path2=None, files=None, mode='train'):
        super(FoodDataset).__init__()
        self.path = path
        self.mode = mode
        self.files = (sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])+
                      sorted([os.path.join(path2, x) for x in os.listdir(path2) if x.endswith(".jpg")]))
        if files != None:
            self.files = files

        self.transform = tfm

        self.valid_tfm = transforms.Compose([
            # Resize the image into a fixed shape (height = width = 128)
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        if self.mode != 'valid':
            im = self.transform(im)
        else:
            im = self.valid_tfm(im)
        try:
            label = int(fname.split("\\")[-1].split("_")[0])
        except:
            label = -1  # test has no label

        return im, label


"""# Model"""
import torchvision.models as models


# class Classifier(nn.Module):
#     def __init__(self, num_classes=11):
#         super(Classifier, self).__init__()
#         # 加载预定义的 VGG-13 模型，未使用预训练权重
#         self.vgg = models.vgg13(pretrained=False)
#
#         # 获取 VGG-13 最后一个全连接层的输入特征数量
#         num_ftrs = self.vgg.classifier[-1].in_features
#
#         # 重新定义 VGG-13 的分类器部分以匹配目标类别数
#         # 直接在最后一个全连接层输出目标类别数
#         self.vgg.classifier = nn.Sequential(
#             nn.Linear(25088, num_classes)
#         )
#
#     def forward(self, x):
#         # 使用修改后的 VGG-13 模型处理输入
#         x = self.vgg(x)
#         return x


# class Classifier(nn.Module):
#     def __init__(self, num_classes=11):
#         super(Classifier, self).__init__()
#         # 加载预定义的resnet50模型
#         self.resnet = models.resnet18(pretrained=False)
#
#         # 获取resnet最后一个全连接层的输入特征数量
#         num_ftrs = self.resnet.fc.out_features
#
#         # 替换resnet的全连接层以匹配目标类别数，同时增加更多的线性层
#         self.classifier = nn.Sequential(
#             nn.Linear(num_ftrs, 1000),  # 第一个线性层
#             nn.LeakyReLU(0.01),  # 激活函数
#             nn.Dropout(0.5),  # Dropout层，防止过拟合
#             nn.Linear(1000, 1000),  # 第二个线性层
#             nn.LeakyReLU(0.01),  # 激活函数
#             nn.Dropout(0.5),  # Dropout层，防止过拟合
#             nn.Linear(1000, num_classes)  # 最后一个线性层，输出类别数
#         )
#
#     def forward(self, x):
#         # 使用修改后的resnet模型处理输入
#         x = self.resnet(x)
#
#         # 将resnet的输出通过额外的线性层
#         x = self.classifier(x)
#         return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),  # BatchNorm for 1D data
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),  # BatchNorm for 1D data
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),  # BatchNorm for 1D data
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""
if __name__ == '__main__':
    """# Configurations"""
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    model = Classifier().to(device)
    checkpoint = torch.load('ensemble/sample2024-02-29_21-37-05v2_best.ckpt')
    model.load_state_dict(checkpoint)
    print(model)
    # The number of batch size.
    batch_size = 64

    # The number of training epochs.
    n_epochs = 5

    # If no improvement in 'patience' epochs, early stop.
    patience = 70

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=1e-5, momentum=0,
    #                                 centered=False)
    # from torch.optim.lr_scheduler import CosineAnnealingLR
    #
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00005)
    """# Dataloader"""

    # Construct train and valid datasets.
    # The argument "loader" tells how torchvision reads the data.
    train_set = FoodDataset("./train", tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True,
                              persistent_workers=True)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_set = FoodDataset("./valid", tfm=test_tfm, mode='valid')
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                              persistent_workers=True)
    # valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    """# Start Training"""

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0
    train_acc_list = []
    valid_acc_list = []

    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()
            # print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # current_lr = scheduler.get_last_lr()[0]
            # print(f"Epoch {epoch + 1}, Current learning rate: {current_lr}")
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
        # scheduler.step()
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        train_acc_list.append(train_acc)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            # break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        valid_acc_list.append(valid_acc)
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt", "a"):
                print(
                    f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt", "a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(),
                       f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break

    """# Dataloader for test"""

    # Construct test datasets.
    # The argument "loader" tells how torchvision reads the data.
    test_set = FoodDataset("./test", tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    """# Testing and generate prediction CSV"""

    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    import matplotlib.pyplot as plt


    # create test csv
    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)


    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(test_set))]
    df["Category"] = prediction
    df.to_csv(f"submission_{current_time}.csv", index=False)

    train_acc_list = [item.cpu().item() for item in train_acc_list]  # 绘制训练和验证损失
    valid_acc_list = [item.cpu().item() for item in valid_acc_list]  # 绘制训练和验证损失

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_list, label='Training Loss')
    plt.plot(valid_acc_list, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
