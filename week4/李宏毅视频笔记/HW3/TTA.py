"""# Dataloader for test"""
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from tqdm import tqdm
import numpy as np
import os
from train import FoodDataset
from datetime import datetime
import torchvision.transforms as transforms
from scipy.stats import mode
from PIL import Image
import torch.nn as nn
import pandas as pd

your_model_path='ensemble/sample2024-02-29_23-09-06_best.ckpt'
# 你的扩增倍数
k = 100

aug_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.25),  # 以 20% 的概率将图像转换为灰度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    transforms.RandomRotation(degrees=(-45, 45)),
    transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.9, 1.1)),
    transforms.RandomVerticalFlip(),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),

])


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

if __name__ == '__main__':
    # 获取当前系统时间当作文件名
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construct test datasets.
    # The argument "loader" tells how torchvision reads the data.
    batch_size = 64

    # 定义一个增强变换列表
    augmentations = [
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
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
        ])
    ]


    # 你可以添加更多的增强技术




    """# Testing and generate prediction CSV"""
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_set = FoodDataset("./test", mode='valid')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Classifier().to(device)
    checkpoint = torch.load(your_model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    prediction = []
    # for images, _ in test_loader:
    #     for image in images:
    #         final_prediction = apply_tta(model, image, augmentations)
    #         prediction.append(final_prediction)
    #

    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            test_pred = model(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    prediction_list = [[item] for item in prediction]
    test_set = FoodDataset("./test", aug_tfm, mode='train')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=18, pin_memory=True,
                             persistent_workers=True)
    for i in range(k):
        count = 0
        for data, _ in tqdm(test_loader):
            test_pred = model(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            temp = test_label.squeeze().tolist()
            for item in temp:
                prediction_list[count].append(item)
                count += 1

    # 将列表转换为 numpy 数组，以便使用 scipy 的 mode 函数
    array = np.array(prediction_list)

    # 计算每个维度的众数
    mode_result = mode(array, axis=1)
    modes = mode_result.mode
    count = mode_result.count
    for item in count:
        if item != k + 1:
            print(item)


    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)


    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(test_set))]
    df["Category"] = modes
    df.to_csv(f"ensemble/aug_submission_{current_time}.csv", index=False)
    print('finish')

    # create test csv
