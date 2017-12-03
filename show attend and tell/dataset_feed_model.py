# 生成训练batch

import os
import sys

import torch
import torchvision as tv

from PIL import Image
from torch.utils import data

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RUN_MODE                = sys.argv[1]
IMAGES_FILE_PATH        = 'images_' + RUN_MODE
IMAGES_LABEL_TRAIN_PATH = 'data/images_label_train.pt'

BATCH_SIZE  = 5
SHUFFLE     = False
NUM_WORKERS = 6

# 训练和验证时采用的dataset, 因为训练和验证时一张图片有5个描述
class FeedDatasetTrain(data.Dataset):
    def __init__(self, images_file_path, images_label_train_path):
        self.image_transforms = tv.transforms.Compose([tv.transforms.Scale(224),
                                                       tv.transforms.RandomHorizontalFlip(),    # 数据增强
                                                       tv.transforms.RandomCrop(224),
                                                       tv.transforms.ToTensor(),
                                                       tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        self.images_file_path = images_file_path
        self.images_names     = torch.load(images_label_train_path)['images_names']
        self.images_label     = torch.load(images_label_train_path)['images_label']

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.images_file_path, self.images_names[index])).convert('RGB')
        img = self.image_transforms(img)
        return self.images_names[index], img, self.images_label[index]

    def __len__(self):
        return len(self.images_names)

def collate_fn_train(batch):
    #batch.sort(key=lambda x: len(x[2]), reverse=True)
    img, feats, label = zip(*batch)
    feats       = torch.stack(feats)
    lengths     = [len(icap) for icap in label]
    label_torch = torch.LongTensor(len(img), max(lengths)).zero_()
    for ix, icap in enumerate(label):
        end_id  = lengths[ix]
        label_torch[ix, :end_id].copy_(torch.LongTensor(icap))
    return img, feats, label_torch, lengths


# 预测时采用的dataset, 因为预测时一张图片只有1个描述
class FeedPredictPred(data.Dataset):
    def __init__(self, images_file_path):
        self.image_transforms = tv.transforms.Compose([tv.transforms.Scale(224),
                                                       tv.transforms.CenterCrop(224),
                                                       tv.transforms.ToTensor(),
                                                       tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        self.images_file_path = images_file_path
        self.images_names     = os.listdir(images_file_path)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.images_file_path, self.images_names[index])).convert('RGB')
        img = self.image_transforms(img)
        return self.images_names[index], img

    def __len__(self):
        return len(self.images_names)


if __name__=='__main__':
    dataset    = FeedDatasetTrain(IMAGES_FILE_PATH, IMAGES_LABEL_TRAIN_PATH)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, collate_fn=collate_fn_train)
    d=torch.load('data/vocabulary.pt')
    dv={v:k for k,v in d.items()}
    print(len(dataset))
    print(dataset[0])
    print([dv[i] for i in dataset[0][2]])
    for i in dataloader:
        print(i)
        break
