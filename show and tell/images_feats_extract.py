# 提取图像特征

import os
import sys
import torch
import torch.nn as nn
import torchvision as tv

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torch.autograd import Variable

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RUN_MODE            = sys.argv[1]
IMAGES_FILE_PATH    = 'images_' + RUN_MODE
IMAGES_FEATURE_PATH = 'data/images_feature_' + RUN_MODE + '.pt'

BATCH_SIZE  = 512
SHUFFLE     = False
NUM_WORKERS = 6
EPOCH_NUM   = 1 if RUN_MODE == 'pred' else 5

class CaptionDataset(data.Dataset):
    def __init__(self, images_file_path, mode):
        self.image_corp       = tv.transforms.CenterCrop if mode == 'pred' else tv.transforms.RandomCrop
        self.image_transforms = tv.transforms.Compose([tv.transforms.Scale(224),
                                                       tv.transforms.RandomHorizontalFlip(),
                                                       self.image_corp(224),
                                                       tv.transforms.ToTensor(),
                                                       tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        self.images_file_path  = images_file_path
        self.images_names      = os.listdir(images_file_path)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.images_file_path, self.images_names[index])).convert('RGB')
        img = self.image_transforms(img)
        return self.images_names[index], img

    def __len__(self):
        return len(self.images_names)

if __name__=='__main__':
    dataset    = CaptionDataset(IMAGES_FILE_PATH, RUN_MODE)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)

    resnet50 = tv.models.resnet50(pretrained=True)
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
    resnet50.cuda()
    resnet50.eval()

    images_names   = []
    images_feature = torch.FloatTensor(len(dataset)*EPOCH_NUM, 2048).zero_()

    for iepoch in range(EPOCH_NUM):
        for ix, ibatch in enumerate(tqdm(dataloader)):
            images_names.extend(list(ibatch[0]))
            start = iepoch*len(dataset) + ix*BATCH_SIZE
            end   = iepoch*len(dataset) + min((ix+1)*BATCH_SIZE, len(dataset))
            images_input = Variable(ibatch[1].cuda(), volatile=True)
            images_out   = resnet50(images_input)
            images_out   = images_out.cpu().data.view(images_out.size(0), -1)
            images_feature[start:end].copy_(images_out)

    print('the number of images: {}'.format(len(images_names)))
    print('the shape of images feature: {}'.format(images_feature.size()))

    result={'images_names':images_names, 'images_feature':images_feature}
    torch.save(result, IMAGES_FEATURE_PATH)
