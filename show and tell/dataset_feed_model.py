# 生成训练batch

import torch
from torch.utils import data

IMAGES_FEATURE_TRAIN_PATH = 'data/images_feature_train.pt'
IMAGES_LABEL_TRAIN_PATH   = 'data/images_label_train.pt'
IMAGES_FEATURE_PRED_PATH  = 'data/images_feature_pred.pt'

BATCH_SIZE  = 5
SHUFFLE     = False
NUM_WORKERS = 6

class FeedDatasetTrain(data.Dataset):
    def __init__(self, images_label_train_path, images_feature_train_path):
        assert torch.load(images_feature_train_path)['images_names'] == torch.load(images_label_train_path)['images_names']
        self.images_names   = torch.load(images_label_train_path)['images_names']
        self.images_label   = torch.load(images_label_train_path)['images_label']
        self.images_feature = torch.load(images_feature_train_path)['images_feature']

    def __getitem__(self, index):
        img   = self.images_names[index]
        label = self.images_label[index]
        feats = self.images_feature[index]
        return img, feats, label

    def __len__(self):
        return len(self.images_names)

def collate_fn_train(batch):
    batch.sort(key=lambda x: len(x[2]), reverse=True)
    img, feats, label = zip(*batch)
    feats       = torch.stack(feats)
    lengths     = [len(icap) for icap in label]
    label_torch = torch.LongTensor(len(img), max(lengths)).zero_()
    for ix, icap in enumerate(label):
        end_id  = lengths[ix]
        label_torch[ix, :end_id].copy_(torch.LongTensor(icap))
    return img, feats, label_torch, lengths


class FeedPredictPred(data.Dataset):
    def __init__(self, images_feature_pred_path):
        self.images_names   = torch.load(images_feature_pred_path)['images_names']
        self.images_feature = torch.load(images_feature_pred_path)['images_feature']

    def __getitem__(self, index):
        img   = self.images_names[index]
        feats = self.images_feature[index]
        return img, feats

    def __len__(self):
        return len(self.images_names)


if __name__=='__main__':
    dataset    = FeedDatasetTrain(IMAGES_LABEL_TRAIN_PATH, IMAGES_FEATURE_TRAIN_PATH)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, collate_fn=collate_fn_train)
    d=torch.load('data/vocabulary.pt')
    dv={v:k for k,v in d.items()}
    print(len(dataset))
    print(dataset[0])
    print([dv[i] for i in dataset[0][2]])
    for i in dataloader:
        print(i)
        break
