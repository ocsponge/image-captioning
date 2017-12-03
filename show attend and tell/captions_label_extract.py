# 将文字标注生成字典和训练集

import gc
import sys
import jieba
import torch
import pandas as pd
from collections import Counter

RUN_MODE            = sys.argv[1]
CAPTIONS_FILE_PATH  = 'captions_' + RUN_MODE + '.json'
VOCABULARY_PATH     = 'data/vocabulary.pt'
IMAGES_LABEL_PATH   = 'data/images_label_' + RUN_MODE + '.pt'

if __name__=='__main__':
    images_captions = pd.read_json(CAPTIONS_FILE_PATH)
    images_captions.caption = images_captions.caption.map(lambda line: [elem.replace('\n', '') for elem in line])

    # 每张图片有5个描述,生成5个样本
    images_captions['caption0'] = images_captions.caption.map(lambda line: line[0])
    images_captions['caption1'] = images_captions.caption.map(lambda line: line[1])
    images_captions['caption2'] = images_captions.caption.map(lambda line: line[2])
    images_captions['caption3'] = images_captions.caption.map(lambda line: line[3])
    images_captions['caption4'] = images_captions.caption.map(lambda line: line[4])

    images_captions_0 = images_captions[['image_id', 'caption0']].rename(columns={'caption0': 'caption'})
    images_captions_1 = images_captions[['image_id', 'caption1']].rename(columns={'caption1': 'caption'})
    images_captions_2 = images_captions[['image_id', 'caption2']].rename(columns={'caption2': 'caption'})
    images_captions_3 = images_captions[['image_id', 'caption3']].rename(columns={'caption3': 'caption'})
    images_captions_4 = images_captions[['image_id', 'caption4']].rename(columns={'caption4': 'caption'})

    images_captions_all = pd.concat([images_captions_0, images_captions_1, images_captions_2, images_captions_3, images_captions_4])

    del images_captions, images_captions_0, images_captions_1, images_captions_2, images_captions_3, images_captions_4
    gc.collect()

    images_captions_all.caption = images_captions_all.caption.map(lambda string: list(jieba.cut(string, cut_all=False)))

    # 构造词典, 添加<start> <end> <unknown> <padding>
    if RUN_MODE == 'train':
        vocabulary = images_captions_all.caption.tolist()
        vocabulary = [word for line in vocabulary for word in line]
        word_count = Counter(vocabulary)
        vocabulary = [k for k,v in word_count.items() if v >= 3]
        vocabulary = {v:(k+4) for k,v in enumerate(vocabulary)}
        vocabulary.update({'<padding>':0, '<start>':1, '<end>':2, '<unknown>':3})
        print('the length of vocabulary: {}'.format(len(vocabulary)))
        torch.save(vocabulary, VOCABULARY_PATH)

    vocabulary = torch.load(VOCABULARY_PATH)
    images_captions_all.caption = images_captions_all.caption.map(lambda line: [word if word in vocabulary else '<unknown>' for word in line])
    images_captions_all.caption = images_captions_all.caption.map(lambda line: [1]+[vocabulary[word] for word in line]+[2])

    images_names = images_captions_all.image_id.tolist()
    images_label = images_captions_all.caption.tolist()

    print('the number of images: {}'.format(len(images_names)))
    print('the number of images: {}'.format(len(images_label)))

    result={'images_names': images_names, 'images_label': images_label}
    torch.save(result, IMAGES_LABEL_PATH)
