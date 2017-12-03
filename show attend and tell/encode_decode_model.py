# 定义训练模型

import os
import json
import logging

import torch
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
import torch.nn.functional as F
import dataset_feed_model as feed
import model_config_options as config

from tqdm import tqdm
from torch.utils import data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

IMAGES_FILE_TRAIN_PATH  = 'images_train'
IMAGES_FILE_PRED_PATH   = 'images_pred'
IMAGES_LABEL_TRAIN_PATH = 'data/images_label_train.pt'
VOCABULARY_PATH         = 'data/vocabulary.pt'

# 利用resnet50提取图像特征
class EncodeResnet50(nn.Module):
    def __init__(self, opt):
        super(EncodeResnet50, self).__init__()
        self.att_feat_size          = opt.att_feat_size
        self.encode_model           = tv.models.resnet50(pretrained=True)
        self.feature_maps_extractor = nn.Sequential(*list(self.encode_model.children())[:-2])     # 提取feature_maps
        for param in self.feature_maps_extractor.parameters():
            param.requires_grad = False
        #self.feature_maps_extractor.cuda()
        #self.feature_maps_extractor.eval()

    def forward(self, images_variable):
        feature_maps = self.feature_maps_extractor(images_variable)
        feature_maps = feature_maps.view(feature_maps.size(0), opt.att_feat_size, -1)
        feature_list = torch.mean(feature_maps, dim=2)
        return feature_maps, feature_list

# attention
class AttentionModel(nn.Module):                    # 注意力attention模型
    def __init__(self, opt):
        super(AttentionModel, self).__init__()
        self.embedding_size   = opt.embedding_size
        self.hidden_size      = opt.hidden_size
        self.num_layers       = opt.num_layers
        self.drop_prob        = opt.drop_prob
        self.att_feat_size    = opt.att_feat_size                     # 就是feature map的数量
        self.att_hidden_size  = opt.att_hidden_size

        self.lstm_layer       = nn.LSTM(self.embedding_size+self.att_feat_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.drop_prob)
        self.image2att_layer  = nn.Linear(self.att_feat_size, self.att_hidden_size)
        self.hidden2att_layer = nn.Linear(self.hidden_size, self.att_hidden_size)
        self.alpha_layer      = nn.Linear(self.att_hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.image2att_layer.bias.data.fill_(0)
        self.image2att_layer.weight.data.uniform_(-initrange, initrange)
        self.hidden2att_layer.bias.data.fill_(0)
        self.hidden2att_layer.weight.data.uniform_(-initrange, initrange)
        self.alpha_layer.bias.data.fill_(0)
        self.alpha_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, word_embeddings, feature_maps, state):
        #self.lstm_layer.flatten_parameters()
        batch_size       = feature_maps.size(0)
        attention_image  = feature_maps.contiguous().view(-1, self.att_feat_size)                  # (batch * map_size) * att_feat_size
        attention_image  = self.image2att_layer(attention_image)                      # (batch * map_size) * att_hidden_size
        attention_image  = attention_image.view(batch_size, -1, self.att_hidden_size) # batch * map_size * att_hidden_size
        attention_hidden = self.hidden2att_layer(state[0][-1])                    # 表示最后一层的隐状态 batch * att_hidden_size
        attention_hidden = attention_hidden.unsqueeze(1).expand_as(attention_image)   # batch * map_size * att_hidden_size
        attention        = attention_image + attention_hidden                         # batch * map_size * att_hidden_size
        attention        = F.tanh(attention)                                          # batch * map_size * att_hidden_size
        attention        = attention.view(-1, self.att_hidden_size)                   # (batch * map_size) * att_hidden_size
        attention        = self.alpha_layer(attention)                                # (batch * map_size) * 1
        attention        = attention.view(batch_size, -1)                             # batch * map_size
        weights          = F.softmax(attention)                                       # batch * map_size
        feature_maps_    = feature_maps.view(batch_size, -1, self.att_feat_size)      # batch * map_size * att_feat_size
        attention_feats  = torch.bmm(weights.unsqueeze(1), feature_maps_)             # batch * 1 * att_feat_size
        output, state    = self.lstm_layer(torch.cat([word_embeddings, attention_feats], 2), state)
        return output.view(batch_size, -1), state                                     # batch * hidden_size

# lstm做decode
class DecodeLSTM(nn.Module):                        # 整体decode模型
    def __init__(self, opt):
        super(DecodeLSTM, self).__init__()
        self.vocab_size        = opt.vocab_size
        self.embedding_size    = opt.embedding_size
        self.hidden_size       = opt.hidden_size
        self.num_layers        = opt.num_layers
        self.drop_prob         = opt.drop_prob
        self.image_feat_size   = opt.image_feat_size
        self.sentence_length   = opt.sentence_length
        self.beam_size         = opt.beam_size

        self.image_embed_layer = nn.Linear(self.image_feat_size, self.num_layers * self.hidden_size)   # 图片生成初始隐状态
        self.word_embed_layer  = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm_layer        = AttentionModel(opt)
        self.dropout_layer     = nn.Dropout(self.drop_prob)
        self.logit_layer       = nn.Linear(self.hidden_size, self.vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.image_embed_layer.bias.data.fill_(0)
        self.image_embed_layer.weight.data.uniform_(-initrange, initrange)
        self.word_embed_layer.weight.data.uniform_(-initrange, initrange)
        self.logit_layer.bias.data.fill_(0)
        self.logit_layer.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, images_feats):
        init_state = self.image_embed_layer(images_feats).view(self.num_layers, -1, self.hidden_size)    # (batch_size, num_layers, hidden_size)
        return (init_state, init_state)

    def forward(self, images_feats, feature_maps, caption_sentences):
        batch_size = images_feats.size(0)
        state      = self.init_hidden(images_feats)
        outputs    = []
        for i in range(caption_sentences.size(1) - 1):
            word          = caption_sentences[:, i]                                # (batch_size, 1)
            embeddings    = self.word_embed_layer(word)
            embeddings    = embeddings.unsqueeze(1)                            # (batch_size, 1, embedding_size)
            output, state = self.lstm_layer(embeddings, feature_maps, state)  # (batch_size, hidden_size)
            output        = self.logit_layer(self.dropout_layer(output))           # (batch_size, vocab_size)
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1).view(-1, self.vocab_size)   # (batch_size, max_len-1, vocab_size)

    def beam_search(self, images_feats, feature_maps):
        sample_size         = images_feats.size(0)    # 样本数
        sample_seq          = torch.LongTensor(self.sentence_length, sample_size).zero_()   # 一批数据里每个样本的句子
        sample_seq_logprobs = torch.FloatTensor(self.sentence_length, sample_size)  # 每个样本的每个词的对数概率
        sample_done_beams   = [[] for _ in range(sample_size)]    # 搜索完成的样本
        for k in range(sample_size):         # 每个样本进行循环
            this_image_feats  = images_feats[k:k+1].expand(self.beam_size, self.image_feat_size)
            this_feature_map  = feature_maps[k:k+1].expand(*((self.beam_size,) + feature_maps.size()[1:]))
            state             = self.init_hidden(this_image_feats)
            beam_seq          = torch.LongTensor(self.sentence_length, self.beam_size).zero_()  # 每个词的beam_size种选项
            beam_seq_logprobs = torch.FloatTensor(self.sentence_length, self.beam_size).zero_()   # 每个词每个选项的对数概率
            beam_logprobs_sum = torch.zeros(self.beam_size)     # 每个beam的对数概率和
            for t in range(self.sentence_length + 1):     # 每个词进行循环, 第t步存储的是第t-1的词
                if t == 0:     # 输入 <start> = 1
                    word_feats = torch.ones(self.beam_size, 1).long().cuda()    # 输入beam_size个开始符 (beam_size, 1)
                    embeddings = self.word_embed_layer(Variable(word_feats, requires_grad=False))  # (beam_size, 1, embedding_size)
                else:
                    tsorted, isorted = torch.sort(logprobs, dim=1, descending=True)   # 每行降序排列
                    candidates = []   # 候选词
                    rows = self.beam_size
                    cols = self.beam_size
                    if t == 1:  # 第一个词前一次输入均是<start>, 因此beam_size行每行输出都是一样的, 只取一行就行了；后面每个词的前一个词输入都是不一样的
                        rows = 1
                    for c in range(cols): # 有beam_size个候选
                        for q in range(rows): # 维护beam_size个序列
                            local_logprob = tsorted[q, c]  #候选单词的对数概率
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob   # beam_size行候选序列的对数概率和
                            # c: 所在词典中的位置索引   q: 序列号   p: 所在序列的总对数概率   r: 本身的对数概率
                            candidates.append({'c':isorted.data[q, c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])  # 按序列总对数概率降序排列

                    new_state = [_.clone() for _ in state]     # hn和cn  (num_layers, beam_size, hidden_size)
                    if t > 1:  # 如果是第二个词及以后
                        beam_seq_prev = beam_seq[:t-1].clone()   # 前缀序列的词
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-1].clone()   # 前缀序列词的对数概率
                    for vix in range(self.beam_size):   # 从candidates中选出beam_size个候选词
                        v = candidates[vix]  #第vix个序列所对应的词
                        if t > 1:  # 如果是第二个词及以后
                            beam_seq[:t-1, vix] = beam_seq_prev[:, v['q']]    #找出词对应的前缀序列
                            beam_seq_logprobs[:t-1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        for state_ix in range(len(new_state)):   # state_ix: 0---hn    1---cn
                            new_state[state_ix][:, vix] = state[state_ix][:, v['q']]

                        # 添加新的词进入序列
                        beam_seq[t-1, vix] = v['c']
                        beam_seq_logprobs[t-1, vix] = v['r']
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 2 or t == self.sentence_length:   # 遇到结束符<end>或者达到最大长度
                            sample_done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                         'logps': beam_seq_logprobs[:, vix].clone(),
                                                         'p': beam_logprobs_sum[vix]})   # 第k个样本完成

                    word_feats = beam_seq[t-1]  # 预测下一个时的输入, 当前词作为输入字 (beam_size, )
                    word_feats = word_feats.view(self.beam_size, -1).cuda()   # (beam_size, 1)
                    embeddings = self.word_embed_layer(Variable(word_feats, requires_grad=False))  # (beam_size, 1, embedding_size)
                    state = new_state

                hiddens, state = self.lstm_layer(embeddings, this_feature_map, state)  # (beam_size, hidden_size)
                logprobs       = F.log_softmax(self.logit_layer(self.dropout_layer(hiddens)))          # (beam_size, vocab_size)

            sample_done_beams[k] = sorted(sample_done_beams[k], key=lambda x: -x['p']) # 所有候选序列按对数概率和降序排列
            sample_seq[:, k] = sample_done_beams[k][0]['seq'] # 第一个对数概率和最大
            sample_seq_logprobs[:, k] = sample_done_beams[k][0]['logps']

        return sample_seq.transpose(0, 1), sample_seq_logprobs.transpose(0, 1)


#####################################################################################################################################################


def train(encode_model, decode_model, opt):
    encode_model.train()
    encode_model.cuda()
    decode_model.train()
    decode_model.cuda()

    criterion  = nn.CrossEntropyLoss()
    params     = decode_model.parameters()
    optimizer  = optim.Adam(params, lr=opt.learning_rate)
    scheduler  = ReduceLROnPlateau(optimizer, mode=opt.scheduler_mode, factor=opt.lr_factor, verbose=True,
                                              patience=opt.patience, threshold=1e-3, threshold_mode='abs')
    dataset    = feed.FeedDatasetTrain(IMAGES_FILE_TRAIN_PATH, IMAGES_LABEL_TRAIN_PATH)
    dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=6, collate_fn=feed.collate_fn_train)
    best_loss  = 1e15
    batch_n    = len(dataloader)

    for iepoch in range(opt.max_epochs):
        for ix, (img, feats, label, lengths) in enumerate(dataloader):
            torch_images_feats = Variable(feats).cuda()
            torch_label_feats  = Variable(label).cuda()
            torch_label        = torch_label_feats[:, 1:].contiguous().view(-1)

            torch_images_feature_maps, torch_images_feature_list = encode_model(torch_images_feats)
            torch_output       = decode_model(torch_images_feature_list, torch_images_feature_maps, torch_label_feats)
            batch_loss = criterion(torch_output, torch_label)
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm(params, opt.grad_clip)
            optimizer.step()

            if (ix+1) % opt.log_steps == 0 or (ix+1) == batch_n:
                logging.info('Epoch {:>2}/{}   Step {:>4}/{}   Loss: {:.6f}'.format(iepoch, opt.max_epochs, ix, batch_n, batch_loss.data[0]))
                if batch_loss.data[0] < best_loss:
                    best_loss = batch_loss.data[0]
                    for imodel in os.listdir(opt.save_path):
                        os.remove(os.path.join(opt.save_path, imodel))
                    model_path = os.path.join(opt.save_path, 'lstm-{}-{}.pkl'.format(iepoch, ix))
                    torch.save(decode_model.state_dict(), model_path)

                scheduler.step(batch_loss.data[0])


def predict(encode_model, decode_model, opt):
    encode_model.eval()
    encode_model.cuda()
    decode_model.eval()
    decode_model.cuda()

    all_images_name = []
    all_sample_seqs = []
    dataset         = feed.FeedPredictPred(IMAGES_FILE_PRED_PATH)
    dataloader      = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=6)

    for ix, (img, feats) in enumerate(tqdm(dataloader)):
        torch_images_feats = Variable(feats).cuda()
        torch_images_feature_maps, torch_images_feature_list = encode_model(torch_images_feats)

        sample_seq, sample_seq_logprobs = decode_model.beam_search(torch_images_feature_list, torch_images_feature_maps)   #(batch_size, sentence)
        sample_seq = sample_seq.cpu().numpy().tolist()
        all_images_name.extend(img)
        all_sample_seqs.extend(sample_seq)
    return all_images_name, all_sample_seqs


if __name__=='__main__':
    torch.manual_seed(2017)
    opt = config.parse_opt()
    encode_model = EncodeResnet50(opt)

    if not opt.pred:             # train
        decode_model = DecodeLSTM(opt)
        train(encode_model, decode_model, opt)
    else:                        # predict
        decode_model_path = os.path.join(opt.save_path, 'lstm-{}.pkl'.format(opt.model))
        decode_model = DecodeLSTM(opt)
        decode_model.load_state_dict(torch.load(decode_model_path))
        vocabulary    = torch.load(VOCABULARY_PATH)
        word_map      = {v:k for k,v in vocabulary.items()}
        all_word_seqs = []
        all_images_name, all_sample_seqs = predict(encode_model, decode_model, opt)

        for row in all_sample_seqs:
            new_word_seq = [word_map[iw] for iw in row if iw != 0 and iw != 1 and iw != 2]
            all_word_seqs.append(''.join(new_word_seq))

        result_json = [{'image_id': all_images_name[ix].split('.')[0], 'caption': all_word_seqs[ix]} for ix in range(len(all_images_name))]
        with open('data/result.json', 'w') as fw:
            json.dump(result_json, fw, ensure_ascii=False, indent=4)
