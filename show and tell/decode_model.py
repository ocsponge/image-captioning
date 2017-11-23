# 定义训练模型

import os
import json
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config_options as config
import dataset_feed_model as feed

from tqdm import tqdm
from torch.utils import data
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

IMAGES_FEATURE_TRAIN_PATH = 'data/images_feature_train.pt'
IMAGES_LABEL_TRAIN_PATH   = 'data/images_label_train.pt'
IMAGES_FEATURE_PRED_PATH  = 'data/images_feature_pred.pt'
VOCABULARY_PATH           = 'data/vocabulary.pt'


class DecodeLSTM(nn.Module):
    def __init__(self, opt):
        super(DecodeLSTM, self).__init__()
        self.vocab_size        = opt.vocab_size
        self.embedding_size    = opt.embedding_size
        self.hidden_size       = opt.hidden_size
        self.num_layers        = opt.num_layers
        self.drop_prob         = opt.drop_prob
        self.image_feat_size   = opt.image_feat_size
        self.sentence_length   = opt.sentence_length

        self.image_embed_layer = nn.Linear(self.image_feat_size, self.embedding_size)
        self.word_embed_layer  = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm_layer        = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.drop_prob)
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

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()).cuda(),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()).cuda())

    def forward(self, images_feats, caption_sentences, lengths):
        batch_size       = images_feats.size(0)
        init_states      = self.init_hidden(batch_size)
        image_embeddings = self.image_embed_layer(images_feats)                     # 图片嵌入 (batch_size, embedding_size)
        char_embeddings  = self.word_embed_layer(caption_sentences)                # 描述嵌入 (batch_size, max_length, embedding_size)
        embeddings       = torch.cat((image_embeddings.unsqueeze(1), char_embeddings), 1)   # (batch_size, max_length+1, embedding_size)
        packed           = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _       = self.lstm_layer(packed, init_states)                              # (batch_size, max_length, hidden_size)
        outputs          = self.logit_layer(self.dropout_layer(hiddens.data))
        return outputs

    def sample(self, images_feats):
        self.lstm_layer.flatten_parameters()
        batch_size       = images_feats.size(0)
        states           = self.init_hidden(batch_size)
        sampled_ids      = []
        image_embeddings = self.image_embed_layer(images_feats)                     # 图片嵌入 (batch_size, embedding_size)
        inputs           = image_embeddings.unsqueeze(1)
        for i in range(self.sentence_length):                                      # maximum sampling length
            hiddens, states = self.lstm_layer(inputs, states)          # (batch_size, 1, hidden_size)
            outputs         = self.logit_layer(self.dropout_layer(hiddens.squeeze(1)))            # (batch_size, vocab_size)
            predicted       = outputs.max(1)[1].view(-1, 1)           # (batch_size, 1)
            sampled_ids.append(predicted)
            inputs          = self.word_embed_layer(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, sentence_length)
        return sampled_ids

    def beam_search(self, images_feats, opt):
        self.lstm_layer.flatten_parameters()
        sample_size         = images_feats.size(0)    # 样本数
        sample_seq          = torch.LongTensor(opt.sentence_length, sample_size).zero_()   # 一批数据里每个样本的句子
        sample_seq_logprobs = torch.FloatTensor(opt.sentence_length, sample_size)  # 每个样本的每个词的对数概率
        sample_done_beams   = [[] for _ in range(sample_size)]    # 搜索完成的样本
        for k in range(sample_size):         # 每个样本进行循环
            state             = self.init_hidden(opt.beam_size)
            beam_seq          = torch.LongTensor(opt.sentence_length, opt.beam_size).zero_()  # 每个词的beam_size种选项
            beam_seq_logprobs = torch.FloatTensor(opt.sentence_length, opt.beam_size).zero_()   # 每个词每个选项的对数概率
            beam_logprobs_sum = torch.zeros(opt.beam_size)     # 每个beam的对数概率和
            for t in range(opt.sentence_length+2):     # 每个词进行循环, 第t步存储的是第t-2的词
                if t == 0:     # 输入图像
                    embeddings = self.image_embed_layer(images_feats[k:k+1])
                    embeddings = embeddings.expand(opt.beam_size, self.embedding_size) # (beam_size, embedding_size)
                    embeddings = embeddings.unsqueeze(1)   # (beam_size, 1, embedding_size)
                elif t == 1:     # 输入 <start>
                    word_feats = torch.LongTensor(opt.beam_size, 1).zero_().cuda()    # 输入beam_size个开始符 (beam_size, 1)
                    embeddings = self.word_embed_layer(Variable(word_feats, requires_grad=False))  # (beam_size, 1, embedding_size)
                else:
                    tsorted, isorted = torch.sort(logprobs, dim=1, descending=True)   # 每行降序排列
                    candidates = []   # 候选词
                    rows = opt.beam_size
                    cols = opt.beam_size
                    if t == 2:  # 第一个词前一次输入均是<start>, 因此beam_size行每行输出都是一样的, 只取一行就行了；后面每个词的前一个词输入都是不一样的
                        rows = 1
                    for c in range(cols): # 有beam_size个候选
                        for q in range(rows): # 维护beam_size个序列
                            local_logprob = tsorted[q, c]  #候选单词的对数概率
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob   # beam_size行候选序列的对数概率和
                            # c: 所在词典中的位置索引   q: 序列号   p: 所在序列的总对数概率   r: 本身的对数概率
                            candidates.append({'c':isorted.data[q, c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])  # 按序列总对数概率降序排列

                    new_state = [_.clone() for _ in state]     # hn和cn  (num_layers, beam_size, hidden_size)
                    if t > 2:  # 如果是第二个词及以后
                        beam_seq_prev = beam_seq[:t-2].clone()   # 前缀序列的词
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()   # 前缀序列词的对数概率
                    for vix in range(opt.beam_size):   # 从candidates中选出beam_size个候选词
                        v = candidates[vix]  #第vix个序列所对应的词
                        if t > 2:  # 如果是第二个词及以后
                            beam_seq[:t-2, vix] = beam_seq_prev[:, v['q']]    #找出词对应的前缀序列
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        for state_ix in range(len(new_state)):   # state_ix: 0---hn    1---cn
                            new_state[state_ix][:, vix] = state[state_ix][:, v['q']]

                        # 添加新的词进入序列
                        beam_seq[t-2, vix] = v['c']
                        beam_seq_logprobs[t-2, vix] = v['r']
                        beam_logprobs_sum[vix] = v['p']

                        if v['c'] == 1 or (t-1) == opt.sentence_length:   # 遇到结束符<end>或者达到最大长度
                            sample_done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                         'logps': beam_seq_logprobs[:, vix].clone(),
                                                         'p': beam_logprobs_sum[vix]})   # 第k个样本完成

                    word_feats = beam_seq[t-2]  # 预测下一个时的输入, 当前词作为输入字 (beam_size, )
                    word_feats = word_feats.view(opt.beam_size, -1).cuda()   # (beam_size, 1)
                    embeddings = self.word_embed_layer(Variable(word_feats, requires_grad=False))  # (beam_size, 1, embedding_size)
                    state = new_state

                hiddens, state = self.lstm_layer(embeddings, state)  # (beam_size, 1, hidden_size)
                logprobs       = F.log_softmax(self.logit_layer(self.dropout_layer(hiddens.squeeze(1))))          # (beam_size, vocab_size)

            sample_done_beams[k] = sorted(sample_done_beams[k], key=lambda x: -x['p']) # 所有候选序列按对数概率和降序排列
            sample_seq[:, k] = sample_done_beams[k][0]['seq'] # 第一个对数概率和最大
            sample_seq_logprobs[:, k] = sample_done_beams[k][0]['logps']

        return sample_seq.transpose(0, 1), sample_seq_logprobs.transpose(0, 1)


#####################################################################################################################################################


def train(model, opt):
    model.train()
    model.cuda()
    criterion  = nn.CrossEntropyLoss()
    params     = model.parameters()
    optimizer  = optim.Adam(params, lr=opt.learning_rate)
    scheduler  = ReduceLROnPlateau(optimizer, mode=opt.scheduler_mode, factor=opt.lr_factor, verbose=True,
                                              patience=opt.patience, threshold=1e-3, threshold_mode='abs')
    dataset    = feed.FeedDatasetTrain(IMAGES_LABEL_TRAIN_PATH, IMAGES_FEATURE_TRAIN_PATH)
    dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=6, collate_fn=feed.collate_fn_train)
    batch_n    = len(dataloader)

    for iepoch in range(opt.max_epochs):
        for ix, (img, feats, label, lengths) in enumerate(dataloader):
            torch_images_feats = Variable(feats).cuda()
            torch_label_feats  = Variable(label).cuda()
            torch_label        = pack_padded_sequence(torch_label_feats, lengths, batch_first=True).data
            torch_output       = model.forward(torch_images_feats, torch_label_feats, lengths)

            batch_loss = criterion(torch_output, torch_label)
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm(params, opt.grad_clip)
            optimizer.step()

            if (ix+1) % opt.log_steps == 0 or (ix+1) == batch_n:
                logging.info('Epoch {:>2}/{}   Step {:>4}/{}   Loss: {:.6f}'.format((iepoch+1), opt.max_epochs, (ix+1),
                                                                                     batch_n, batch_loss.data[0]))

        scheduler.step(batch_loss.data[0])
        model_path = os.path.join(opt.save_path, 'lstm-{}.pkl'.format(iepoch+1))
        torch.save(model.state_dict(), model_path)

def predict(model, opt):
    model.eval()
    model.cuda()
    all_images_name = []
    all_sample_seqs = []
    dataset         = feed.FeedPredictPred(IMAGES_FEATURE_PRED_PATH)
    dataloader      = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=6)

    for ix, (img, feats) in enumerate(tqdm(dataloader)):
        torch_images_feats = Variable(feats).cuda()
        sample_seq, sample_seq_logprobs=model.beam_search(torch_images_feats, opt)   #(batch_size, sentence)
        sample_seq = sample_seq.cpu().numpy().tolist()
        #sample_seq         = model.sample(torch_images_feats).data.cpu().numpy().tolist()
        all_images_name.extend(img)
        all_sample_seqs.extend(sample_seq)
    return all_images_name, all_sample_seqs


if __name__=='__main__':
    torch.manual_seed(2017)
    opt = config.parse_opt()

    if not opt.pred:             # train
        decode_lstm_model = DecodeLSTM(opt)
        train(decode_lstm_model, opt)
    else:                        # predict
        decode_model_path = os.path.join(opt.save_path, 'lstm-{}.pkl'.format(opt.model))
        decode_lstm_model = DecodeLSTM(opt)
        decode_lstm_model.load_state_dict(torch.load(decode_model_path))
        vocabulary        = torch.load(VOCABULARY_PATH)
        word_map          = {v:k for k,v in vocabulary.items()}
        all_word_seqs     = []
        all_images_name, all_sample_seqs = predict(decode_lstm_model, opt)

        for row in all_sample_seqs:
            new_word_seq = [word_map[iw] for iw in row if iw != 0 and iw != 1]
            all_word_seqs.append(''.join(new_word_seq))

        result_json = [{'image_id': all_images_name[ix].split('.')[0], 'caption': all_word_seqs[ix]} for ix in range(len(all_images_name))]
        with open('data/result.json', 'w') as fw:
            json.dump(result_json, fw, ensure_ascii=False, indent=4)
