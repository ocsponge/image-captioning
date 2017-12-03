# 定义常见参数

import argparse

def parse_opt():
    parser=argparse.ArgumentParser()

    # 模型参数
    model_group=parser.add_argument_group('model arguments')
    model_group.add_argument('-vocab-size', metavar='', type=int, default=9813, help='size of vocabulary')
    model_group.add_argument('-sentence-length', metavar='', type=int, default=32, help='length of caption sentence')
    model_group.add_argument('-hidden-size', metavar='', type=int, default=512, help='size of hidden nodes in each layer')
    model_group.add_argument('-num-layers', metavar='', type=int, default=2, help='number of layers in the rnn')
    model_group.add_argument('-embedding-size', metavar='', type=int, default=512, help='embdding size of each token in the vocabulary')
    model_group.add_argument('-image-feat-size', metavar='', type=int, default=2048, help='2048 for resnet50')
    model_group.add_argument('-drop-prob', metavar='', type=float, default=0.5, help='dropout in the Language model rnn')
    model_group.add_argument('-att-feat-size', metavar='', type=int, default=2048, help='2048 for resnet50')
    model_group.add_argument('-att-hidden-size', metavar='', type=int, default=512, help='the hidden size of the attention layer')

    # 训练参数
    train_group=parser.add_argument_group('train arguments')
    train_group.add_argument('-max-epochs', metavar='', type=int, default=15, help='number of epochs')
    train_group.add_argument('-batch-size', metavar='', type=int, default=256, help='batch size')
    train_group.add_argument('-grad-clip', metavar='', type=float, default=0.25, help='clip gradients at this value')
    train_group.add_argument('-beam-size', metavar='', type=int, default=3, help='number of beams in beam search')
    train_group.add_argument('-learning-rate', metavar='', type=float, default=1e-4, help='learning rate')
    train_group.add_argument('-scheduler-mode', metavar='', type=str, default='min', help='one of min, max')
    train_group.add_argument('-lr-factor', metavar='', type=float, default=0.1, help='factor by which the learning rate will be reduced')
    train_group.add_argument('-patience', metavar='', type=int, default=3, help='number of epochs with no improvement after which lr will be reduced')
    train_group.add_argument('-log-steps', metavar='', type=int, default=500, help='number of steps after which training info will be printed')
    train_group.add_argument('-save-path', metavar='', type=str, default='model', help='model save path')

    # 预测参数
    pred_group=parser.add_argument_group('pred arguments')
    pred_group.add_argument('-pred', action='store_true', help='change to predict mode')
    pred_group.add_argument('-model', metavar='', type=str, default='50-50', help='the order number of saved models to predict')

    args = parser.parse_args()

    # 检查参数是否合法
    assert args.vocab_size > 0, "vocab size should be greater than 0"
    assert args.hidden_size > 0, "rnn size should be greater than 0"
    assert args.num_layers > 0, "num layers should be greater than 0"
    assert args.embedding_size > 0, "embedding size should be greater than 0"
    assert args.max_epochs > 0, "max epochs should be greater than 0"
    assert args.batch_size > 0, "batch size should be greater than 0"
    assert args.drop_prob >= 0 and args.drop_prob < 1, "drop prob should be between 0 and 1"
    assert args.beam_size > 0, "beam size should be greater than 0"
    assert args.learning_rate > 0, "learning rate should be greater than 0"

    return args

if __name__=='__main__':
    args=parse_opt()
    print(args)
