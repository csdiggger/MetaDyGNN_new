# coding: utf-8
# author: wcc
# create date: 2021-01-10 20:35
# main.py 作为入口文件

import random
import time
import numpy as np
import torch
from MetaDyGNN import MetaDyGNN
from DataHelper import DataHelper

# 指定使用的显卡编号 默认从0开始
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设置随机种子是为了确保每次生成固定的随机数 全局变量
# 这就使得每次实验结果显示一致了，有利于实验的比较和改进
np.random.seed(2021)
torch.manual_seed(2021)

# 以下分别是训练、验证、测试函数 由主函数发起调用
def training(model, model_save=True, model_file=None):
    print('training model...')
    if config['use_cuda']:
        model.cuda()
    model.train()

    batch_size = config['batch_size'] # 48
    num_epoch = config['num_epoch'] # 24
    print('num_batch: ', int(len(train_data) / batch_size))

    # 一共 24 epoch
    for _ in range(num_epoch):  
        loss, acc, ap, f1, auc = [], [], [], [], []
        start = time.time()

        random.shuffle(train_data)
        num_batch = int(len(train_data) / batch_size) # 2093/48 = 43.6
        support_x, support_y, query_x, query_y = zip(*train_data)  # supp_um_s:(list,list,...,2553)
        
        # 每个 epoch 有 43 个 batch
        for i in range(num_batch):  # each batch contains some tasks (each task contains a support set and a query set)
            # 每轮仅取得 48 个data进行训练
            support_xs = list(support_x[batch_size * i:batch_size * (i + 1)])
            support_ys = list(support_y[batch_size * i:batch_size * (i + 1)])
            query_xs = list(query_x[batch_size * i:batch_size * (i + 1)])
            query_ys = list(query_y[batch_size * i:batch_size * (i + 1)])

            # 调用dygnn类中的全局更新
            print('调用dygnn类中的全局更新，当前为第', i, '次，一共次43次')
            _loss, _acc, _ap, _f1, _auc = model.global_update(support_xs, support_ys, query_xs, query_ys)

            loss.append(_loss)
            acc.append(_acc)
            ap.append(_ap)
            f1.append(_f1)
            auc.append(_auc)

            # 每20个batch输出
            if i % 20 == 0 and i != 0:
                print('batch: {}, loss: {:.6f}, cost time: {:.1f}s, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, auc: {:.5f}'.
                      format(i, np.mean(loss), time.time() - start, np.mean(acc), np.mean(ap), np.mean(f1), np.mean(auc)))
        
        print('所有数据已经训练一遍 即第', _, '个epoch已完成，一共24个epoch')
        print('epoch: {}, loss: {:.6f}, cost time: {:.1f}s, acc: {:.5f}, ap: {:.5f}, f1: {:.5f}, auc: {:.5f}'.
              format(_, np.mean(loss), time.time() - start, np.mean(acc), np.mean(ap), np.mean(f1), np.mean(auc)))
        
        # 每个epoch时需要验证、测试
        if _ % 1 == 0:
            validation(model)
            testing(model)

            model.train()

    # 24 epoch后保存模型
    if model_save:
        print('saving model...')
        torch.save(model.state_dict(), model_file)


def validation(model):
    # testing
    print('evaluating model...')
    if config['use_cuda']:
        model.cuda()
    model.eval()

    support_x, support_y, query_x, query_y = zip(*valid_data)

    # 调用dygnn类中的评价函数
    acc, ap, f1, auc = model.evaluate(support_x, support_y, query_x, query_y)

    print('val acc: {:.5f}, val ap: {:.5f}, val f1: {:.5f}, val auc: {:.5f}'.
          format(np.mean(acc), np.mean(ap), np.mean(f1), np.mean(auc)))

# 与验证的流程完全一样 仅数据不一样
def testing(model):
    # testing
    print('evaluating model...')
    if config['use_cuda']:
        model.cuda()
    model.eval()

    support_x, support_y, query_x, query_y = zip(*test_data)

    acc, ap, f1, auc = model.evaluate(support_x, support_y, query_x, query_y)

    print('tst acc: {:.5f}, tst ap: {:.5f}, tst f1: {:.5f}, tst auc: {:.5f}'.
          format(np.mean(acc), np.mean(ap), np.mean(f1), np.mean(auc)))

def new_func(train_ngh_finder):
    return train_ngh_finder

# 主函数入口
if __name__ == "__main__":

    data_set = 'wikipedia'
    # data_set = 'reddit'
    # data_set = 'dblp'

    res_dir = '../res/' + data_set
    load_model = False

    # training model.

    model_name = 'time_interval_sp'

    # 从Config中导入三种配置 load边特征（真实）和节点特征（随机生成）
    # 格式化字符串的函数 str.format(string) string替换str中的{0}

    if data_set == 'wikipedia':
        from Config import config_wikipedia as config
        e_feat = np.load('../data/{0}/ml_{0}.npy'.format(data_set))
        n_feat = np.load('../data/{0}/ml_{0}_node.npy'.format(data_set))
    elif data_set == 'reddit':
        from Config import config_reddit as config
        e_feat = np.load('../data/{0}/ml_{0}.npy'.format(data_set))
        n_feat = np.load('../data/{0}/ml_{0}_node.npy'.format(data_set))
    elif data_set == 'dblp':
        from Config import config_dblp as config
        e_feat = np.load('../data/{0}/ml_{0}.npy'.format(data_set))
        n_feat = np.load('../data/{0}/ml_{0}_node.npy'.format(data_set))

    # 输出UTC标准时间 config参数
    print('Ready to begin.')
    print(time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()))
    print(config)
    print('Config settings confirmation.')

    # 如果有已保存的模型 记录模型路径
    model_filename = "{}/mdgnn.pkl".format(res_dir)
    
    # 调用DataHelper类的构造函数 传入数据集名称 shot 节点可见性 进行元训练集和测试集的划分
    print('调用datahelper的init函数')
    datahelper = DataHelper(data_set, k_shots=config['k_shots'], full_node=True) # 仅调用了init
    print('data训练集三元组已打包好，还没load_data')

    # 根据是否具有时间间隔 调用不同的load函数 为 [训练集 验证集 测试集 全部邻居节点 用于训练的邻居节点] 赋值
    # 本例是time_interval_sp 不满足if条件 调用else 如满足if 需要除以时间间隔
    if 'time_interval' in model_name:
        print('wikipedia 数据集调用了这个函数')
        # \ 是分行的标识符
        train_data, valid_data, test_data, full_ngh_finder, train_ngh_finder \
            = datahelper.load_data_in_time_sp(interval=config['interval'])
    else:
        train_data, valid_data, test_data, full_ngh_finder, train_ngh_finder = datahelper.load_data()
    print('这才load_data完')

    # dygnn模块 用数据训练模型
    our_model = MetaDyGNN(config, new_func(train_ngh_finder), n_feat, e_feat, model_name)

    print('--------------- {} ---------------'.format(model_name))

    if not load_model:
        # Load training dataset
        print('loading train data...')
        print('Begin to training...')
        # print('loading warm data...')
        # warm_data = data_helper.load_data(data_set=data_set, state='warm_up',load_from_file=True)
        training(our_model, model_save=True, model_file=model_filename)
        # testing(our_model)
    else:
        trained_state_dict = torch.load(model_filename)
        our_model.load_state_dict(trained_state_dict)

    # testing
    print('Begin to testing...')
    testing(our_model)
    print('--------------- {} ---------------'.format(model_name))

