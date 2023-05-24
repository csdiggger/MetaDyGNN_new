# coding: utf-8
# author: wcc
# create date: 2021-01-10 20:35

import numpy as np
import torch
from torch.nn import functional as F
from Evaluation import Evaluation
from MetaLearner import TGNN_Encoder, MetaLearner


class MetaDyGNN(torch.nn.Module):
    def __init__(self, config, ngh_finder, n_feat, e_feat, model_name):
        # 子类调用父类的方法
        super(MetaDyGNN, self).__init__()

        self.config = config
        self.use_cuda = self.config['use_cuda']
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.model_name = model_name

        self.dataset = self.config['dataset']

        # 调用元学习器模块
        self.emb_encoder = TGNN_Encoder(config, ngh_finder, n_feat, e_feat)
        self.classifier = MetaLearner(config)

        self.num_neighbors = self.config['num_neighbors']
        self.encoder_lr = config['encoder_lr']
        self.local_lr = config['local_lr']
        self.base_lr = config['base_lr']
        self.emb_dim = self.config['embedding_dim']

        # 评价函数
        self.cal_metrics = Evaluation()

        # 节点编码器和分类器的参数长度和名称
        self.enc_weight_len = len(self.emb_encoder.update_parameters())
        self.enc_weight_name = list(self.emb_encoder.update_parameters().keys())
        self.cl_weight_len = len(self.classifier.update_parameters())
        self.cl_weight_name = list(self.classifier.update_parameters().keys())
        self.criterion = torch.nn.BCELoss() # 二进制交叉熵损失函数

        # adam 优化器
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])
        
        # 调用转换函数
        self.transform_liners = self.transformer()

    # 对分类器参数实行维度变换 从第一参数变为第二个参数
    def transformer(self):

        liners = {}
        cl_par = self.classifier.update_parameters()

        for w in self.cl_weight_name:
            liners[w] = torch.nn.Linear(self.emb_dim, np.prod(cl_par[w].shape), bias=False)
        # 将子模型以排序字典形式保存并返回
        return torch.nn.ModuleDict(liners)

    # 全局参数更新 本类中没有调用 由main-training调用
    def global_update(self, support_x, support_y, query_x, query_y):
        # print('调用dygnn类中的全局更新')
        # 每个batch的任务大小是48
        task_size_s = len(support_x)
        # 损失 正确率 平均正确率 f1 auc
        loss_s, acc_s, ap_s, f1_s, auc_s = [], [], [], [], []

        for i in range(task_size_s):
            if self.model_name == 'by_MAML':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.by_MAML(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'no_finetune':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.no_finetune(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'time_interval_sp': # 执行该逻辑
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.time_MAML(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'fine_tune':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.no_finetune(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'time_interval_pool':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.time_MAML_pool(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'time_MAML_sp':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.time_MAML_sp(support_x[i], support_y[i], query_x[i], query_y[i])

            loss_s.append(_loss_s)
            acc_s.append(_acc_s)
            ap_s.append(_ap_s)
            f1_s.append(_f1_s)
            auc_s.append(_auc_s)

        loss = torch.stack(loss_s).mean(0)
        acc_s = np.mean(acc_s)
        ap_s = np.mean(ap_s)
        f1_s = np.mean(f1_s)
        auc_s = np.mean(auc_s)

        # optimizer.zero_grad()清除了优化器中所有梯度 ，在每次loss.backward()之前，不要忘记使用，否则之前的梯度将会累积，这通常不是我们所期望的(也不排除也有人需要利用这个功能)
        # loss.backward()故名思义，就是将损失loss向输入侧进行反向传播，同时对于需要进行梯度计算的所有变量计算梯度，并将其累积备用 
        # optimizer.step()是优化器对参数的梯度值进行更新
        self.meta_optimizer.zero_grad()
        loss.backward() # 公式（13）（14）
        self.meta_optimizer.step()

        # 阻断loss反向传播 数据移动至cpu 返回值为numpy
        return loss.cpu().detach().numpy(), acc_s, ap_s, f1_s, auc_s

    # 评价函数 本类中未调用 由validation testing调用 模型已固定 仅仅不更新loss
    def evaluate(self, support_x, support_y, query_x, query_y):

        task_size_s = len(support_x)
        loss_s, acc_s, ap_s, f1_s, auc_s = [], [], [], [], []
        instances = []

        for i in range(task_size_s):
            if self.model_name == 'by_MAML':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.by_MAML(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'no_finetune':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.no_finetune(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'time_interval_sp': # 执行
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.time_MAML(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'fine_tune':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.by_MAML(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'time_interval_pool':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.time_MAML_pool(support_x[i], support_y[i], query_x[i], query_y[i])
            elif self.model_name == 'time_MAML_sp':
                _loss_s, _acc_s, _ap_s, _f1_s, _auc_s \
                    = self.time_MAML_sp(support_x[i], support_y[i], query_x[i], query_y[i])

            instances.append(len(query_x[i]))
            acc_s.append(_acc_s)
            ap_s.append(_ap_s)
            f1_s.append(_f1_s)
            auc_s.append(_auc_s)

        acc = np.sum(np.array(acc_s)*np.array(instances))/np.sum(instances)
        f1 = np.mean(f1_s)

        return acc, ap_s, f1, auc_s

    # 时间元学习
    def time_MAML(self, support_x, support_y, query_x, query_y):
        # 编码器和分类器参数初始权重
        enc_initial_weights = self.emb_encoder.update_parameters()
        cl_initial_weights = self.classifier.update_parameters()

        # Φ ω 时间间隔内的loss 的定义
        interval_phi_weights, interval_omega_weights, interval_loss = {}, {}, {}

        interval = len(support_x)
        # print('interval is', interval) 2

        for k in range(interval):

            src_idx_l = np.array(support_x[k]).astype(int).transpose()[0]
            target_idx_l = np.array(support_x[k]).astype(int).transpose()[1]
            cut_time_l = np.array(support_x[k]).transpose()[2]

            # support set get grad
            # print('src_idx_l', src_idx_l)
            # print('target_idx_l', target_idx_l)

            # 第1次
            # 公式（2） 调用TGNN_encoder的forward （3）（4）已包含在内
            src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l, self.num_neighbors)
            # 公式（5） 调用metalearner的forward
            pred_sore = self.classifier(src_embed, target_embed).squeeze(dim=-1).sigmoid()
            # print('pred_sore', pred_sore)

            s_y = torch.tensor(support_y[k]).float().cuda()
            # print('s_y', s_y)
            # print('y_pred', pred_sore)
            loss = self.criterion(pred_sore, s_y) # 公式（6）
            # print('loss', loss)

            # 进入层级自适应元学习步骤
            # encoder fine-tune 编码器调优
            # encoder_lr为时间间隔自适应的学习率
            grad = torch.autograd.grad(loss, enc_initial_weights.values(), create_graph=True)
            enc_fast_weights = {}
            for i in range(self.enc_weight_len):
                weight_name = self.enc_weight_name[i]
                enc_fast_weights[weight_name] = enc_initial_weights[weight_name] - self.encoder_lr * grad[i] # 公式（9）

            # print('phi_update, step ', k)
            for idx in range(1, self.config['phi_update']):
                src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l,
                                                                   self.num_neighbors, vars_dict=enc_fast_weights) # 公式（7）
                pred_sore = self.classifier(src_embed, target_embed).squeeze(dim=-1).sigmoid()
                s_y = torch.tensor(support_y[k]).float().cuda()
                loss = self.criterion(pred_sore, s_y) # 公式（8）
                grad = torch.autograd.grad(loss, enc_fast_weights.values(), create_graph=True)

                for i in range(self.enc_weight_len):
                    weight_name = self.enc_weight_name[i]
                    enc_fast_weights[weight_name] = enc_fast_weights[weight_name] - self.encoder_lr * grad[i] # 公式（9）

            # classifier adaptation 分类器层次自适应
            # 第2次
            src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l,
                                                               self.num_neighbors, vars_dict=enc_fast_weights)

            pred_sore = self.classifier(src_embed, target_embed).squeeze(dim=-1).sigmoid()
            s_y = torch.tensor(support_y[k]).float().cuda()
            loss = self.criterion(pred_sore, s_y)

            # self.local_lr为节点自适应的学习率
            grad = torch.autograd.grad(loss, cl_initial_weights.values(), create_graph=True)
            cl_fast_weights = {}
            for i in range(self.cl_weight_len):
                weight_name = self.cl_weight_name[i]
                cl_fast_weights[weight_name] = cl_initial_weights[weight_name] - self.local_lr * grad[i] # 公式（11）

            # print('omega_update, step ', k)
            for idx in range(1, self.config['omega_update']):
                pred_sore = self.classifier(src_embed, target_embed, vars=cl_fast_weights).squeeze(dim=-1).sigmoid()
                s_y = torch.tensor(support_y[k]).float().cuda()
                loss = self.criterion(pred_sore, s_y) # 公式（8）？
                grad = torch.autograd.grad(loss, cl_fast_weights.values(), create_graph=True)

                for i in range(self.cl_weight_len):
                    weight_name = self.cl_weight_name[i]
                    cl_fast_weights[weight_name] = cl_fast_weights[weight_name] - self.local_lr * grad[i] # 公式（11）

            # query set for calculate att
            src_idx_l = np.array(query_x).astype(int).transpose()[0]
            target_idx_l = np.array(query_x).astype(int).transpose()[1]
            cut_time_l = np.array(query_x).transpose()[2]

            # 第3次
            src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l,
                                                               self.num_neighbors,
                                                               vars_dict=enc_fast_weights)

            pred_sore = self.classifier(src_embed, target_embed, vars_dict=cl_fast_weights).squeeze(dim=-1).sigmoid()

            q_y = torch.tensor(query_y).float().cuda()
            q_loss = self.criterion(pred_sore, q_y)

            interval_phi_weights[k] = enc_fast_weights
            interval_omega_weights[k] = cl_fast_weights
            interval_loss[k] = q_loss.data

        # aggregation parameters 调用聚合函数
        # 计算注意力系数 公式（12）
        interval_att = F.softmax(-torch.stack(list(interval_loss.values())), dim=0)
        phi = self.aggregator(interval_phi_weights, interval_att)
        omega = self.aggregator(interval_omega_weights, interval_att)

        # test on the query set 在query set的测试结果
        src_idx_l = np.array(query_x).astype(int).transpose()[0]
        target_idx_l = np.array(query_x).astype(int).transpose()[1]
        cut_time_l = np.array(query_x).transpose()[2]

        # 第7次 最后一次更新 phi 和 omega
        src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l,
                                                           self.num_neighbors,
                                                           vars_dict=phi)

        pred_sore = self.classifier(src_embed, target_embed, vars_dict=omega).squeeze(dim=-1).sigmoid()

        q_y = torch.tensor(query_y).float().cuda()
        loss = self.criterion(pred_sore, q_y)

        acc, ap, f1, auc = self.cal_metrics.prediction(np.array(query_y).astype(int),
                                                       pred_sore.cpu().detach().numpy())

        return loss, acc, ap, f1, auc

    def time_MAML_sp(self, support_x, support_y, query_x, query_y):

        enc_initial_weights = self.emb_encoder.update_parameters()
        cl_initial_weights = self.classifier.update_parameters()

        interval_phi_weights, interval_omega_weights, interval_loss = {}, {}, {}

        interval = len(support_x)

        src_idx_sup, target_idx_sup, cut_time_sup = [], [], []
        for k in range(interval):
            if k == 0:
                src_idx_sup = np.array(support_x[k]).astype(int).transpose()[0]
                target_idx_sup = np.array(support_x[k]).astype(int).transpose()[1]
                cut_time_sup = np.array(support_x[k]).transpose()[2]
            else:
                src_idx_sup = np.append(src_idx_sup, np.array(support_x[k]).astype(int).transpose()[0])
                target_idx_sup = np.array(support_x[k]).astype(int).transpose()[1]
                cut_time_sup = np.array(support_x[k]).transpose()[2]

        print(src_idx_sup)

        for k in range(interval):

            src_idx_l = np.array(support_x[k]).astype(int).transpose()[0]
            target_idx_l = np.array(support_x[k]).astype(int).transpose()[1]
            cut_time_l = np.array(support_x[k]).transpose()[2]

            # support set get grad
            # print('src_idx_l', src_idx_l)
            # print('target_idx_l', target_idx_l)
            src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l, self.num_neighbors)

            pred_sore = self.classifier(src_embed, target_embed).squeeze(dim=-1).sigmoid()
            # print('pred_sore', pred_sore)

            s_y = torch.tensor(support_y[k]).float().cuda()
            # print('s_y', s_y)
            # print('y_pred', pred_sore)
            loss = self.criterion(pred_sore, s_y)
            # print('loss', loss)
            # encoder fine-tune
            grad = torch.autograd.grad(loss, enc_initial_weights.values(), create_graph=True)
            enc_fast_weights = {}
            for i in range(self.enc_weight_len):
                weight_name = self.enc_weight_name[i]
                enc_fast_weights[weight_name] = enc_initial_weights[weight_name] - self.encoder_lr * grad[i]

            for idx in range(1, self.config['phi_update']):
                src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l,
                                                                   self.num_neighbors, vars_dict=enc_fast_weights)
                pred_sore = self.classifier(src_embed, target_embed).squeeze(dim=-1).sigmoid()
                s_y = torch.tensor(support_y[k]).float().cuda()
                loss = self.criterion(pred_sore, s_y)
                grad = torch.autograd.grad(loss, enc_fast_weights.values(), create_graph=True)

                for i in range(self.enc_weight_len):
                    weight_name = self.enc_weight_name[i]
                    enc_fast_weights[weight_name] = enc_fast_weights[weight_name] - self.encoder_lr * grad[i]

            # classifier adaptation
            src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l,
                                                               self.num_neighbors, vars_dict=enc_fast_weights)

            fast_weights = {}
            for w, liner in self.transform_liners.items():
                fast_weights[w] = cl_initial_weights[w] + liner(src_embed.mean(0)).view(cl_initial_weights[w].shape)

            pred_sore = self.classifier(src_embed, target_embed, vars=fast_weights).squeeze(dim=-1).sigmoid()
            s_y = torch.tensor(support_y[k]).float().cuda()
            loss = self.criterion(pred_sore, s_y)

            # grad = torch.autograd.grad(loss, cl_initial_weights.values(), create_graph=True)
            grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            cl_fast_weights = {}
            for i in range(self.cl_weight_len):
                weight_name = self.cl_weight_name[i]
                # cl_fast_weights[weight_name] = cl_initial_weights[weight_name] - self.local_lr * grad[i]
                cl_fast_weights[weight_name] = fast_weights[weight_name] - self.local_lr * grad[i]


            for idx in range(1, self.config['omega_update']):
                pred_sore = self.classifier(src_embed, target_embed, vars=cl_fast_weights).squeeze(dim=-1).sigmoid()
                s_y = torch.tensor(support_y[k]).float().cuda()
                loss = self.criterion(pred_sore, s_y)
                grad = torch.autograd.grad(loss, cl_fast_weights.values(), create_graph=True)

                for i in range(self.cl_weight_len):
                    weight_name = self.cl_weight_name[i]
                    cl_fast_weights[weight_name] = cl_fast_weights[weight_name] - self.local_lr * grad[i]

            # support set for calculate att
            # src_idx_l = np.array(query_x).astype(int).transpose()[0]
            # target_idx_l = np.array(query_x).astype(int).transpose()[1]
            # cut_time_l = np.array(query_x).transpose()[2]

            src_embed, target_embed = self.emb_encoder.forward(src_idx_sup, target_idx_sup, cut_time_sup,
                                                               self.num_neighbors,
                                                               vars_dict=enc_fast_weights)

            pred_sore = self.classifier(src_embed, target_embed, vars_dict=cl_fast_weights).squeeze(dim=-1).sigmoid()

            q_y = torch.tensor(query_y).float().cuda()
            q_loss = self.criterion(pred_sore, q_y)

            interval_phi_weights[k] = enc_fast_weights
            interval_omega_weights[k] = cl_fast_weights
            interval_loss[k] = q_loss.data

        # aggregation parameters
        interval_att = F.softmax(-torch.stack(list(interval_loss.values())), dim=0)
        phi = self.aggregator(interval_phi_weights, interval_att)
        omega = self.aggregator(interval_omega_weights, interval_att)

        # test on the query set
        src_idx_l = np.array(query_x).astype(int).transpose()[0]
        target_idx_l = np.array(query_x).astype(int).transpose()[1]
        cut_time_l = np.array(query_x).transpose()[2]

        src_embed, target_embed = self.emb_encoder.forward(src_idx_l, target_idx_l, cut_time_l,
                                                           self.num_neighbors,
                                                           vars_dict=phi)

        pred_sore = self.classifier(src_embed, target_embed, vars_dict=omega).squeeze(dim=-1).sigmoid()

        q_y = torch.tensor(query_y).float().cuda()
        loss = self.criterion(pred_sore, q_y)

        acc, ap, f1, auc = self.cal_metrics.prediction(np.array(query_y).astype(int),
                                                       pred_sore.cpu().detach().numpy())

        return loss, acc, ap, f1, auc

    def aggregator(self, task_weights_s, att):

        for idx in range(self.config['interval']):
            if idx == 0:
                att_task_weights = dict({k: v * att[idx] for k, v in task_weights_s[idx].items()})
                continue
            tmp_att_task_weights = dict({k: v * att[idx] for k, v in task_weights_s[idx].items()})

            att_task_weights = dict(zip(att_task_weights.keys(),
                                        list(map(lambda x: x[0] + x[1],
                                                 zip(att_task_weights.values(), tmp_att_task_weights.values())))))

        return att_task_weights



