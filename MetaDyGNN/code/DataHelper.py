# coding: utf-8
# author: wcc
# create date: 2021-01-10 20:35

import numpy as np
import pandas as pd
import random

# 获取邻居节点类
class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """
        # 节点编号 时间戳 边编号 偏移量
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l

        self.uniform = uniform

    # 设置偏移量 什么偏移量？？
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]

        for i in range(len(adj_list)):
            curr = adj_list[i] # （节点，索引号，时间戳）
            curr = sorted(curr, key=lambda x: x[1]) # 按索引号排序
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
            off_set_l.append(len(n_idx_l))

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))
        
        # 返回节点 时间戳 边索引号 偏移量
        return n_idx_l, n_ts_l, e_idx_l, off_set_l
    
    # 被 get_temporal_neighbor() 调用 本文件没有调用
    def find_before(self, src_idx, cut_time):
        """

        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1

        # 二分法
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]

    # 寻找时间邻居 被 find_k_hop() 调用 本文件没有调用
    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]

                    assert (len(ngh_idx) <= num_neighbors)
                    assert (len(ngh_ts) <= num_neighbors)
                    assert (len(ngh_eidx) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    # 寻找k跳节点 本文件没有调用
    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k - 1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1]  # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est,
                                                                                                 ngn_t_est,
                                                                                                 num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors)  # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records


class DataHelper:
    # 构造函数
    # self 代表类的实例，self 在定义类的方法时是必须有的，即使在调用时不必传入相应的参数
    # 类似于C++显示提供this指针
    def __init__(self, dataset, k_shots=10, full_node=False, model_name='time_interval_update'):

        # 动态边集合（四元式形式） 边特征 节点特征
        self.g_df = pd.read_csv('../data/{0}/ml_{0}.csv'.format(dataset))
        self.e_feat = np.load('../data/{0}/ml_{0}.npy'.format(dataset))
        self.n_feat = np.load('../data/{0}/ml_{0}_node.npy'.format(dataset))

        self.dataset = dataset

        if dataset == 'dblp':
            # dblp [0.8846149999999999, 0.923077, 0.9615379999999999, 1.0]
            # set() 函数创建一个无序不重复元素集 把csv文件的ts列元素拿出来
            # sorted 函数对所有可迭代的对象进行排序操作 升序 -2 -3没看懂啥意思 暂不需要
            self.test_time = sorted(set(self.g_df.ts))[-2]
            self.val_time = sorted(set(self.g_df.ts))[-3]
            # print(type(self.test_time))
            # print(type(self.val_time))
        else:
            # 序列赋值 设置0.6和0.8分位数 前60%训练集 60%-80%为验证集 后20%测试集
            self.val_time, self.test_time = np.quantile(self.g_df.ts, [0.6, 0.8])
            # print('分位数')
            # print(type(self.test_time)) <class 'numpy.float64'>
            # print(type(self.val_time)) <class 'numpy.float64'>
            # print('self.g_df.ts', max(self.g_df.ts)) 2678373
            # print('self.test_time', self.test_time) 2092163.8000000003
            # print('self.val_time', self.val_time) 1591129.8

        # 记录一系列变量
        self.full_node = full_node # 传入的是true
        self.NEG_SAMPLING_POWER = 0.75 # 负采样次幂
        self.neg_table_size = int(1e6) # 1000000
        self.node_set = set()
        self.degrees = dict()
        self.k_shots = k_shots
        self.node2hist = dict()
        self.data_size = 0  # number of edges, undirected x2


        # 数据集由 源节点 目的节点 索引号 时间戳 四元式表示
        if dataset == 'dblp':
            src_l = self.g_df.src.values
            dst_l = self.g_df.dst.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values
        else:
            src_l = self.g_df.u.values
            dst_l = self.g_df.i.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values

        print('max node: ', max(max(dst_l), max(src_l))) # 9227

        random.seed(2021)

        # union() 方法返回两个集合的并集，即包含了所有集合的元素，重复的元素只会出现一次
        # 总节点集
        self.node_set = set(src_l).union(set(dst_l))

        # 时间戳大于val_time的源节点和目的节点求并集 与小于该时间戳的并集做差
        # 只出现在val_time之后的节点 验证集+测试集节点个数
        self.mask_node_set = (set(src_l[ts_l > self.val_time]).union(set(dst_l[ts_l > self.val_time]))
                              - set(src_l[ts_l <= self.val_time]).union(set(dst_l[ts_l <= self.val_time])))

        # 只出现在test_time之后的节点 即测试集节点个数
        self.test_mask_node_set = (set(src_l[ts_l > self.test_time]).union(set(dst_l[ts_l > self.test_time]))
                                   - set(src_l[ts_l <= self.test_time]).union(set(dst_l[ts_l <= self.test_time])))
        
        # 二者差值为 仅出现在[val_time, test_time]之间的节点数量 即为验证集节点个数
        self.valid_mask_node_set = self.mask_node_set - self.test_mask_node_set
        # print('self.node_set', len(self.node_set))
        # print('self.mask_node_set', len(self.mask_node_set))
        # print('self.test_mask_node_set', len(self.test_mask_node_set))
        # print('self.valid_mask_node_set', len(self.valid_mask_node_set))
        # self.node_set 9227
        # self.mask_node_set 2388
        # 训练集(未标注) 9227 - 2388 = 6839
        # 验证集 self.test_mask_node_set 1220
        # 测试集 self.valid_mask_node_set 1168

        # 这一步可不看 后面也没用到
        if dataset == 'dblp':
            mask_src_flag = self.g_df.src.map(
                lambda x: x in self.mask_node_set).values  # array([False, False, False, ..., False,  True, False])
            mask_dst_flag = self.g_df.dst.map(
                lambda x: x in self.mask_node_set).values  # array([False, False, False, ..., False,  True, False])
        else:
            # map 返回的则是 True 和 False 组成的迭代器
            mask_src_flag = self.g_df.u.map(
                lambda x: x in self.mask_node_set).values  # array([False, False, False, ..., False,  True, False])
            mask_dst_flag = self.g_df.i.map(
                lambda x: x in self.mask_node_set).values  # array([False, False, False, ..., False,  True, False])
            # print('mask_src_flag', mask_src_flag)
            # print(mask_src_flag.shape)
            # print('mask_dst_flag', mask_dst_flag)
            # print(mask_dst_flag.shape)
            # mask_src_flag [False False False ... False  True False]
            # (157474,)
            # mask_dst_flag [False False False ... False False False]
            # (157474,)

        # meta training set edges flags 
        # 元训练集打flag val时间之前均为true 之后为false
        self.valid_train_edges_flag = (ts_l <= self.val_time)
        # print('self.valid_train_edges_flag', self.valid_train_edges_flag)
        # print(type(self.valid_train_edges_flag))
        # print(self.valid_train_edges_flag.shape)
        # self.valid_train_edges_flag [ True  True  True ... False False False]
        # <class 'numpy.ndarray'>
        # (157474,)

        # meta validation set nodes
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约内存
        # [val, tset]之间均为true 两头为false
        self.valid_val_flag = np.array(
            [(a in self.valid_mask_node_set or b in self.valid_mask_node_set) for a, b in zip(src_l, dst_l)])
        
        # test之后的为true 之前为false
        # meta testing set nodes
        self.valid_test_flag = np.array(
            [(a in self.test_mask_node_set or b in self.test_mask_node_set) for a, b in zip(src_l, dst_l)])

        # print('self.valid_val_flag', self.valid_val_flag)
        # print(self.valid_val_flag.shape)
        # print(type(self.valid_val_flag))
        # print('self.valid_test_flag', self.valid_test_flag)
        # print(self.valid_test_flag.shape)
        # print(type(self.valid_test_flag))
        # self.valid_val_flag [False False False ... False False False]
        # (157474,)
        # <class 'numpy.ndarray'>
        # self.valid_test_flag [False False False ... False  True False]
        # (157474,)
        # <class 'numpy.ndarray'>

        # 元训练集数据
        # meta training all edges (include support set and query set) for calculate the distribution of the node degree
        train_src_l = src_l[self.valid_train_edges_flag]
        train_dst_l = dst_l[self.valid_train_edges_flag]
        train_ts_l = ts_l[self.valid_train_edges_flag]
        # print(train_src_l.shape, train_dst_l.shape, train_ts_l.shape) # (94484,)

        # 将训练集(一共157475前面的94484)的 (u, i, timestamp) 打包成三元组 计算节点度数
        for src, dst, ts in zip(train_src_l, train_dst_l, train_ts_l):
            
            # self.degrees是 节点序号->度数
            if src not in self.degrees:
                self.degrees[src] = 0
            if dst not in self.degrees:
                self.degrees[dst] = 0

            # self.node2hist是 节点序号->邻居节点序号+时间戳
            if src not in self.node2hist:
                self.node2hist[src] = list()
            if dst not in self.node2hist:
                self.node2hist[dst] = list()

            # 不严格区分出度 和入度
            self.degrees[src] += 1
            self.degrees[dst] += 1

            # 源节点、目的节点都添加
            self.node2hist[src].append((dst, ts))
            self.node2hist[dst].append((src, ts))

        # 验证集和测试集的节点度数置为0
        for node in self.mask_node_set:
            self.degrees[node] = 0

        # print('len(self.degrees)', len(self.degrees)) 9227
        # print('len(self.node_set)', len(self.node_set)) 9227
        # print(self.degrees)
        # print('len(self.degrees)', len(self.degrees)) 9227
        # print(self.node2hist)
        # print('len(self.node2hist)', len(self.node2hist)) 6839
        # print(sum(list(self.degrees.values()))) 188968

        self.node_dim = len(self.node_set) # 节点总个数 9227
        self.neg_table = np.zeros((self.neg_table_size,)) # 负采样表 array([0,0,...,0]) 10e6列
        # print(self.neg_table)
        # print('len(self.neg_table)', len(self.neg_table))

        # 负采样初始化表 定义在下面
        self.init_neg_table()
        # print(self.neg_table) [1.000e+00 1.000e+00 1.000e+00 ... 9.152e+03 9.152e+03 9.152e+03]
        # print('len(self.neg_table)', len(self.neg_table)) len(self.neg_table) 1000000
        # print('Initialization over')

    # 数据加载函数 1 wikipedia未调用
    def load_data(self):

        if self.dataset == 'dblp':
            src_l = self.g_df.src.values
            dst_l = self.g_df.dst.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values
        else:
            src_l = self.g_df.u.values
            dst_l = self.g_df.i.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values

        # 1.元训练阶段所有数据
        # meta training all edges (include support set and query set)
        train_src_l = src_l[self.valid_train_edges_flag]
        train_dst_l = dst_l[self.valid_train_edges_flag]
        train_ts_l = ts_l[self.valid_train_edges_flag]
        train_e_idx_l = e_idx_l[self.valid_train_edges_flag]

        # check the set partition is right
        # 总节点集 - 只出现在val_time以后的节点数量 == 训练集的数量
        train_node_set = set(train_src_l).union(train_dst_l)
        assert (len(self.node_set - self.mask_node_set) == len(train_node_set))

        # 2.验证集
        # sample for validation set
        valid_src_l = src_l[self.valid_val_flag]
        valid_dst_l = dst_l[self.valid_val_flag]
        valid_ts_l = ts_l[self.valid_val_flag]
        valid_e_idx_l = e_idx_l[self.valid_val_flag]

        # 3.测试集
        # meta testing all edges (include support set and query set)
        test_src_l = src_l[self.valid_test_flag]
        test_dst_l = dst_l[self.valid_test_flag]
        test_ts_l = ts_l[self.valid_test_flag]
        test_e_idx_l = e_idx_l[self.valid_test_flag]

        # 打包验证集三元组
        for src, dst, ts in zip(valid_src_l, valid_dst_l, valid_ts_l):
            
            # 更新self.degrees self.node2hist
            if src in self.mask_node_set:

                if src not in self.node2hist:
                    self.node2hist[src] = list()
                self.node2hist[src].append((dst, ts))
                self.degrees[src] += 1

            if dst in self.mask_node_set:

                if dst not in self.node2hist:
                    self.node2hist[dst] = list()
                self.node2hist[dst].append((src, ts))
                self.degrees[dst] += 1

        # 打包测试集三元组
        for src, dst, ts in zip(test_src_l, test_dst_l, test_ts_l):

            # 更新self.degrees self.node2hist
            if src in self.mask_node_set:

                if src not in self.node2hist:
                    self.node2hist[src] = list()
                self.node2hist[src].append((dst, ts))
                self.degrees[src] += 1

            if dst in self.mask_node_set:

                if dst not in self.node2hist:
                    self.node2hist[dst] = list()
                self.node2hist[dst].append((src, ts))
                self.degrees[dst] += 1

        # ↑ prepared for sample task

        # 对训练节点集进行采样 分成支持集和查询集
        # sample train task
        train_support_x, train_support_y, train_query_x, train_query_y = [], [], [], []

        for node in train_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

            # 度小于4不考虑
            if self.degrees[node] < 4:
                continue

            # if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:
            # 邻居节点数能够采样出 k shot时
            if len(self.node2hist[node]) >= self.k_shots:
                
                # 支持集正、负节点采样
                pos = random.sample(self.node2hist[node], self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1) 按时间升序排序
                for i in range(len(pos) // 2):
                    support_y.append(1)
                    query_y.append(1)
                
                # 负采样函数 定义在下面
                neg = self.negative_sampling(int(self.k_shots // 2), pos[int(self.k_shots // 2) - 1][1])
                for i in range(len(neg)):
                    support_y.append(0)

                target = pos[:int(self.k_shots / 2)] + neg

                for i in target:
                    support_x.append([node] + list(i))

                train_support_x.append(support_x)
                train_support_y.append(support_y)

                # 查询集正、负节点采样
                neg = self.negative_sampling(int(self.k_shots / 2), pos[self.k_shots - 1][1])
                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[-int(self.k_shots / 2):] + neg
                for i in target:
                    query_x.append([node] + list(i))

                train_query_x.append(query_x)
                train_query_y.append(query_y)

        # train_support_x = np.array(train_support_x)
        # train_support_y = np.array(train_support_y)
        # train_query_x = np.array(train_query_x)
        # train_query_y = np.array(train_query_y)
        print('train_support_x', len(train_support_x))
        print('train_support_y', len(train_support_y))
        print('train_query_x', len(train_query_x))
        print('train_query_y', len(train_query_y))

        train_data = list(zip(train_support_x, train_support_y, train_query_x, train_query_y))

        # 对验证集采样 分成支持集和查询集
        # sample validation task
        valid_support_x, valid_support_y, valid_query_x, valid_query_y = [], [], [], []

        for node in self.valid_mask_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

            if self.degrees[node] < 4:
                continue

            # if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:
            if len(self.node2hist[node]) >= self.k_shots:

                pos = random.sample(self.node2hist[node], self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)
                for i in range(len(pos) // 2):
                    support_y.append(1)
                    query_y.append(1)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[int(self.k_shots / 2) - 1][1])
                for i in range(len(neg)):
                    support_y.append(0)

                target = pos[:int(self.k_shots / 2)] + neg

                for i in target:
                    support_x.append([node] + list(i))

                valid_support_x.append(support_x)
                valid_support_y.append(support_y)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[self.k_shots - 1][1])
                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[-int(self.k_shots / 2):] + neg
                for i in target:
                    query_x.append([node] + list(i))

                valid_query_x.append(query_x)
                valid_query_y.append(query_y)

        # valid_support_x = np.array(valid_support_x)
        # valid_support_y = np.array(valid_support_y)
        # valid_query_x = np.array(valid_query_x)
        # valid_query_y = np.array(valid_query_y)
        print('valid_support_x', len(valid_support_x))
        print('valid_support_y', len(valid_support_y))
        print('valid_query_x', len(valid_query_x))
        print('valid_query_y', len(valid_query_y))

        valid_data = list(zip(valid_support_x, valid_support_y, valid_query_x, valid_query_y))

        # 对测试集采样 分成支持集和查询集
        # sample test task
        test_support_x, test_support_y, test_query_x, test_query_y = [], [], [], []

        for node in self.test_mask_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

            if self.degrees[node] < 4:
                continue

            if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:

                pos = random.sample(self.node2hist[node], self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)
                for i in range(len(pos) // 2):
                    support_y.append(1)
                    query_y.append(1)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[int(self.k_shots / 2) - 1][1])
                for i in range(len(neg)):
                    support_y.append(0)

                target = pos[:int(self.k_shots / 2)] + neg

                for i in target:
                    support_x.append([node] + list(i))

                test_support_x.append(support_x)
                test_support_y.append(support_y)

                neg = self.negative_sampling(int(self.k_shots / 2), pos[self.k_shots - 1][1])
                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[-int(self.k_shots / 2):] + neg
                for i in target:
                    query_x.append([node] + list(i))

                test_query_x.append(query_x)
                test_query_y.append(query_y)

            elif len(self.node2hist[node]) >= self.k_shots and self.full_node is True:

                pos = sorted(self.node2hist[node], key=lambda x: x[1])  # from past(0) to now(1)
                for i in range(self.k_shots // 2):
                    support_y.append(1)
                for i in range(len(pos)-self.k_shots//2):
                    query_y.append(1)

                neg = self.negative_sampling(self.k_shots // 2, pos[int(self.k_shots / 2) - 1][1])
                for i in range(len(neg)):
                    support_y.append(0)

                target = pos[:int(self.k_shots / 2)] + neg

                for i in target:
                    support_x.append([node] + list(i))

                test_support_x.append(support_x)
                test_support_y.append(support_y)

                neg = self.negative_sampling(len(pos)-self.k_shots//2, pos[-1][1])
                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[-(len(pos)-self.k_shots//2):] + neg
                for i in target:
                    query_x.append([node] + list(i))

                test_query_x.append(query_x)
                test_query_y.append(query_y)

        # test_support_x = np.array(test_support_x)
        # test_support_y = np.array(test_support_y)
        # test_query_x = np.array(test_query_x)
        # test_query_y = np.array(test_query_y)
        print('test_support_x', len(test_support_x))
        print('test_support_y', len(test_support_y))
        print('test_query_x', len(test_query_x))
        print('test_query_y', len(test_query_y))

        test_data = list(zip(test_support_x, test_support_y, test_query_x, test_query_y))

        max_idx = len(self.node_set)

        # 全部节点的邻居关系集合
        full_adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            full_adj_list[src].append((dst, eidx, ts))
            full_adj_list[dst].append((src, eidx, ts))
        
        full_ngh_finder = NeighborFinder(full_adj_list, uniform=True)
        
        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
            adj_list[src].append((dst, eidx, ts))
            adj_list[dst].append((src, eidx, ts))

        # 仅用作训练的节点的关系集合
        train_ngh_finder = NeighborFinder(adj_list, uniform=False)
        print('is here???')
        print("train_set: ", len(train_data))
        print("valid_set: ", len(valid_data))
        print("test_set: ", len(test_data))
        print('really???')

        return train_data, valid_data, test_data, full_ngh_finder, train_ngh_finder

    # 数据加载函数 2
    def load_data_in_time_sp(self, interval=4):

        if self.dataset == 'dblp':
            src_l = self.g_df.src.values
            dst_l = self.g_df.dst.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values
        else:
            src_l = self.g_df.u.values
            dst_l = self.g_df.i.values
            e_idx_l = self.g_df.idx.values
            ts_l = self.g_df.ts.values

        # meta training all edges (include support set and query set)
        train_src_l = src_l[self.valid_train_edges_flag]
        train_dst_l = dst_l[self.valid_train_edges_flag]
        train_ts_l = ts_l[self.valid_train_edges_flag]
        train_e_idx_l = e_idx_l[self.valid_train_edges_flag]

        # check the set partition is right
        # 总 - val之后的数据 = 训练数据
        train_node_set = set(train_src_l).union(train_dst_l)
        assert (len(self.node_set - self.mask_node_set) == len(train_node_set))
        # print('train_node_set len is ', len(train_node_set)) 6839

        # sample for validation set
        valid_src_l = src_l[self.valid_val_flag]
        valid_dst_l = dst_l[self.valid_val_flag]
        valid_ts_l = ts_l[self.valid_val_flag]
        valid_e_idx_l = e_idx_l[self.valid_val_flag]

        # meta testing all edges (include support set and query set)
        test_src_l = src_l[self.valid_test_flag]
        test_dst_l = dst_l[self.valid_test_flag]
        test_ts_l = ts_l[self.valid_test_flag]
        test_e_idx_l = e_idx_l[self.valid_test_flag]

        # 计算验证、测试集的self.node2hist和self.degrees
        for src, dst, ts in zip(valid_src_l, valid_dst_l, valid_ts_l):

            if src in self.mask_node_set:

                if src not in self.node2hist:
                    self.node2hist[src] = list()
                self.node2hist[src].append((dst, ts))
                self.degrees[src] += 1

            if dst in self.mask_node_set:

                if dst not in self.node2hist:
                    self.node2hist[dst] = list()
                self.node2hist[dst].append((src, ts))
                self.degrees[dst] += 1

        for src, dst, ts in zip(test_src_l, test_dst_l, test_ts_l):

            if src in self.mask_node_set:

                if src not in self.node2hist:
                    self.node2hist[src] = list()
                self.node2hist[src].append((dst, ts))
                self.degrees[src] += 1

            if dst in self.mask_node_set:

                if dst not in self.node2hist:
                    self.node2hist[dst] = list()
                self.node2hist[dst].append((src, ts))
                self.degrees[dst] += 1

        # ↑ prepared for sample task

        # sample train task
        train_support_x, train_support_y, train_query_x, train_query_y = [], [], [], []

        # 按照元学习的策略，训练集的数据，还要分为多组支持集和查询集
        for node in train_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

#             if self.degrees[node] < 4:
#                 continue

            # if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:
            # 当前节点的邻居节点数量 大于等于二倍的k shots时
            if len(self.node2hist[node]) >= 2 * self.k_shots:
                
                # 随机采样2k个邻居节点 按时间排序
                pos = random.sample(self.node2hist[node], 2 * self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)
                # print('pos type is ', type(pos)) list
                # print('len(pos)', len(pos)) 12
                # print(pos) pos是由 12个(邻居节点, 链接形成时间) 组成的list


                # 元训练阶段 采样support set
                for i in range(interval):

                    x, y = [], []

                    # 每个任务是k shot 被分散在每个interval中 每个任务的支持集的边数是pos_keys
                    # 由config中的参数决定
                    pos_keys = self.k_shots // interval
                    # print('self.k_shots', self.k_shots) 6
                    # print('interval', interval) 2
                    # print('pos_keys', pos_keys) 3

                    for j in range(pos_keys):
                        y.append(1) # 3 个“标签1” 插入y

                    neg = self.negative_sampling(pos_keys, pos[(i + 1) * pos_keys - 1][1])
                    # print('neg type is ', type(neg))
                    # print('len(neg)', len(neg))
                    # print(neg)
                    # neg type is  <class 'list'>
                    # len(neg) 3
                    # [(8886.0, 1551437.0), (1428.0, 1551437.0), (8959.0, 1551437.0)]

                    for j in range(len(neg)):
                        y.append(0) # 3个 “标签0” 插入y
                    
                    # 将pos的前[0:3] [3:6]分别提出 和neg组合 拼成长度为6的list
                    
                    target = pos[i * pos_keys:(i + 1) * pos_keys] + neg
                    # print('target type is ', type(target)) list
                    # print('len(target)', len(target)) 6
                    # print(target)
                    # [(5519, 1539404.0), (5519, 1540743.0), (5519, 1541103.0), (3203.0, 1541103.0), (8439.0, 1541103.0), (8580.0, 1541103.0)]

                    for j in target:
                        # 当前节点node为源点 插入target（目的点，时间戳） 形成长度为6的三元组list
                        x.append([node] + list(j))
                        # [[9147, 5792, 1542641.0], [9147, 5792, 1542694.0], [9147, 5792, 1542747.0], [9147, 719.0, 1542747.0], [9147, 1708.0, 1542747.0], [9147, 9132.0, 1542747.0]]

                    # 针对某一node 在该interval内 已划分好支持集的x y
                    support_x.append(x)
                    support_y.append(y)
                
                # 针对训练集中的所有node 已划分好支持集的x y
                train_support_x.append(support_x)
                train_support_y.append(support_y)


                # 元训练阶段 采样query set
                for i in range(len(pos[self.k_shots:])):
                    query_y.append(1)
                # print('len is ', len(pos[self.k_shots:])) 6
                # print(pos[self.k_shots:]) 是pos中6-12位

                # 以pos中最后一个元素的time为截止时间 随机生成neg
                neg = self.negative_sampling(len(pos[self.k_shots:]), pos[-1][1])

                for i in range(len(neg)):
                    query_y.append(0)

                # target由pos中剩下的元素组成
                target = pos[self.k_shots:] + neg
                for i in target:
                    query_x.append([node] + list(i))

                train_query_x.append(query_x)
                train_query_y.append(query_y)

        # train_support_x = np.array(train_support_x)
        # train_support_y = np.array(train_support_y)
        # train_query_x = np.array(train_query_x)
        # train_query_y = np.array(train_query_y)
        # print('train_support_x len is ', len(train_support_x)) 2093
        # print('train_support_y len is ', len(train_support_y)) 2093
        # print('train_query_x len is ', len(train_query_x)) 2093
        # print('train_query_y len is ', len(train_query_y)) 2093

        train_data = list(zip(train_support_x, train_support_y, train_query_x, train_query_y))
        # 仅出现在训练集的节点一共6839种，但是节点度数大于 2*k 的只有2093种 验证集 测试集 同理
        # print('train_data SIZE', len(train_data)) 2093
        # print('train_data', train_data)
        # ([[[9149, 5889, 1575013.0], [9149, 5889, 1575136.0], [9149, 5889, 1575801.0], [9149, 4943.0, 1575801.0], [9149, 8937.0, 1575801.0], [9149, 8538.0, 1575801.0]], [[9149, 5889, 1576136.0], [9149, 5889, 1576403.0], [9149, 5889, 1576637.0], [9149, 904.0, 1576637.0], [9149, 8618.0, 1576637.0], [9149, 1443.0, 1576637.0]]], [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]], [[9149, 5889, 1576755.0], [9149, 5889, 1577003.0], [9149, 5889, 1577306.0], [9149, 5889, 1579011.0], [9149, 5889, 1579380.0], [9149, 5889, 1579738.0], [9149, 3135.0, 1579738.0], [9149, 9004.0, 1579738.0], [9149, 8986.0, 1579738.0], [9149, 179.0, 1579738.0], [9149, 4552.0, 1579738.0], [9149, 4988.0, 1579738.0]], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])]

        # sample validation task 与测试集思路完全相同
        valid_support_x, valid_support_y, valid_query_x, valid_query_y = [], [], [], []

        # print('mask_node_set:', len(mask_node_set))
        for node in self.valid_mask_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

#             if self.degrees[node] < 4:
#                 continue

            # if len(self.node2hist[node]) >= self.k_shots and self.full_node is False:
            if len(self.node2hist[node]) >= 2 * self.k_shots:

                pos = random.sample(self.node2hist[node], 2 * self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)

                for i in range(interval):

                    x, y = [], []

                    pos_keys = self.k_shots // interval

                    for j in range(pos_keys):
                        y.append(1)

                    neg = self.negative_sampling(pos_keys, pos[(i + 1) * pos_keys - 1][1])
                    for j in range(len(neg)):
                        y.append(0)

                    target = pos[i * pos_keys:(i + 1) * pos_keys] + neg

                    for j in target:
                        x.append([node] + list(j))

                    support_x.append(x)
                    support_y.append(y)

                valid_support_x.append(support_x)
                valid_support_y.append(support_y)

                for i in range(len(pos[self.k_shots:])):
                    query_y.append(1)

                neg = self.negative_sampling(len(pos[self.k_shots:]), pos[-1][1])

                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[self.k_shots:] + neg
                for i in target:
                    query_x.append([node] + list(i))

                valid_query_x.append(query_x)
                valid_query_y.append(query_y)

        # test_support_x = np.array(test_support_x)
        # test_support_y = np.array(test_support_y)
        # test_query_x = np.array(test_query_x)
        # test_query_y = np.array(test_query_y)

        # 验证集的数据已准备完毕
        valid_data = list(zip(valid_support_x, valid_support_y, valid_query_x, valid_query_y))

        # sample test task
        test_support_x, test_support_y, test_query_x, test_query_y = [], [], [], []

        # print('mask_node_set:', len(mask_node_set))
        for node in self.test_mask_node_set:

            support_x, support_y, query_x, query_y = [], [], [], []

#             if self.degrees[node] < 4:
#                 continue

            if len(self.node2hist[node]) >= 2 * self.k_shots and self.full_node is False:
                
                pos = random.sample(self.node2hist[node], 2*self.k_shots)
                pos = sorted(pos, key=lambda x: x[1])  # from past(0) to now(1)

                for i in range(interval):

                    x, y = [], []

                    pos_keys = self.k_shots // interval

                    for j in range(pos_keys):
                        y.append(1)

                    neg = self.negative_sampling(pos_keys, pos[(i + 1) * pos_keys - 1][1])
                    for j in range(len(neg)):
                        y.append(0)

                    target = pos[i * pos_keys:(i + 1) * pos_keys] + neg

                    for j in target:
                        x.append([node] + list(j))

                    support_x.append(x)
                    support_y.append(y)

                test_support_x.append(support_x)
                test_support_y.append(support_y)

                for i in range(len(pos[self.k_shots:])):
                    query_y.append(1)

                neg = self.negative_sampling(len(pos[self.k_shots:]), pos[-1][1])

                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[interval * pos_keys:] + neg
                for i in target:
                    query_x.append([node] + list(i))

                test_query_x.append(query_x)
                test_query_y.append(query_y)

            # 实际运行了该分支
            elif len(self.node2hist[node]) >= 2 * self.k_shots and self.full_node is True:
                
                # pos = random.sample(self.node2hist[node], self.k_shots) 不再采样k个邻居节点

                # pos存放node所有邻居节点 并按照时间顺序排序 pos大小从12-179不等
                pos = sorted(self.node2hist[node], key=lambda x: x[1])  # from past(0) to now(1)
                # print('pos len is ', len(pos))
                # print(pos)

                for i in range(interval):

                    x, y = [], []

                    pos_keys = self.k_shots // interval

                    for j in range(pos_keys):
                        y.append(1)

                    neg = self.negative_sampling(pos_keys, pos[(i + 1) * pos_keys - 1][1])
                    for j in range(len(neg)):
                        y.append(0)

                    target = pos[i * pos_keys:(i + 1) * pos_keys] + neg
                    # print('target size is ', len(target)) 6
                    
                    for j in target:
                        x.append([node] + list(j))

                    support_x.append(x)
                    support_y.append(y)

                test_support_x.append(support_x)
                test_support_y.append(support_y)

                # 除了前k个外 node所有的邻接节点都是query
                for i in range(len(pos[self.k_shots:])):
                    query_y.append(1)

                neg = self.negative_sampling(len(pos[self.k_shots:]), pos[-1][1])

                for i in range(len(neg)):
                    query_y.append(0)

                target = pos[self.k_shots:] + neg
                for i in target:
                    query_x.append([node] + list(i))

                test_query_x.append(query_x)
                test_query_y.append(query_y)

        # test_support_x = np.array(test_support_x)
        # test_support_y = np.array(test_support_y)
        # test_query_x = np.array(test_query_x)
        # test_query_y = np.array(test_query_y)

        # 测试集 数据已准备完毕
        test_data = list(zip(test_support_x, test_support_y, test_query_x, test_query_y))
        # print(test_data)

        max_idx = len(self.node_set) # 9227 最大的节点编号

        full_adj_list = [[] for _ in range(max_idx + 1)]
        # 总的edge集合
        for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            full_adj_list[src].append((dst, eidx, ts))
            full_adj_list[dst].append((src, eidx, ts))
        # print('full_adj_list type is ', type(full_adj_list)) <class 'list'>
        # print('full_adj_list len is ', len(full_adj_list)) 9228
        # # print(full_adj_list)

        full_ngh_finder = NeighborFinder(full_adj_list, uniform=True) # true 是啥
        # print('full_ngh_finder type is ', type(full_ngh_finder))
        # print('full_ngh_finder len is ', len(full_ngh_finder))
        # print(full_ngh_finder)

        adj_list = [[] for _ in range(max_idx + 1)]
        # 元训练阶段的edge集合 对每个node记录下所有的邻接节点
        for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
            adj_list[src].append((dst, eidx, ts))
            adj_list[dst].append((src, eidx, ts))
        # print('adj_list type is ', type(adj_list)) <class 'list'>
        # print('adj_list len is ', len(adj_list)) 9228
        # print(adj_list)

        train_ngh_finder = NeighborFinder(adj_list, uniform=False) # false 是啥

        print("train_set: ", len(train_data)) # 2093
        print("valid_set: ", len(valid_data)) # 148
        print("test_set: ", len(test_data)) # 73

        return train_data, valid_data, test_data, full_ngh_finder, train_ngh_finder

    # 返回节点的度
    def get_node_dim(self):
        return self.node_dim

    # 负采样表初始化函数
    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 1
        # tot_sum记录了 1-9228每个节点度数的0.75次方
        for k in range(1, self.node_dim + 1):
            # print('k', k)
            # print('self.degrees[k]', self.degrees[k])
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        # print('tot_sum', tot_sum)
        
        # 计算当前节点度占总体的比例？更新self.neg_table
        for k in range(self.neg_table_size):
            #             print('n_id', n_id)
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    # 负采样函数 形参是要得到的 负样本数量 和截止时间
    def negative_sampling(self, neg_size, cut_time):
        sampled_negs = []
        # 输出 [0,self.neg_table_size] 之间的随机数 返回是neg_size行 1列的数组
        rand_idx = np.random.randint(0, self.neg_table_size, (neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        for i in sampled_nodes:
            sampled_negs.append((i, cut_time))
        # 返回值是（随机生成的节点编号，传入的截止时间）
        return sampled_negs

