import json
import numpy as np
import pandas as pd
import torch

# 预处理文件 将数据和特征分离
def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    # 该打开的文件记为f
    with open(data_name) as f:
        # 打印出文件每列的表头信息
        s = next(f)
        print(s)

        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
        # 同时列出数据line 和数据下标idx，一般用在 for 循环当中
        for idx, line in enumerate(f):
            # print(idx)
            e = line.strip().split(',') # strip表示删除掉数据中的换行符 split则是数据中遇到‘,’ 就隔开
            u = int(e[0]) # user_id转存为u
            i = int(e[1]) # item_id转存为i

            ts = float(e[2]) # timestamp
            label = int(e[3]) # state_label

            feat = np.array([float(x) for x in e[4:]]) # 下标第四列开始的所有数据都是特征feat

            # 插入新建好的6个列表中
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    
    # 组织成df格式返回
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)

# 重新排索引，否则会重复
def reindex(df):
    # u和i的互不相同的元素数量
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    # 重排i的索引，偏移量为upper_u
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df = df.copy() # 深拷贝
    # 重排索引前 8226 999
    print('new_df.u.max()', new_df.u.max())
    print('new_df.i.max()', new_df.i.max())

    new_df.i = new_i # 偏移后的i作为新i
    new_df.u += 1 # 加一 让u的标号从1开始
    new_df.i += 1 # 同理
    new_df.idx += 1 # 同理

    # 重排索引后 8227 9227
    print('new_df.u.max()', new_df.u.max())
    print('new_df.i.max()', new_df.i.max())

    return new_df


def run(data_name):
    # 从csv文件中读取原数据，路径记为path，输出三个新文件
    # PATH = '../data/{0}/{0}.csv'.format(data_name)
    # OUT_DF = '../data/{0}/ml_{0}.csv'.format(data_name)
    # OUT_FEAT = '../data/{0}/ml_{0}.npy'.format(data_name)
    # OUT_NODE_FEAT = '../data/{0}/ml_{0}_node.npy'.format(data_name)

    PATH = '{0}.csv'.format(data_name)
    OUT_DF = 'ml_{0}.csv'.format(data_name)
    OUT_FEAT = 'ml_{0}.npy'.format(data_name)
    OUT_NODE_FEAT = 'ml_{0}_node.npy'.format(data_name)

    # 其他信息存到df、特征存到feat + 重排df的索引
    df, feat = preprocess(PATH)
    new_df = reindex(df)

    # print('feat.shape', feat.shape) # (157474, 172)
    empty = np.zeros(feat.shape[1])[np.newaxis, :] # 变为二维，从1行feat.shape[1]列到二维
    # print('empty.shape', empty.shape) # (1, 172)
    feat = np.vstack([empty, feat]) # 按列方向堆叠empty和feat
    # print('feat.shape', feat.shape) # (157474, 172)

    max_idx = max(new_df.u.max(), new_df.i.max()) # 9227

    # rand_feat = np.zeros((max_idx + 1, feat.shape[1]))

    rand_feat = torch.empty((max_idx + 1, feat.shape[1])) # 生成一个填充了随机数的9228*172的张量
    rand_feat = torch.nn.init.uniform_(rand_feat) # 归一化
    rand_feat = torch.nn.init.xavier_uniform_(rand_feat, gain=1.414) # 哈维 归一化

    print('new_df.shape', new_df.shape) # (157474, 172)
    print('feat.shape', feat.shape) # (157475, 172) 边特征是给定的
    print('rand_feat.shape', rand_feat.shape) # torch.Size([9228, 172]) 节点特征是随机生成的

    # new_feat = np.zeros(feat.shape)
    # print('new_feat.shape', new_feat.shape)

    # new_df转存为csv
    # feat和rand_feat转换为npy
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    # np.save(OUT_FEAT, new_feat)
    np.save(OUT_NODE_FEAT, rand_feat)
    print('File is prepared, exit.')

    return


run('wikipedia')

# run('reddit')

