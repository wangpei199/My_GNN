#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import networkx as nx 
import pgl 
from pgl import graph
from paddle.regularizer import L2Decay
import UniMP.my_sage as my_sage

###导入和预览数据
forw_edges = pd.read_csv('data/data61620/edges.csv',header = None, prefix='name_')

back_edges = forw_edges.copy(deep=True)
back_edges['name_0'], back_edges['name_1'] = forw_edges['name_1'], forw_edges['name_0']

edges = pd.concat([forw_edges, back_edges], ignore_index=True)
edges.head()


test_data = pd.read_csv('data/data61620/test.csv',header=0)
train_data = pd.read_csv('./data/data61620/train.csv',header=0)

val_data = test_data
#train_data = train_data_all.sample(frac=0.8,replace=False,random_state=101)
#colist = train_data['nid'].tolist()
#val_data = train_data_all[~train_data_all['nid'].isin(colist)]
#print(train_data_all.shape,train_data.shape,val_data.shape)

features = np.load('./data/data61620/feat.npy')
features = pd.DataFrame(features)


'''
def data_proc():
    data_1_more = train_data.loc[(train_data['label'] != 16)&(train_data['label'] != 24)
                            &(train_data['label'] != 28)].sample(frac = 1)
    data_1_less = train_data.loc[(train_data['label'] == 16)|(train_data['label'] == 24)
                            |(train_data['label'] == 28)].sample(frac = 0.25)
    train_data = pd.concat([data_1_more, data_1_less])
    train_data['label'].hist(bins=100)

    features['idxmax'] = features.idxmax(axis=1)/100
    features['idxmin'] = features.idxmin(axis=1)/100
    features['max'] = features.max(axis=1)
    features['mean'] = features.mean(axis=1)
    features['min'] = features.min(axis=1)
    features['median'] = features.median(axis=1)
    features['sum'] = features.sum(axis=1)
    features['var'] = features.var(axis=1)
    features['quan_10'] = features.quantile(q=0.1, axis=1)
    features['quan_25'] = features.quantile(q=0.25, axis=1)
    features['quan_50'] = features.quantile(q=0.50, axis=1)
    features['quan_75'] = features.quantile(q=0.75, axis=1)
    features['quan_90'] = features.quantile(q=0.90, axis=1)
    features['mad'] = features.mad(axis=1)
    features['skew'] = features.skew(axis=1)
    features['kurt'] = features.kurt(axis=1)

    for i in range(100):
        i_bin = str(i)+'_bin'
        features[i_bin] = pd.qcut(features[i], 10, labels=range(10))

    features.head()
'''
def data_features(features, train_data):
    features['nid'] = features.index
    lbz = train_data.join(pd.get_dummies(train_data.label))
    lbz = lbz.drop(columns=['label'])
    lbz.columns = ['nid'] + ['l_'+ str(i) for i in range(35)]
    features = features.merge(lbz, left_on='nid', right_on='nid',how='left')
    features = features.fillna(0).sort_values('nid')
    features = features.drop(columns=['nid'])
    features = np.array(features).astype('float32')
    return features

###建立图数据
def build_graph():
    num_nodes = 130644
    edge_list = edges
    g = graph.Graph(num_nodes=num_nodes, edges = edge_list
    ,node_feat = {'feature':features}
    )
    return g

###输入数据和标签
def feed_data(my_data):
    y = my_data['label']
    label = np.array(y,dtype="int64")
    label = np.expand_dims(label,-1)
    x = my_data['nid']
    index = np.array(x,dtype='int64')
    index = np.expand_dims(index,-1)
    return label,index


def train(input_size = 100, num_classes=35, g, epochs=50):
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    
    ###开始训练
    with fluid.dygraph.guard():
        gw = g.tensor()
        sage = my_sage(input_size=input_size,num_classes=num_classes)
        adam = fluid.optimizer.Adam(learning_rate=0.005,parameter_list=sage.parameters(), regularization=L2Decay(0.001))
        #sage_state_dict = paddle.load('./sage_5.pkl')
        #sage.set_state_dict(sage_state_dict)
        sage.train()
        feats = gw.node_feat['feature']

        for i in range(epochs):
            output = sage(gw,feats)

            node_index = fluid.dygraph.to_variable(feed_dict_train['node_index'])
            node_label = fluid.dygraph.to_variable(feed_dict_train['node_label'])

            node_index.stop_gradient = True
            node_label.stop_gradient = True

            pred = fluid.layers.gather(output,node_index)
            loss,pred = fluid.layers.softmax_with_cross_entropy(
                logits=pred,label=node_label,return_softmax=True
            )
            acc = fluid.layers.accuracy(input=pred,label=node_label,k=1)
            loss = fluid.layers.mean(loss)
            loss.backward()
            adam.minimize(loss)
            sage.clear_gradients()

            #dy_param_value = {}
            #for param in sage.parameters():
            #    dy_param_value[param.name] = param.numpy()

            if i%10==0:
                print(i,'loss:',loss,'acc:',acc)
    return sage

def feed_data_val(my_data):
    x = my_data['nid']
    index = np.array(x,dtype='int64')
    index = np.expand_dims(index,-1)
    return index

def pred_test():
    index_np = node_index_val.numpy()
    pred_np = pred.numpy()
    result = pd.DataFrame(np.hstack((pred_np, index_np)), columns=[*[str(i) for i in range(35)],'nid'])
    result['label'] = result.iloc[:,:-1].idxmax(axis=1).astype(int)
    result['max'] = result.iloc[:,:-2].max(axis=1)
    #result['sum35'] = result.iloc[:,:-3].sum(axis=1)
    result['label'][result['max']<0.4] = 12
    result_cls = result['label'].values.reshape(-1,1).astype('int64')
    result = np.hstack((index_np,result_cls))
    result = np.vstack((['nid','label'],result))

    index_np = node_index_val.numpy()
    pred_np = pred.numpy()
    result_cls = np.argmax(pred_np,axis=1)
    result_cls = result_cls.reshape(-1,1)
    result = np.hstack((index_np,result_cls))
    result = np.vstack((['nid','label'],result))
    #print(result[0:5])

    np.savetxt('result.csv',result,fmt='%s',delimiter=',')
    return None

###建立随机森林比较
def rf_compare():
    id_RF = feed_dict_train['node_index']
    train_RF_row = feed_dict_train['graph/node_feat/feature'][id_RF].squeeze()

    train_RF = train_RF_row
    #train_RF = normalize(train_RF_row, axis=1, norm='max')
    label_RF = feed_dict_train['node_label'].squeeze()

    clf = RF(n_estimators=30,max_depth=6,max_features=35)
    clf.fit(train_RF,label_RF)
    score = clf.score(train_RF, label_RF)
    return score

if __name__ == '__main__':
    features = data_features(features, train_data)
    g = build_graph()
    feed_dict = {}
    feed_dict['node_label'],feed_dict['node_index']=feed_data(train_data)
    feed_dict_train = feed_dict
    sage_ = train(g)
    save_path1 = 'sage_5.pkl'
    paddle.save(sage_.state_dict(),save_path1)
    feed_dict_val = {}
    feed_dict_val['node_index']=feed_data_val(test_data)
    node_index_val = fluid.dygraph.to_variable(feed_dict_val['node_index'])
    node_index_val.stop_gradient=True
    pred = fluid.layers.gather(output,node_index_val)
    pred_test()
    score = rf_compare()
