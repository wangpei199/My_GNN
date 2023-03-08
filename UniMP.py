import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import networkx as nx 
import pgl 
from pgl import graph
from paddle.regularizer import L2Decay

###定义SAGE网络方法 继承
class my_sage(fluid.dygraph.Layer):
    def __init__(self,input_size,num_classes):
        super(my_sage,self).__init__()
        self.sageconv1 = GraphSageConv(input_size=input_size,hidden_size=200,aggr_func='sum')
        self.sageconv2 = GraphSageConv(input_size=200,hidden_size=200,aggr_func='sum')
        self.sageconv3 = GraphSageConv(input_size=250,hidden_size=250,aggr_func='mean')
        self.sageconv4 = GraphSageConv(input_size=250,hidden_size=250,aggr_func='mean')
        self.sageconv5 = GraphSageConv(input_size=250,hidden_size=200,aggr_func='mean')
        #self.last_linear = nn.Linear(200, 200)
        self.sageconv6 = GraphSageConv(input_size=235,hidden_size=num_classes,aggr_func='sum')

    def forward(self,gw,feats):
        h, _ = self.sageconv1(gw,feats, act='leaky_relu', first_layer = True)
        h, _ = self.sageconv2(gw,h, act='leaky_relu', first_layer = True)
        #h, _ = self.sageconv5(gw,h, act='leaky_relu')
        h, _ = self.sageconv6(gw,h, act='sigmoid')
        return h
    
###修改采样方法    
class GraphSageConv(nn.Layer):
    def __init__(self, input_size, hidden_size, aggr_func="sum"):
        super(GraphSageConv, self).__init__()
        assert aggr_func in ["sum", "mean", "max", "min"], \
                "Only support 'sum', 'mean', 'max', 'min' built-in receive function."
        self.aggr_func = "reduce_%s" % aggr_func

        self.self_linear = nn.Linear(input_size, hidden_size)
        self.neigh_linear_1 = nn.Linear(input_size, hidden_size*2)
        self.neigh_linear_2 = nn.Linear(hidden_size*2, hidden_size)
        self.labels0 = paddle.zeros(shape=[130644, 35])

    def forward(self, graph, feature, act=None, first_layer = False):
        def _send_func(src_feat, dst_feat, edge_feat):
            return {"msg": src_feat["h"]}

        def _recv_func(message):
            aggr_feat = getattr(message, self.aggr_func)(message["msg"])
            #aggr_feat = message.reduce_max(message['msg'])
            #aggr_feat = message.edge_expand(aggr_feat)
            #expand_feat = message['msg'] - aggr_feat
            return aggr_feat

        msg = graph.send(_send_func, src_feat={"h": feature})
        neigh_feature = graph.recv(reduce_func=_recv_func, msg=msg)

        if first_layer:
            self_feature = self.self_linear(feature[:, 0:-35])
            neigh_label = neigh_feature[:, -35:]
            output = fluid.layers.concat([self_feature, neigh_label], axis=-1)
        else:
            self_feature = feature[:, 0:-35]
            self_feature = fluid.layers.concat([self_feature, self.labels0], axis=-1)
            
            output = self_feature + neigh_feature
            output = self.neigh_linear_1(output)
            output = F.leaky_relu(output)
            output = self.neigh_linear_2(output)
            
            #neigh_label = neigh_feature[:, -35:]
            #output = fluid.layers.concat([self_feature, neigh_feature], axis=-1)
            #output = self.lstm(paddle.reshape(output, [1,-1,435]))
            #neigh_feature = neigh_feature.T
            #neigh_features = paddle.where(neigh_feature[-35:].sum(axis=0)>14, f_linear(neigh_feature), s_linear(neigh_feature)).T
        
        if act is not None:
            output = getattr(F, act)(output)

        output = F.normalize(output, axis=1)
        return output, None