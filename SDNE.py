import pgl
import paddle
from paddle import fliud

def build_model(args, graph):
    u_i = fl.data(
        name='u_i', shape=[None, 1], dtype='int64', append_batch_size=False)
    u_j = fl.data(
        name='u_j', shape=[None, 1], dtype='int64', append_batch_size=False)

    label = fl.data(
        name='label', shape=[None], dtype='float32', append_batch_size=False)

    lr = fl.data(
        name='learning_rate',
        shape=[1],
        dtype='float32',
        append_batch_size=False)

    u_i_embed = fl.embedding(
        input=u_i,
        size=[graph.num_nodes, args.embed_dim],
        param_attr='shared_w')

    if args.order == 'first_order':
        u_j_embed = fl.embedding(
            input=u_j,
            size=[graph.num_nodes, args.embed_dim],
            param_attr='shared_w')
    elif args.order == 'second_order':
        u_j_embed = fl.embedding(
            input=u_j,
            size=[graph.num_nodes, args.embed_dim],
            param_attr='context_w')
    else:
        raise ValueError("order should be first_order or second_order, not %s"
                         % (args.order))

    inner_product = fl.reduce_sum(u_i_embed * u_j_embed, dim=1)

    loss = -1 * fl.reduce_mean(fl.logsigmoid(label * inner_product))
    optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    return loss, optimizer