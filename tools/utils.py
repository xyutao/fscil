import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
from mxnet import gluon

def mx_calculate_dist(anchor, positive):
    d1 = mx.ndarray.sum(anchor * anchor, axis=1).reshape(1,1)
    d2 = mx.ndarray.sum(positive * positive, axis=1).reshape(-1,1)
    eps = 1e-12
    a = d1.repeat(int(positive.shape[0]))
    b = mx.ndarray.transpose(d2.repeat(1))
    c = 2.0 * mx.ndarray.dot(anchor, mx.ndarray.transpose(positive))
    return mx.ndarray.sqrt(mx.ndarray.abs((a + b - c))+eps)

def np_calculate_dist(anchor, positive):
    d1 = np.sum(anchor * anchor, axis=1).reshape(1, 1)
    d2 = np.sum(positive * positive, axis=1).reshape(-1, 1)
    eps = 1e-12
    a = d1.repeat(int(positive.shape[0])).reshape(1, -1)
    b = d2.T
    c = 2.0 * np.dot(anchor, positive.T)
    return np.sqrt(np.abs((a + b - c)) + eps)

def processing(input, label, features, dims):
    bmu = input[0]
    mydict = {}
    for i in range(bmu.shape[0]):
        node_bmu = int(bmu[i])
        if node_bmu not in mydict.keys():
            mydict[node_bmu] = {}

        node_class = int(label[i])
        if 'majority' not in mydict[node_bmu].keys():
            mydict[node_bmu]['majority'] = {}
            mydict[node_bmu]['majority']['class'] = {}
            mydict[node_bmu]['majority']['pic_ind'] = {}
            mydict[node_bmu]['majority']['pic_anchor'] = {}

        if node_class not in mydict[node_bmu].keys():
            mydict[node_bmu][node_class] = {}
            mydict[node_bmu][node_class]['index'] = []
            mydict[node_bmu][node_class]['index'].append(int(i))
            mydict[node_bmu][node_class]['feature'] = features[i]
            mydict[node_bmu][node_class]['num'] = 1

        else:
            mydict[node_bmu][node_class]['index'].append(int(i))
            mydict[node_bmu][node_class]['feature'] = \
                np.concatenate([mydict[node_bmu][node_class]['feature'],features[i]],axis=0)
            mydict[node_bmu][node_class]['num'] += 1

    all_node = list(mydict.keys())

    for node in all_node:
        all_class = list(mydict[node].keys())
        all_class.remove('majority')
        num_flag = 0
        for node_class in all_class:
            if num_flag == 0:
                max_num = mydict[node][node_class]['num']
                major_class = node_class
                all_index = mydict[node][node_class]['index']
                all_feature = mydict[node][node_class]['feature'].reshape(-1,dims)
                num_flag =1
            else:
                if max_num > mydict[node][node_class]['num']:
                    pass
                else:
                    max_num = mydict[node][node_class]['num']
                    major_class = node_class
                    all_index = mydict[node][node_class]['index']
                    all_feature = mydict[node][node_class]['feature'].reshape(-1,dims)
        mydict[node]['majority']['class'] = major_class
        mydict[node]['majority']['pic_ind'] = all_index
        mydict[node]['majority']['pic_anchor'] = all_feature

    return mydict

def process_to_anchor(mydict2, ng_centroid, dim, ctx):

    all_nodes = list(mydict2.keys())
    all_nodes.sort()
    all_index = []
    all_class = []
    ng_flag = 0
    ng_centroid = mx.ndarray.array(ng_centroid, ctx=ctx[0])

    for node in all_nodes:
        dist = mx_calculate_dist(ng_centroid[node].reshape(-1,dim),
                                 mx.nd.array(mydict2[node]['majority']['pic_anchor'],ctx=ctx[0]))
        ind = mx.nd.argmin(dist,axis=1)
        ind = int(ind.asnumpy())
        variance = np.var(mydict2[node]['majority']['pic_anchor'], 0)
        mydict2[node]['majority']['pic_ind'] = mydict2[node]['majority']['pic_ind'][ind]
        mydict2[node]['majority']['pic_anchor'] = mydict2[node]['majority']['pic_anchor'][ind]

        all_index.append(mydict2[node]['majority']['pic_ind'])
        all_class.append(mydict2[node]['majority']['class'])
        if ng_flag == 0 :
            anchor = mydict2[node]['majority']['pic_anchor']
            variances = variance
            ng=ng_centroid[node].asnumpy()
            all_bmu = ng.reshape(-1,dim)
            ng_flag = 1
        else:
            anchor = np.concatenate([anchor,mydict2[node]['majority']['pic_anchor']],axis=0)
            variances = np.concatenate([variances,variance],axis=0)
            ng = ng_centroid[node].asnumpy()
            all_bmu = np.concatenate([all_bmu,ng.reshape(-1,dim)],axis=0)


    #anchor = anchor.reshape(-1,dim)
    all_bmu = all_bmu.reshape(-1,dim,1,1)
    return all_class, all_index, all_bmu, variances.reshape(-1, dim)

def plot_schedule(schedule_fn, iterations=50):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.savefig('./lr.png')

class LinearWarmUp():
    def __init__(self, schedule, start_lr, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        start_lr: learning rate used at start of the warm-up (float)
        length: number of iterations used for the warm-up (int)
        """
        self.schedule = schedule
        self.start_lr = start_lr
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.finish_lr = copy.copy(schedule)(0)
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.length:
            return iteration * (self.finish_lr - self.start_lr) / (self.length) + self.start_lr
        else:
            return self.schedule(iteration - self.length)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--gpus', type=str, default='0,',
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='resnet',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=1,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='./params',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')

    parser.add_argument('--save-name', type=str, default='_')
    parser.add_argument('--sess-num', type=int, default=9)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--base-decay-epoch', type=str, default='40,60')
    parser.add_argument('--inc-decay-epoch', type=int, default=200)
    parser.add_argument('--lrs', type=float, default=0.1)
    parser.add_argument('--base-lrs', type=float, default=0.1)
    parser.add_argument('--lr-decay', type=float, default=0.1)

    parser.add_argument('--AL-weight', type=float, default=200)
    parser.add_argument('--smooth-weight', type=float, default=200)
    parser.add_argument('--min-weight', type=float, default=0.01)
    parser.add_argument('--ng-min-lr', type=float, default=0.01)
    parser.add_argument('--max-weight', type=float, default=1)
    parser.add_argument('--oce-weight', type=float, default=1)
    parser.add_argument('--pdl-weight', type=float, default=1)
    parser.add_argument('--nme-weight', type=float, default=1)
    parser.add_argument('--temperature', type=float, default=1)

    parser.add_argument('--use-AL', action='store_true', default=False)
    parser.add_argument('--use-smooth', action='store_true', default=False)
    parser.add_argument('--use-ng-min', action='store_true', default=False)
    parser.add_argument('--use-oce', action='store_true', default=False)
    parser.add_argument('--use-lda', action='store_true', default=False)
    parser.add_argument('--use-pdl', action='store_true', default=False)
    parser.add_argument('--use-ng-max', action='store_true', default=False)
    parser.add_argument('--use-nme', action='store_true', default=False)
    parser.add_argument('--small-lr', action='store_true', default=False)
    parser.add_argument('--wo-bn', action='store_true', default=False)
    parser.add_argument('--fw', action='store_true', default=False)
    parser.add_argument('--use-ng', action='store_true', default=False)
    parser.add_argument('--ng-update', action='store_true', default=False)
    parser.add_argument('--ng-var', action='store_true', default=False)

    parser.add_argument('--use-warmUp', action='store_true', default=False)
    parser.add_argument('--use-cw', action='store_true', default=False)

    parser.add_argument('--use-all-novel', action='store_true', default=False)
    parser.add_argument('--cum', action='store_true', default=False)
    parser.add_argument('--fix-conv', action='store_true', default=False)
    parser.add_argument('--fix-epoch', type=int, default=200)

    parser.add_argument('--c-way', type=int, default=5)
    parser.add_argument('--k-shot', type=int, default=5)
    parser.add_argument('--base-acc', type=float, default=0)

    parser.add_argument('--select-best', type=strprepare_anchor, default='select_best2')
    parser.add_argument('--dataset', type=str, default='NC_CIFAR100')
    opt = parser.parse_args()
    return opt


def select_best(acc_dict, sess):
    # select best params.
    sess_all_acc = acc_dict[str(sess)]
    for e, pr in enumerate(sess_all_acc):
        trade_off = pr[0] * pr[1] / (pr[0] + pr[1])
        if e == 0:
            best_num = trade_off
            best_e = e
        else:
            if trade_off > best_num:
                best_num = trade_off
                best_e = e
    return best_e

def select_best2(acc_dict, sess):
    # select best params.
    sess_all_acc = acc_dict[str(sess)]
    f = 0
    best_e = len(sess_all_acc)-1
    for e, pr in enumerate(sess_all_acc):
        if pr[0] < pr[1] and f==0:
            best_base = pr[0]
            best_novel = pr[1]
            best_e = e
            f = 1
        if f==1:
            if pr[0] > best_base and pr[1] > best_novel:
                best_base = pr[0]
                best_novel = pr[1]
                best_e = e
    return best_e

def select_best3(acc_dict, sess):
    # select best params.
    sess_all_acc = acc_dict[str(sess)]
    return len(sess_all_acc)-1

def DataLoader(Dataset, transform, batch_size, num_workers, shuffle=True):
    dataloader = gluon.data.DataLoader(
        Dataset.transform_first(transform),
        batch_size=batch_size, shuffle=shuffle, last_batch='discard', num_workers=num_workers)
    return dataloader
