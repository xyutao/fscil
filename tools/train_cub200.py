import matplotlib
matplotlib.use('Agg')
import argparse, time, logging
import os
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms
import sys
sys.path.append('./')
from model.resnet18_cub200 import resnet18_v1
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms
from dataloader.dataloader import NC_CUB200, merge_datasets, merge_mini_imagenet
from tools.ng_anchor import prepare_anchor

from tools.utils import plot_schedule, LinearWarmUp
from tools.loss import NG_Max_Loss, NG_Min_Loss
from tools.utils import parse_args
from tools.plot import plot_pr, plot_all_sess
from tools.loss import DistillationSoftmaxCrossEntropyLoss
import json
from tools.utils import select_best, select_best2, select_best3

opt = parse_args()

batch_size = opt.batch_size

num_gpus = len(opt.gpus.split(','))
batch_size *= max(1, num_gpus)
context = [mx.gpu(int(i)) for i in opt.gpus.split(',')]
num_workers = opt.num_workers

model_name = opt.model

# ==========================================================================
if model_name=='resnet18' and opt.dataset=='NC_CUB200':
    classes = 100
    net = resnet18_v1(classes=classes, wo_bn=opt.wo_bn,fw=opt.fw)
    feature_size = 512
    pretrained=True
else:
    raise KeyError('network key error')

if opt.resume_from:
    net.load_parameters(opt.resume_from, ctx = context)

DATASET = eval(opt.dataset)
# ==========================================================================

optimizer = 'nag'

save_period = opt.save_period

plot_path = opt.save_plot_dir

save_dir = time.strftime('./experimental_result/{}/{}/10-way-5-shot/%Y-%m-%d-%H-%M-%S'.format(opt.dataset, model_name), time.localtime())
save_dir = save_dir + opt.save_name

makedirs(save_dir)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_save_dir = os.path.join(save_dir, 'log.txt')
fh = logging.FileHandler(log_save_dir)
fh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.info(opt)

if opt.dataset=='NC_CUB200':
    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop(84),
        transforms.Resize(256),
        gcv_transforms.RandomCrop(224, pad=0),
        # gcv_transforms.RandomCrop(224, pad=8),
        transforms.RandomFlipLeftRight(),
        # transforms.RandomColorJitter(brightness=0.4, contrast=0.4,
        #                              saturation=0.4),
        # transforms.RandomLighting(0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    anchor_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def test(ctx, val_data, net, sess):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X, sess)[1] for X in data]
        metric.update(label, outputs)
    return metric.get()

def DataLoader(Dataset, transform, batch_size, num_workers, shuffle=True):
    dataloader = gluon.data.DataLoader(
        Dataset.transform_first(transform),
        batch_size=batch_size, shuffle=shuffle, last_batch='keep', num_workers=num_workers)
    return dataloader

def train(net, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    if not opt.resume_from:
        net.initialize(mx.init.Xavier(), ctx=ctx)
        if pretrained:
            logger.info('loading pre-trained model...')
            net.load_parameters('/home/dsl/FICIL2020/params/CUB200/resnet19_inchead/resnet18_v1-a0666292.params', allow_missing=True, ignore_extra=True)
        if opt.dataset == 'NC_CUB200':
            n = mx.nd.zeros(shape=(1,3,224,224),ctx=ctx[0])  #####init CNN
        else:
            raise KeyError('dataset keyerror')
        for m in range(11):
           net(n,m)
        logger.info('init')

    def makeSchedule(start_lr,base_lr,length,step,factor):
        schedule = mx.lr_scheduler.MultiFactorScheduler(step=step, factor=factor)
        schedule.base_lr = base_lr
        schedule = LinearWarmUp(schedule, start_lr=start_lr, length=length)
        return schedule
# ==========================================================================
    sesses = list(np.arange(opt.sess_num))
    epochs = [opt.epoch]*opt.sess_num
    lrs = [opt.base_lrs]+[opt.lrs]*(opt.sess_num-1)
    lr_decay = opt.lr_decay
    base_decay_epoch = [int(i) for i in opt.base_decay_epoch.split(',')] + [np.inf]
    lr_decay_epoch = [base_decay_epoch]+[[opt.inc_decay_epoch, np.inf]]*(opt.sess_num-1)

    AL_weight = opt.AL_weight
    min_weight = opt.min_weight
    oce_weight = opt.oce_weight
    pdl_weight = opt.pdl_weight
    max_weight = opt.max_weight
    temperature = opt.temperature

    use_AL = opt.use_AL  # anchor loss
    use_ng_min = opt.use_ng_min  # Neural Gas min loss
    use_ng_max = opt.use_ng_max  # Neural Gas min loss
    ng_update = opt.ng_update  # Neural Gas update node
    use_oce = opt.use_oce  # old samples cross entropy loss
    use_pdl = opt.use_pdl  # probability distillation loss
    use_nme = opt.use_nme  # Similarity loss
    use_warmUp = opt.use_warmUp
    use_ng = opt.use_ng  # Neural Gas
    fix_conv = opt.fix_conv  # fix cnn to train novel classes
    fix_epoch = opt.fix_epoch
    c_way = opt.c_way
    k_shot = opt.k_shot
    base_acc = opt.base_acc  # base model acc
    select_best_method = opt.select_best  # select from _best, _best2, _best3
    init_class = 100
    anchor_num = 400
# ==========================================================================
    acc_dict = {}
    all_best_e = []

    if model_name[-7:]!='maxhead':
        net.fc3.initialize(mx.init.Normal(sigma=0.001), ctx=ctx, force_reinit=True)
        net.fc4.initialize(mx.init.Normal(sigma=0.001), ctx=ctx, force_reinit=True)
        net.fc5.initialize(mx.init.Normal(sigma=0.001), ctx=ctx, force_reinit=True)
        net.fc6.initialize(mx.init.Normal(sigma=0.001), ctx=ctx, force_reinit=True)
        net.fc7.initialize(mx.init.Normal(sigma=0.001), ctx=ctx, force_reinit=True)
        net.fc8.initialize(mx.init.Normal(sigma=0.001), ctx=ctx, force_reinit=True)
        net.fc9.initialize(mx.init.Normal(sigma=0.001), ctx=ctx, force_reinit=True)
        net.fc10.initialize(mx.init.Normal(sigma=0.001), ctx=ctx, force_reinit=True)

    for sess in sesses:
        logger.info('session : %d'%sess)

        # prepare the first anchor batch
        if sess==0 and opt.resume_from:
            acc_dict[str(sess)] = list()
            acc_dict[str(sess)].append([base_acc,0])
            all_best_e.append(0)
            continue

        # update anchor
        if sess!= 0:
            if ng_update == True:
                if sess==1:
                    update_anchor1, bmu, variances = \
                        prepare_anchor(DATASET, logger, anchor_trans, num_workers, feature_size, net, ctx,use_ng,init_class)
                    update_anchor_data = DataLoader(update_anchor1, anchor_trans, update_anchor1.__len__(), num_workers,
                                                    shuffle=False)
                    if opt.ng_var:
                        idx_1 = np.where(variances.asnumpy() > 0.5)
                        idx_2 = np.where(variances.asnumpy() < 0.5)
                        variances[idx_1] = 0.8
                        variances[idx_2] = 1
                else:
                    base_class = init_class + (sess - 1) * c_way
                    new_class = list(init_class + (sess - 1) * c_way + (np.arange(c_way)))
                    new_set = DATASET(train=True, fine_label=True, fix_class=new_class, base_class=base_class,
                                logger=logger)
                    update_anchor2 = merge_datasets(update_anchor1, new_set)
                    update_anchor_data = DataLoader(update_anchor2, anchor_trans, update_anchor2.__len__(), num_workers,
                                                    shuffle=False)
            elif sess==1:
                update_anchor1, bmu, variances =prepare_anchor(DATASET, logger, anchor_trans, num_workers, feature_size, net, ctx, use_ng, init_class)
                update_anchor_data = DataLoader(update_anchor1, anchor_trans, update_anchor1.__len__(), num_workers,
                                                shuffle=False)
                if opt.ng_var:
                  idx_1 = np.where(variances.asnumpy() > 0.5)
                  idx_2 = np.where(variances.asnumpy() < 0.5)
                  variances[idx_1] = 0.8
                  variances[idx_2] = 1

            for batch in update_anchor_data:
                anc_data = gluon.utils.split_and_load(batch[0], ctx_list=[ctx[0]], batch_axis=0)
                anc_label = gluon.utils.split_and_load(batch[1], ctx_list=[ctx[0]], batch_axis=0)
                with ag.pause():
                    anchor_feat, anchor_logit = net(anc_data[0], sess-1)
                anchor_feat = [anchor_feat]
                anchor_logit = [anchor_logit]

        trainer = gluon.Trainer(net.collect_params(), optimizer,
                                {'learning_rate': lrs[sess], 'wd': opt.wd, 'momentum': opt.momentum})

        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        if use_nme:
            def loss_fn_disG(f1, f2, weight):
                f1 = f1.reshape(anchor_num,-1)
                f2 = f2.reshape(anchor_num,-1)
                similar = mx.nd.sum(f1*f2, 1)
                return (1-similar)*weight
            digG_weight = opt.nme_weight
        if use_AL:
            if opt.ng_var:
                def l2lossVar(feat, anc, weight, var):
                    dim = feat.shape[1]
                    feat = feat.reshape(-1, dim)
                    anc = anc.reshape(-1, dim)
                    loss = mx.nd.square(feat - anc)
                    loss = loss * weight * var
                    return mx.nd.mean(loss, axis=0, exclude=True)
                loss_fn_AL = l2lossVar
            else:
                loss_fn_AL = gluon.loss.L2Loss(weight=AL_weight)

        if use_pdl:
            loss_fn_pdl = DistillationSoftmaxCrossEntropyLoss(temperature=temperature, hard_weight=0, weight=pdl_weight)
        if use_oce:
            loss_fn_oce = gluon.loss.SoftmaxCrossEntropyLoss(weight=oce_weight)
        if use_ng_min:
            loss_fn_max = NG_Max_Loss(lmbd=max_weight, margin=0.5)
        if use_ng_min:
            min_loss = NG_Min_Loss(num_classes=opt.c_way, feature_size=feature_size, lmbd=min_weight, # center weight = 0.01 in the paper
                                     ctx=ctx[0])
            min_loss.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)  # init matrix.
            center_trainer = gluon.Trainer(min_loss.collect_params(),
                                           optimizer="sgd", optimizer_params={"learning_rate": opt.ng_min_lr}) # alpha=0.1 in the paper.


        # train_history = TrainingHistory(['base acc', 'new acc'])
        lr_decay_count = 0

        # dataloader
        if opt.cum and sess==1 :
            base_class = list(np.arange(init_class))
            joint_data = DATASET(train=True, fine_label=True, c_way=init_class, k_shot=100, fix_class=base_class, logger=logger)

        if sess==0 :
            base_class = list(np.arange(init_class))
            new_class = list(init_class + (np.arange(c_way)))
            base_data = DATASET(train=True, fine_label=True, c_way=init_class, k_shot=100, fix_class=base_class, logger=logger)
            bc_val_data = DataLoader(DATASET(train=False, fine_label=True, fix_class=base_class, logger=logger)
                                      , transform_test, 100, num_workers, shuffle=False)
            nc_val_data = DataLoader(
                DATASET(train=False, fine_label=True, fix_class=new_class, base_class=len(base_class), logger=logger)
                , transform_test, 100, num_workers, shuffle=False)
        else:
            base_class = list(np.arange(init_class + (sess-1)*c_way))
            new_class = list(init_class + (sess-1)*c_way + (np.arange(c_way)))
            train_data = DATASET(train=True, fine_label=True, c_way=c_way, k_shot=k_shot, fix_class=new_class, base_class=len(base_class), logger=logger)
            bc_val_data = DataLoader(DATASET(train=False, fine_label=True, fix_class=base_class, logger=logger)
                                      , transform_test, 100, num_workers, shuffle=False)
            nc_val_data = DataLoader(
                DATASET(train=False, fine_label=True, fix_class=new_class, base_class=len(base_class), logger=logger)
                , transform_test, 100, num_workers, shuffle=False)

        if sess == 0:
            train_data = DataLoader(base_data, transform_train, min(batch_size, base_data.__len__()), num_workers, shuffle=True)
        else:
            if opt.cum: # cumulative : merge base and novel dataset.
                if opt.dataset == 'NC_CIFAR100':
                    joint_data = merge_datasets(joint_data, train_data)
                elif opt.dataset == 'NC_MiniImageNet' or 'NC_CUB200':
                    joint_data = merge_mini_imagenet(joint_data, train_data)
                train_data = DataLoader(joint_data, transform_train, min(batch_size, joint_data.__len__()), num_workers, shuffle=True)

            elif opt.use_all_novel: # use all novel data
                if opt.dataset == 'NC_CIFAR100':
                    if sess==1:
                        novel_data = train_data
                    else:
                        novel_data = merge_datasets(novel_data, train_data)
                elif opt.dataset == 'NC_MiniImageNet' or 'NC_CUB200':
                    if sess==1:
                        novel_data = train_data
                    else:
                        novel_data = merge_mini_imagenet(novel_data, train_data)
                train_data = DataLoader(novel_data, transform_train, min(batch_size, novel_data.__len__()), num_workers, shuffle=True)

            else: # basic method
                train_data = DataLoader(train_data, transform_train, min(batch_size, train_data.__len__()), num_workers, shuffle=True)
        for epoch in range(epochs[sess]):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss, train_anchor_loss, train_oce_loss = 0, 0, 0
            train_disg_loss, train_pdl_loss, train_min_loss = 0, 0, 0
            train_max_loss = 0
            num_batch = len(train_data)

            lr = trainer.learning_rate
            if epoch == lr_decay_epoch[sess][lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
                lr_decay_count += 1

            if sess!=0 and epoch<fix_epoch:
                fix_cnn = fix_conv
            else:
                fix_cnn = False

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                all_loss = list()
                with ag.record():
                    output_feat, output = net(data[0],sess,fix_cnn)
                    output_feat = [output_feat]
                    output = [output]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                    all_loss.extend(loss)

                    if sess>0 and use_nme:
                        anchor_h = [net(X, sess, fix_cnn)[0] for X in anc_data]
                        disg_loss = [loss_fn_disG(a_h, a, weight=digG_weight) for a_h, a in zip(anchor_h, anchor_feat)]
                        all_loss.extend(disg_loss)

                    if sess>0 and use_ng_max:
                        max_loss = [loss_fn_max(feat, label, feature_size, epoch, sess, init_class) for feat, label in zip(output_feat, label)]
                        all_loss.extend(max_loss[0])

                    if sess > 0 and use_AL:    # For anchor loss
                        anchor_h = [net(X, sess, fix_cnn)[0] for X in anc_data]
                        if opt.ng_var:
                            anchor_loss = [loss_fn_AL(anchor_h[0], anchor_feat[0], AL_weight, variances)]
                            all_loss.extend(anchor_loss)
                        else:
                            anchor_loss = [loss_fn_AL(a_h, a) for a_h, a in zip(anchor_h, anchor_feat)]
                            all_loss.extend(anchor_loss)

                    if sess > 0 and use_ng_min:
                        loss_min = min_loss(output_feat[0], label[0])
                        all_loss.extend(loss_min)

                    if sess > 0 and use_pdl:
                        anchor_l = [net(X, sess, fix_cnn)[1] for X in anc_data]
                        anchor_l = [anchor_l[0][:,:init_class+(sess-1)*c_way]]
                        soft_label = [mx.nd.softmax(anchor_logit[0][:,:init_class+(sess-1)*c_way] / temperature)]
                        pdl_loss = [loss_fn_pdl(a_h, a, soft_a) for a_h, a, soft_a in zip(anchor_l, anc_label, soft_label)]
                        all_loss.extend(pdl_loss)

                    if sess > 0 and use_oce:
                        anchorp = [net(X, sess, fix_cnn)[1] for X in anc_data]
                        oce_Loss = [loss_fn_oce(ap, a) for ap, a in zip(anchorp, anc_label)]
                        all_loss.extend(oce_Loss)

                    # Normalize each loss for the trainer with batch_size=1
                    all_loss = [nd.mean(l) for l in all_loss]

                ag.backward(all_loss)
                trainer.step(1,ignore_stale_grad=True)
                if use_ng_min:
                    center_trainer.step(opt.c_way*opt.k_shot)

                train_loss += sum([l.sum().asscalar() for l in loss])
                if sess > 0 and use_AL:
                    train_anchor_loss += sum([al.mean().asscalar() for al in anchor_loss])
                if sess > 0 and use_oce:
                    train_oce_loss += sum([al.mean().asscalar() for al in oce_Loss])
                if sess > 0 and use_nme:
                    train_disg_loss += sum([al.mean().asscalar() for al in disg_loss])
                if sess > 0 and use_pdl:
                    train_pdl_loss += sum([al.mean().asscalar() for al in pdl_loss])
                if sess > 0 and use_ng_min:
                    train_min_loss += sum([al.mean().asscalar() for al in loss_min])
                if sess > 0 and use_ng_max:
                    train_max_loss += sum([al.mean().asscalar() for al in max_loss[0]])

                train_metric.update(label, output)

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()

            name, bc_val_acc = test(ctx, bc_val_data, net, sess)
            name, nc_val_acc = test(ctx, nc_val_data, net, sess)

            if epoch==0:
                acc_dict[str(sess)]=list()
            acc_dict[str(sess)].append([bc_val_acc,nc_val_acc])

            if sess==0:
                overall = bc_val_acc
            else:
                overall = (bc_val_acc*(init_class+(sess-1)*c_way)+nc_val_acc*c_way)/(init_class+sess*c_way)
            logger.info(
                '[Epoch %d] lr=%.4f train=%.4f | val(base)=%.4f val(novel)=%.4f overall=%.4f | loss=%.8f anc loss=%.8f '
                'pdl loss:%.8f oce loss: %.8f time: %.8f' %
                (epoch, lr, acc, bc_val_acc, nc_val_acc, overall, train_loss, train_anchor_loss / AL_weight,
                 train_pdl_loss / pdl_weight, train_oce_loss / oce_weight, time.time() - tic))
            if use_nme:
                logger.info('digG loss:%.8f' % (train_disg_loss / digG_weight))
            if use_ng_min:
                logger.info('min_loss:%.8f' % (train_min_loss / min_weight))
            if use_ng_max:
                logger.info('max_loss:%.8f' % (train_max_loss / max_weight))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                # net.save_parameters('%s/sess-%s-cifar-%s-%d.params'%(save_dir, sess, model_name, epoch))
                net.save_parameters('%s/sess-%s-%d.params'%(save_dir, model_name, epoch))


        select = eval(select_best_method)
        best_e = select(acc_dict, sess)
        logger.info('best select : base: %f novel: %f '%(acc_dict[str(sess)][best_e][0],acc_dict[str(sess)][best_e][1]))
        all_best_e.append(best_e)

        # reload_path = '%s/sess-%s-cifar-%s-%d.params'%(save_dir, sess, model_name, best_e)
        reload_path = '%s/sess-%s-%d.params'%(save_dir, model_name, best_e)
        net.load_parameters(reload_path, ctx=context)

        # save acc_dict each session.
        with open('%s/acc_dict.json'%save_dir, 'w')  as json_file:
            json.dump(acc_dict, json_file)

        plot_pr(acc_dict,sess,save_dir)
    plot_all_sess(acc_dict,save_dir,all_best_e)

def main():
    if opt.mode == 'hybrid':
        net.hybridize()
    train(net, context)

if __name__ == '__main__':
    main()
