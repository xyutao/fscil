from mxnet.gluon.data import dataset
import os, gzip, tarfile, struct, warnings
import numpy as np

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
from mxnet import nd, image, recordio, base
import pickle as pkl


class NC_CIFAR100(dataset._DownloadedDataset):
    """CIFAR100 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html

    Each sample is an image (in 3D NDArray) with shape (32, 32, 1).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/cifar100
        Path to temp folder for storing data.
    fine_label : bool, default False
        Whether to load the fine-grained (100 classes) or coarse-grained (20 super-classes) labels.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample. For example:
    ::

        transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, logger, root=os.path.join(base.data_dir(), 'datasets', 'cifar100'),
                 fine_label=False, train=True, transform=None, c_way=5, k_shot=5, fix_class=None, base_class=0):
        self.name = 'NC_CIFAR100'
        self._train = train
        self._archive_file = ('cifar-100-binary.tar.gz', 'a0bb982c76b83111308126cc779a992fa506b90b')
        self._train_data = [('train.bin', 'e207cd2e05b73b1393c74c7f5e7bea451d63e08e')]
        self._test_data = [('test.bin', '8fb6623e830365ff53cf14adec797474f5478006')]
        self._fine_label = fine_label
        self._namespace = 'cifar100'
        self._c_way = c_way
        self._k_shot = k_shot
        self._fix_class = fix_class
        self._base_class = base_class
        self._logger = logger
        super(NC_CIFAR100, self).__init__(root, transform) # pylint: disable=bad-super-call

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072+2)
        return data[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
               data[:, 0+self._fine_label].astype(np.int32)

    def _get_data(self):

        if any(not os.path.exists(path) or not check_sha1(path, sha1)
               for path, sha1 in ((os.path.join(self._root, name), sha1)
                                  for name, sha1 in self._train_data + self._test_data)):
            namespace = 'gluon/dataset/'+self._namespace
            filename = download(_get_repo_file_url(namespace, self._archive_file[0]),
                                path=self._root,
                                sha1_hash=self._archive_file[1])

            with tarfile.open(filename) as tar:
                tar.extractall(self._root)

        if self._train:
            data_files = self._train_data
        else:
            data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name, _ in data_files))

        data = np.concatenate(data)
        label = np.concatenate(label)
        if not self._fix_class:
            np.random.seed(0)
            classes = np.random.choice(np.max(label)+1,size=self._c_way,replace=False)
            self._fix_class = list(classes)
        if self._logger:
            self._logger.info('select CIFAR100 classes : {} , fine label = {}, train = {}'.
                  format(self._fix_class, self._fine_label, self._train))

        if self._train:
            select_index = list()
            new_label = list()
            for i,l in enumerate(self._fix_class):
                ind = list(np.where(l==label)[0])
                np.random.seed(1)
                random_ind = np.random.choice(ind,self._k_shot,replace=False)
                select_index.extend(random_ind)
                new_label.extend([i+self._base_class]*len(random_ind))
            data = data[select_index]
            label = np.array(new_label)
        else:
            select_index = list()
            new_label = list()
            for i,l in enumerate(self._fix_class):
                ind = list(np.where(l==label)[0])
                select_index.extend(ind)
                new_label.extend([i+self._base_class]*len(ind))
            data = data[select_index]
            label = np.array(new_label)

        self._data = nd.array(data, dtype=data.dtype)
        self._label = label
        if self._logger:
            self._logger.info('the number of cifar100 new class samples : %d'%(label.shape[0]))

class NC_MiniImageNet(dataset.Dataset):
    def __init__(self, logger, root='~/.mxnet/dataset/mini_imagenet', train=True, transform=None, c_way=5, k_shot=5, fix_class=None, base_class=0, fine_label=True):
        if train:
            root = os.path.join(root,'train')
        else:
            root = os.path.join(root,'test')
        self.name = 'NC_MiniImageNet'
        self._logger = logger
        self._root = os.path.expanduser(root)
        self._flag = 1 # flag 1: RGB  0: gray
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)
        self._train = train
        self._class = 100

        self._c_way = c_way
        self._k_shot = k_shot
        self._fix_class = fix_class
        self._base_class = base_class
        self._few_shot()

    def _list_images(self, root):
        self.synsets = []
        self.images = []
        self.labels = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s' % (
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.images.append(filename)
                self.labels.append(label)

    def _few_shot(self):
        self._data = []

        if not self._fix_class:
            np.random.seed(0)
            classes = np.random.choice(self._class, size=self._c_way,replace=False)
            self._fix_class = list(classes)
        if self._logger:
            self._logger.info('select MiniImageNet classes : {} , train = {}'.
                  format(self._fix_class, self._train))

        if self._train:
            select_index = list()
            new_label = list()
            for i,l in enumerate(self._fix_class):
                ind = list(np.where(l==np.array(self.labels))[0])
                np.random.seed(1)
                random_ind = np.random.choice(ind,self._k_shot,replace=False)
                select_index.extend(random_ind)
                new_label.extend([i+self._base_class]*len(random_ind))
            for i in select_index:
                self._data.append(self.images[i])
            self._label = new_label
        else:
            select_index = list()
            new_label = list()
            for i,l in enumerate(self._fix_class):
                ind = list(np.where(l==np.array(self.labels))[0])
                select_index.extend(ind)
                new_label.extend([i+self._base_class]*len(ind))
            for i in select_index:
                self._data.append(self.images[i])
            self._label = new_label
        if self._logger:
            self._logger.info('the number of samples : %d'%(len(new_label)))

    def __getitem__(self, idx):
        img = image.imread(self._data[idx], self._flag)
        label = self._label[idx]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self._label)

class NC_CUB200(dataset.Dataset):
    def __init__(self, logger, root='~/.mxnet/dataset/CUB_200_2011/', train=True, transform=None, c_way=5, k_shot=5, fix_class=None, base_class=0, fine_label=True):

        self.name = 'NC_CUB200'
        self._train = train
        self._logger = logger
        self._root = os.path.expanduser(root)
        self._flag = 1 # flag 1: RGB  0: gray
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._pre_operate(self._root)

        self._class = 100

        self._c_way = c_way
        self._k_shot = k_shot
        self._fix_class = fix_class
        self._base_class = base_class
        self._few_shot()

    def text_read(self,file):
        with open(file,'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self,list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root,'images.txt')
        split_file = os.path.join(root,'train_test_split.txt')
        class_file = os.path.join(root,'image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx  = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.images = []
        self.labels = []
        if self._train:
            for k in train_idx:
                image_path = os.path.join(root,'images',id2image[k])
                self.images.append(image_path)
                self.labels.append(int(id2class[k])-1)
        else:
            for k in test_idx:
                image_path = os.path.join(root,'images',id2image[k])
                self.images.append(image_path)
                self.labels.append(int(id2class[k])-1)


    def _few_shot(self):
        self._data = []

        if not self._fix_class:
            np.random.seed(0)  # random select classes
            classes = np.random.choice(self._class, size=self._c_way,replace=False)
            self._fix_class = list(classes)
        if self._logger:
            self._logger.info('select CUB200 classes : {} , train = {}'.
                  format(self._fix_class, self._train))

        if self._train:
            select_index = list()
            new_label = list()

            for i,l in enumerate(self._fix_class):
                ind = list(np.where(l==np.array(self.labels))[0])
                np.random.seed(1)  # random select pictures
                try:
                    random_ind = np.random.choice(ind,self._k_shot,replace=False)
                except:
                    random_ind = np.random.choice(ind, len(ind), replace=False)
                select_index.extend(random_ind)
                new_label.extend([i+self._base_class]*len(random_ind))
            for i in select_index:
                self._data.append(self.images[i])
            self._label = new_label
        else:
            select_index = list()
            new_label = list()
            for i,l in enumerate(self._fix_class):
                ind = list(np.where(l==np.array(self.labels))[0])
                select_index.extend(ind)
                new_label.extend([i+self._base_class]*len(ind))
            for i in select_index:
                self._data.append(self.images[i])
            self._label = new_label
        if self._logger:
            self._logger.info('the number of samples : %d'%(len(new_label)))

    def __getitem__(self, idx):
        img = image.imread(self._data[idx], self._flag)
        label = self._label[idx]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self._label)

def merge_datasets(d0, d1):
    return MergeDataset(d0, d1)

def merge_mini_imagenet(d0, d1):
    return MergeMiniImageNet(d0, d1)

class MergeDataset(dataset._DownloadedDataset):
    def __init__(self, d0, d1, root=os.path.join(base.data_dir(), 'datasets', 'merge'), transform=None):
        super(MergeDataset, self).__init__(root=root, transform=transform)
        if d1 is None and d0 is not None:
            self._data, self._label = d0._data, d0._label
        elif d0 is None and d1 is not None:
            self._data, self._label = d1._data, d1._label
        elif d0 is not None and d1 is not None:
            self._data = nd.concat(d0._data, d1._data, dim=0)
            self._label = np.concatenate([d0._label, d1._label])
        else:
            self._data, self._label = None, None

    def _get_data(self):
        pass

class MergeMiniImageNet(dataset.Dataset):
    def __init__(self, d0, d1):
        self._flag = 1
        self._transform = None
        self._data = d0._data + d1._data
        self._label = d0._label + d1._label

    def __getitem__(self, idx):
        img = image.imread(self._data[idx], self._flag)
        label = self._label[idx]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self._label)

if __name__=='__main__':
    cub200 = NC_CUB200(None, train=True)
