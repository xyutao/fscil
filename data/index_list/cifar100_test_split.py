import os, os.path as osp, numpy as np

# train_data = [('train.bin', 'e207cd2e05b73b1393c74c7f5e7bea451d63e08e')]
data_dir = './data/dataset/cifar100'
output_dir = './data/index_list/cifar100'

test_data_file = osp.join(data_dir, 'test.bin')
fine_label = True

total_classes = 100
base_classes = 60
inc_classes = 5
sessions = 1 + (total_classes - base_classes) // inc_classes

class_split = [np.arange(base_classes)]
class_split += [base_classes + (s - 1) * 5 + np.arange(5) for s in range(1, sessions)]

data = None
with open(test_data_file, 'rb') as fin:
    data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072 + 2)
label = data[:, 0 + fine_label].astype(np.int32)

label_split = list()
for s in range(sessions):
    test_split_file = osp.join(output_dir, 'test_%d.txt' % (s + 1))
    selected_classes = class_split[s]
    selected_inds = list()

    for i, c in enumerate(selected_classes):
        selected_inds.extend(list(np.where(label == c)[0]))

    with open(test_split_file, 'w') as fout:
        for ind in selected_inds:
            fout.write('%d\n' % ind)


