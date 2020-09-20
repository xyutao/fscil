import os.path as osp

# Your root directory for the unzipped files of CUB200
data_dir = './data/dataset/CUB_200_2011'
output_dir = './data/index_list/cub200'

# It should contain the following files
image_dir = osp.join(data_dir, 'images')
split_file = osp.join(data_dir, 'train_test_split.txt')
image_file = osp.join(data_dir, 'images.txt')
image_class_labels_file = osp.join(data_dir, 'image_class_labels.txt')
class_file = osp.join(data_dir, 'classes.txt')

# The prefix for the outputted file lists
prefix = 'CUB_200_2011/images'

total_classes = 200
base_classes = total_classes // 2
inc_classes = 10    # should divide (num_total_classes - num_base_classes)

# One session for the base classes and the remaining ones for the incremental classes
sessions = 1 + (total_classes - base_classes) // inc_classes

split_list = list()
with open(split_file, 'r') as fin:
    split_list = fin.readlines()

label_dic = dict()
for line in split_list:
    id, label = list(map(int, line.strip().split(' ')))
    label_dic[id] = label

image_list = list()
with open(image_file) as fin:
    image_list = fin.readlines()


train_list_file = open(osp.join(output_dir, 'train.txt'), 'w')
test_list_file = open(osp.join(output_dir, 'test.txt'), 'w')

image_dic = dict()
for line in image_list:
    id, fn = line.strip().split(' ')
    id = int(id)
    label = label_dic[id]
    outline = osp.join(prefix, fn)
    image_dic[id] = outline
    if label == 0:
        test_list_file.write(outline + '\n')
    else:
        train_list_file.write(outline + '\n')


train_list_file.close()
test_list_file.close()

class_image_dic = dict()
for i in range(total_classes):
    class_image_dic[i+1] = list()

with open(image_class_labels_file) as fin:
    lines = fin.readlines()
    for line in lines:
        id, class_label = list(map(int, line.strip().split(' ')))
        class_image_dic[class_label].append(id)

# For each session's test file
for i in range(sessions):
    session_test_file = open(osp.join(output_dir, 'test_%d.txt' % (i+1)), 'w')
    if i == 0:
        start_class = 1
        end_class = base_classes + 1
    else:
        start_class = base_classes + 1 + (i - 1) * inc_classes
        end_class = start_class + inc_classes
    for c in range(start_class, end_class):
        ids = class_image_dic[c]
        for id in ids:
            if label_dic[id] == 0:
                outline = image_dic[id]
                session_test_file.write(outline + '\n')
    session_test_file.close()
