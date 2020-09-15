import os.path as osp

root = './data/index_list/mini_imagenet'
sessions = 9

train_split_files = [osp.join(root, 'session_%d.txt' % (s + 1)) for s in range(sessions)]
test_data_file = osp.join(root, 'test.txt')

class_dic = dict()
test_file_list = list()
with open(test_data_file) as fin:
    test_file_list = fin.readlines()

for path in test_file_list:
    class_name = path.split('/')[2]
    if not class_dic.get(class_name):
        class_dic[class_name] = list()
    class_dic[class_name].append(path)

for s in range(sessions):
    class_list = list()
    data_list = list()
    with open(train_split_files[s]) as fin:
        data_list = fin.readlines()
    for path in data_list:
        class_name = path.split('/')[2]
        if class_name not in class_list:
            class_list.append(class_name)

    test_split_file = osp.join(root, 'test_%d.txt' % (s + 1))
    with open(test_split_file, 'w') as fout:
        for c in class_list:
            for path in class_dic[c]:
                fout.write(path)
