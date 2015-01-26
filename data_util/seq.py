__author__ = 'zxd'


def insert_space(file_list, data_list):
    assert len(file_list) == len(data_list)
    new_list = []
    for i in range(len(file_list)):
        new_list.append(data_list[i])
        if i + 1 < len(file_list) and file_list[i] != file_list[i + 1]:
            new_list.append('\n')
    return new_list


def insert_space_for_data():
    f = open(r'..\data\raw_data_balanced\filelist.txt')
    file_list = f.readlines()
    for i in range(len(file_list)):
        file_list[i] = file_list[i][:file_list[i].index('_')]
    f.close()

    sp_idx = 3701
    f = open(r'..\data\data_seq\seq_train_raw.txt')
    data_list = f.readlines()
    f.close()
    new_list = insert_space(file_list[:sp_idx], data_list)
    f = open(r'..\data\data_seq\seq_train_crf.txt', 'w')
    f.writelines(new_list)
    f.close()

    f = open(r'..\data\data_seq\seq_test_raw.txt')
    data_list = f.readlines()
    f.close()
    new_list = insert_space(file_list[sp_idx:], data_list)
    f = open(r'..\data\data_seq\seq_test_crf.txt', 'w')
    f.writelines(new_list)
    f.close()


if __name__ == '__main__':
    insert_space_for_data()