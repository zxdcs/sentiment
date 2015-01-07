__author__ = 'zxd'


def read_word_vec(file):
    f = open(file, encoding='utf-8')
    f.readline()
    vector_map = {}
    for line in f:
        tokens = line.rstrip('\n ').split(' ')
        vector_map[tokens[0]] = [float(x) for x in tokens[1:]]
    return vector_map


if __name__ == '__main__':
    vector = read_word_vec(r'../data/vectors_cbow')