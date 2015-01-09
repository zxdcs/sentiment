__author__ = 'zxd'


def read_word_vec(file):
    f = open(file, encoding='utf-8')
    f.readline()
    vector_map = {}
    for line in f:
        tokens = line.rstrip('\n ').split(' ')
        vector_map[tokens[0]] = [float(x) for x in tokens[1:]]
    return vector_map


def scale():
    f = open(r'../data/vectors_cbow', 'r', encoding='utf-8')
    fst_line = f.readline()
    words = []
    vecs = []
    for line in f:
        tokens = line.split()
        words.append(tokens[0])
        vecs.append([float(tok) for tok in tokens[1:]])
    f.close()
    from sklearn import preprocessing as pp

    min_max_scaler = pp.MinMaxScaler((min, max), copy='False')
    vec_scaled = min_max_scaler.fit_transform(vecs)
    f = open(r'../data/vectors_cbow01', 'w', encoding='utf-8')
    f.write(fst_line)
    for word, vec in zip(words, vec_scaled):
        data_str = word + ' ' + ' '.join(map('{0:.6f}'.format, vec)) + '\n'
        f.write(data_str)
    f.close()


if __name__ == '__main__':
    # vector = read_word_vec(r'../data/vectors_cbow')
    scale()