import os
import numpy as np
import re
import six
import jieba
import gc


# 获取文件夹下所有文件，包括子文件夹
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            # if os.path.splitext(file)[1] == '.jpeg':
            L.append(os.path.join(root, file))
    return L

def 比较2个句子中存在情况(setence_list_1,setence_list_2):#这里驶入的是2个句子，不是分割后的
    setence_list_1 = list(jieba.cut(setence_list_1))
    setence_list_2 = list(jieba.cut(setence_list_2))
    setence_1_exist_token = [2] * len(setence_list_1)
    setence_2_exist_token = [2] * len(setence_list_2)
    for i in range(len(setence_list_1)):
        if setence_list_1[i] in setence_list_2:
            setence_1_exist_token[i] = 1

    for i in range(len(setence_list_2)):
        if setence_list_2[i] in setence_list_1:
            setence_2_exist_token[i] = 1

    return (setence_1_exist_token),(setence_2_exist_token)


# 文本清理
def clean_text(text):
    regex = re.compile(u'[\u4e00-\u9fa5]*[^a-zA-Z]*[\u4e00-\u9fa5]* ')

    text = regex.sub('', text)  # 过滤任意非中文、非英文、非数字
    text = re.sub(r'\n|&nbsp|\xa0|\\xa0|\u3000|\\u3000|\\u0020|\u0020', '', text)  # 过滤回车换行空格等

    return text


# 同时切分字和词，使用jieba切分词
def get_split_text(text):
    sentence_list = []
    for sentence in text:
        sentence_list.append((" ".join(list(jieba.cut(sentence)))))
        sent = ""
        for char in sentence:
            sent += char
            sent += " "
        sentence_list.append(sent[:-1])
    return sentence_list


# 就是pad_sequences
def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        trunc = s[:maxlen]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        x[idx, :len(trunc)] = trunc
    return x


# 就是one_hot
def one_hot(array, depth=10):
    def label_smoothing(inputs, epsilon=0.1):
        K = np.shape(inputs)[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / K)

    result = label_smoothing(np.eye(depth)[array])
    #result = (np.eye(depth)[array])
    return result

def one_hot_small_memory(array, depth=10):
    temp_array = []
    for line in array:
        line_list = []
        for id in line:
            temp = [0]*depth
            temp[id - 1] = 1
            line_list.append(temp)
        temp_array.append(line_list)
    return temp_array


# 合并一个文件夹中的文本文件。注意：需要合并的每个文件的结尾要有换行符。
def merge_files(filedir, target_file):
    """
    合并一个文件夹中的文本文件。注意：需要合并的每个文件的结尾要有换行符。

    Args:
        filedir: 需要合并文件的文件夹
        target_file: 合并后的写入的目标文件
    """
    filenames = os.listdir(filedir)
    with open(target_file, 'a', encoding='utf-8') as f:
        for filename in filenames:
            filepath = os.path.join(filedir, filename)
            f.writelines(open(filepath, encoding='utf-8').readlines())


# 将一个大的数据集按比例切分为训练集、开发集、测试集
def partition_dataset(dataset, ratio):
    """将一个大的数据集按比例切分为训练集、开发集、测试集

    Args:
        dataset: 列表，原始数据集
        ratio: 三元组，训练集、开发集、测试集切分比例，每个元素为0-1之间的小数

    Returns: train, val, test, 表示训练集、开发集、测试集的三个列表
    """
    data_len = len(dataset)
    train_len = int(np.floor(data_len * ratio[0]))
    val_len = int(np.ceil(data_len * ratio[1]))
    test_len = data_len - train_len - val_len
    return dataset[:train_len], dataset[train_len: -test_len], dataset[-test_len:]


# 按n的大小分割list
def split_list_by_n(array_list, n, new_list=[]):
    if len(array_list) < n:
        new_list.append(array_list)
        return new_list
    else:
        new_list.append(array_list[:n])
        return split_list_by_n(array_list[n:], n)


# shuffle数据的
def shuffle_data(list_1, list_2):
    seed_int = np.random.randint(1, 17000000, 1)

    np.random.seed(seed_int)
    np.random.shuffle(list_1)
    np.random.seed(seed_int)
    np.random.shuffle(list_2)

    return list_1, list_2


def train_test_split(x, y, test_size=0.05, random_seed=17):
    x = np.array(x)
    y = np.array(y)

    np.random.seed(random_seed)
    np.random.shuffle(x)
    np.random.seed(random_seed)
    np.random.shuffle(y)

    # 根据设定的测试集样本比例，划分训练集、测试集
    cut = int(len(x) * (1 - test_size))

    x_train = x[:cut]
    y_train = y[:cut]

    x_test = x[cut:]
    y_test = y[cut:]

    del x, y
    gc.collect()

    # 将dataframe格式的数据转换为numpy array格式，便于调用函数计算
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test


# 这个代码段是用于打断序列，长度不变
def get_shuffle_data(features, labels):
    if len(features) != len(labels):
        print("特征和标签的长度不一致,已暂停")

    features_size = len(features)
    shuffle_indices = np.random.permutation(np.arange(features_size))

    shuffled_features_data = [features[i] for i in shuffle_indices]
    shuffled_labels_data = [labels[i] for i in shuffle_indices]

    return shuffled_features_data, shuffled_labels_data


def shuffle_data(list_1, list_2):
    seed_int = np.random.randint(1, 17000000, 1)

    np.random.seed(seed_int)
    np.random.shuffle(list_1)
    np.random.seed(seed_int)
    np.random.shuffle(list_2)

    return list_1, list_2


def get_splite_dataset(list_data, batch_size):
    """
    :param list_data:           list
    :param batch_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(list_data), batch_size):
        yield list_data[i:i + batch_size]

# 计算F1的，在untils文件夹的eval中
import collections
def compute_f1():
    gold_toks = "糖尿病肾病人感冒服什么药"
    pred_toks = "糖尿病肾病用什么降糖药"
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 10]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(x_train, y_train)
    print(x_test, y_test)

    f1 = compute_f1()
    print(f1)
