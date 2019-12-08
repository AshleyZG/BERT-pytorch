import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import random
import pdb
import numpy as np
import itertools
import re
# from torch.utils.data import

random.seed(1111)

STAGE_PAD = 0
WRANGLE = 1
EXPLORE = 2
MODEL = 3
EVALUATE = 4
IMPORT = 5

SPV_MODE = [STAGE_PAD, WRANGLE, EXPLORE, MODEL, EVALUATE, IMPORT]

wrangle_funcs = ['pandas.read_csv', 'pandas.read_csv.dropna', 'pandas.read_csv.fillna', 'pandas.DataFrame.fillna', 'pandas.read_csv.describe', 'pandas.DataFrame.describe', 'sklearn.datasets.load_iris', 'scipy.misc.imread',
                 'scipy.io.loadmat']

explore_funcs = ['seaborn.distplot', 'matplotlib.pyplot.show', 'matplotlib.pyplot.plot', 'matplotlib.pyplot.figure',
                 'seaborn.pairplot', 'seaborn.heatmap', 'seaborn.lmplot', 'scipy.interpolate.interp1d']
# 'matplotlib.pyplot.xlabel', 'matplotlib.pyplot.ylabel'
model_funcs = ['sklearn.cluster.KMeans',
               'sklearn.cross_validation.train_test_split',
               'sklearn.decomposition.PCA',
               'sklearn.ensemble.RandomForestClassifier',
               'sklearn.linear_model.LinearRegression',
               'sklearn.linear_model.LogisticRegression',
               'sklearn.model_selection.train_test_split',
               'sklearn.neighbors.KNeighborsClassifier',
               'sklearn.svm.SVC',
               'sklearn.tree.DecisionTreeClassifier']

evaluate_funcs = ['sklearn.metrics.confusion_matrix', 'sklearn.cross_validation.cross_val_score',
                  'sklearn.metrics.mean_squared_error', 'sklearn.model_selection.cross_val_score', 'scipy.stats.ttest_ind', 'sklearn.metrics.accuracy_score']


def split_func_name(func):
    """
    split function names
    eg. sklearn.metrics.pairwise.cosine_similarity -> [sklearn, metrics, pairwise, cosine, similarity]
    """
    if ' ' in func:
        return func.split()
    new_str = ''
    for i, l in enumerate(func):
        if i > 0 and l.isupper() and func[i - 1].islower():
            new_str += '.'
        elif i > 0 and i < len(func) - 1 and l.isupper() and func[i - 1].isupper() and func[i + 1].islower():
            new_str += '.'
        elif i > 0 and l.isdigit() and func[i - 1].isalpha():
            new_str += '.'
        elif i < len(func) - 1 and l.isalpha() and func[i - 1].isdigit():
            new_str += '.'
        else:
            pass
        new_str += l
    return re.split('\.|_|/', new_str.lower())


def cell_type(funcs, nodes=None):
    # pdb.set_trace()
    # print()
    if sum([1 for n in nodes if (n["type"] == 'Import' or n["type"] == 'ImportFrom')]) / len(nodes) > 0.3:
        return IMPORT
    if any([f in funcs for f in model_funcs]):
        return MODEL

    if any([f in funcs for f in explore_funcs]):
        return EXPLORE

    if any([f in funcs for f in evaluate_funcs]):
        return EVALUATE
    if any([f in funcs for f in wrangle_funcs]):
        return WRANGLE

    return STAGE_PAD


class DataReader(object):
    """docstring for DataReader"""

    def __init__(self, graph_path, graphs=None, shuffle=False, duplicate=1, seq_len=None, use_sub_token=False):
        super(DataReader, self).__init__()
        # self.arg = arg
        self.graph_path = graph_path
        self.duplicate = duplicate
        self.seq_len = seq_len
        self.use_sub_token = use_sub_token
        if graphs is None:

            graphs = []
            with open(graph_path, 'r', encoding='utf-8') as f:
                for l in tqdm(f):
                    if use_sub_token:
                        g = json.loads(l)
                        new_nodes = []
                        idx_map = {}
                        for i, n in enumerate(g["nodes"]):
                            if "value" not in n:
                                tokens = [n["type"]]
                            else:

                                tokens = split_func_name(n["value"])
                                tokens = [t for t in tokens if t]
                                # if len(tokens) > 1:
                                #     print(n["value"], tokens)
                            idx_map[i] = []
                            for t in tokens:
                                idx_map[i].append(len(new_nodes))
                                new_nodes.append(t)
                        g["new_nodes"] = new_nodes
                        g["idx_map"] = idx_map
                        graphs.append(g)
                        # raise NotImplementedError
                    else:
                        graphs.append(json.loads(l))
                    # if len(graphs) > 10000:
                    #     break
        if seq_len:
            # if use_sub_token:
            #     graphs = [g for g in graphs if len(
            #         g["nodes"]) + 1 <= seq_len]

            #     # raise NotImplementedError
            # else:
            #     # graphs = [g for g in graphs if len(
            #     #     g["nodes"]) + 1 <= seq_len and len(g["nodes"]) > seq_len / 2]
            graphs = [g for g in graphs if len(
                g["nodes"]) + 1 <= seq_len]

        graphs = graphs * duplicate
        if shuffle:
            random.shuffle(graphs)
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)


class Vocab(object):
    """docstring for Vocab"""
    mask_index = 0
    unk_index = 1
    pad_index = 2
    sos_index = 3
    eos_index = 4

    def __init__(self, graphs, use_sub_token=False, min_occur=3):
        super(Vocab, self).__init__()
        self.mask_index = 0
        self.unk_index = 1
        self.pad_index = 2
        self.sos_index = 3
        self.eos_index = 4
        self.use_sub_token = use_sub_token
        self.min_occur = min_occur
        counter = {}
        for g in tqdm(graphs):
            if use_sub_token:
                for n in g["new_nodes"]:
                    if n not in counter:
                        counter[n] = 0
                    counter[n] += 1
            else:
                for n in g["nodes"]:
                    token = n["type"] if "value" not in n else n["value"]
                    if token not in counter:
                        counter[token] = 0
                    counter[token] += 1
        idx2word = ["[MASK]", "[UNK]", "[PAD]", "[CLS]", "[SEP]"] + \
            [w for w in counter if counter[w] >= min_occur]
        word2idx = {w: i for i, w in enumerate(idx2word)}

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)


class CustomBERTDataset(Dataset):
    def __init__(self, graphs, vocab, seq_len, use_sub_token=False, encoding="utf-8", corpus_lines=None, on_memory=True, chunk_size=128, n_neg=5):
        self.vocab = vocab
        self.seq_len = seq_len if not use_sub_token else int(seq_len * 1.5)

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines

        self.encoding = encoding
        self.chunk_size = chunk_size
        self.use_sub_token = use_sub_token
        self.n_neg = n_neg
        graphs = [g for g in graphs if len(g["nodes"]) + 1 <= seq_len]
        if use_sub_token:
            graphs = list(sorted(graphs, key=lambda x: len(x["new_nodes"])))
            # raise NotImplementedError
        else:
            graphs = list(sorted(graphs, key=lambda x: len(x["nodes"])))
        graphs = [graphs[i:i + chunk_size]
                  for i in range(0, len(graphs), chunk_size)]
        random.shuffle(graphs)
        graphs = list(itertools.chain.from_iterable(graphs))
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)
        # return self.corpus_lines

    def __getitem__(self, item):
        def process_graph(graph):
            t1_random, t1_label = self.random_word(graph)
            t1 = [self.vocab.sos_index] + t1_random  # + [self.vocab.eos_index]
            t1_label = [self.vocab.pad_index] + \
                t1_label  # + [self.vocab.pad_index]
            segment_label = [1 for _ in range(len(t1))][:self.seq_len]
            bert_input = t1[:self.seq_len]
            bert_label = t1_label[:self.seq_len]
            # assert len(bert_input) == len(graph["new_nodes"]) + 1
            adj_mat = np.zeros((len(bert_input), len(bert_input)))
            # for i, n in enumerate(graph["nodes"]):
            #     if "children" not in n:
            #         continue
            #     for c in n["children"]:
            #         if self.use_sub_token:
            #             pass
            #             # for ii in graph["idx_map"][i]:
            #             #     for cc in graph["idx_map"][c]:
            #             #         if ii + 1 >= len(bert_input) or cc + 1 >= len(bert_input):
            #             #             continue
            #             #         # assert cc < len(graph["new_nodes"])
            #             #         adj_mat[ii + 1][cc + 1] = 1
            #             # adj_mat[[z + 1 for z in graph["idx_map"][i]],
            #             #         [z + 1 for z in graph["idx_map"][c]]] = 1
            #             # raise NotImplementedError
            #         else:
            #             adj_mat[i + 1][c + 1] = 1
            adj_mat[:len(
                graph["nodes"]) + 1 if not self.use_sub_token else len(graph["new_nodes"]) + 1, 0] = 1
            adj_mat[0, :len(
                graph["nodes"]) + 1 if not self.use_sub_token else len(graph["new_nodes"]) + 1] = 1
            adj_mat = np.ones((len(bert_input), len(bert_input)))

            stage = cell_type(graph["funcs"], nodes=graph["nodes"])
            output = {"bert_input": bert_input,
                      "bert_label": bert_label,
                      "segment_label": segment_label,
                      # "is_next": is_next_label,
                      "adj_mat": adj_mat,
                      "seq_len": len(bert_input),
                      "stage": stage}
            return output
        pos_output = process_graph(self.graphs[item])
        neg_ids = random.sample(range(len(self.graphs)), self.n_neg)

        neg_output = [process_graph(
            self.graphs[i]) for i in neg_ids]

        return pos_output, neg_output
        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, graph):
        if self.use_sub_token:
            nodes = graph["new_nodes"]
            tokens = [self.vocab.word2idx.get(
                n, self.vocab.unk_index) for n in nodes]
            output_label = [self.vocab.word2idx.get(
                n, self.vocab.unk_index) for n in nodes]
            return tokens, output_label
            # raise NotImplementedError
        else:
            nodes = graph["nodes"]
            tokens = [n["value"] if "value" in n else n["type"] for n in nodes]
            # tokens = [self.vocab.word2idx.get(
            #     t, self.vocab.unk_index) for t in tokens]
        type_tokens = set([n["type"] for n in graph["nodes"]])
        # pdb.set_trace()

        output_label = []

        for i, token in enumerate(tokens):
            if token in type_tokens:
                tokens[i] = self.vocab.word2idx.get(
                    token, self.vocab.unk_index)
                output_label.append(self.vocab.pad_index)
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.word2idx.get(
                        token, self.vocab.unk_index)

                output_label.append(self.vocab.word2idx.get(
                    token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.word2idx.get(
                    token, self.vocab.unk_index)
                output_label.append(self.vocab.pad_index)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]


def my_collate(batch):
    def collate_graphs(graphs):
        # pdb.set_trace()
        seq_len = max([item["seq_len"] for item in graphs])
        # print(seq_len)
        bert_input = []
        bert_label = []
        segment_label = []
        adj_mat = []
        stages = []
        for item in graphs:
            item["bert_input"] += [Vocab.pad_index] * \
                (seq_len - item["seq_len"])
            item["bert_label"] += [Vocab.pad_index] * \
                (seq_len - item["seq_len"])
            item["segment_label"] += [Vocab.pad_index] * \
                (seq_len - item["seq_len"])
            mat = np.zeros((seq_len, seq_len))
            mat[:item["adj_mat"].shape[0],
                :item["adj_mat"].shape[1]] = item["adj_mat"]
            # mat[:item["adj_mat"].shape[0], :item["adj_mat"].shape[1]
            #     ] = np.ones(item["adj_mat"].shape)
            bert_input.append(item["bert_input"])
            bert_label.append(item["bert_label"])
            segment_label.append(item["segment_label"])
            adj_mat.append(mat)
            stages.append(item["stage"])
        return {"bert_input": torch.tensor(bert_input),
                "bert_label": torch.tensor(bert_label),
                "segment_label": torch.tensor(segment_label),
                "adj_mat": torch.tensor(adj_mat),
                "stage": torch.tensor(stages)}
    pos_graphs = [item[0] for item in batch]
    neg_graphs = list(itertools.chain.from_iterable(
        [item[1] for item in batch]))
    # return batch
    return collate_graphs(pos_graphs), collate_graphs(neg_graphs)
    return batch
    # pdb.set_trace()


if __name__ == '__main__':
    data_reader = DataReader(
        "/homes/gws/gezhang/jupyter-notebook-analysis/graphs/cell_with_func.txt", use_sub_token=True)
    # for g in data_reader.graphs:
    #     print(len(g["new_nodes"]))
    # print([n["type"] for n in g["nodes"]])
    vocab = Vocab(data_reader.graphs, use_sub_token=True, min_occur=3)
    # pdb.set_trace()

    train_dataset = CustomBERTDataset(
        data_reader.graphs[:int(len(data_reader) * 0.8)], vocab, seq_len=64, use_sub_token=True)
    pdb.set_trace()
    # train_dataset.__getitem__(1)
    train_data_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=1, collate_fn=my_collate)
    n_stages = 0
    n_valid_stages = 0
    pdb.set_trace()

    for data in train_data_loader:
        # pdb.set_trace()

        n_valid_stages += data[0]["stage"][data[0]["stage"] == 4].shape[0]
        n_stages += data[0]["stage"].shape[0]
    print(n_stages)
    print(n_valid_stages)
    print(n_valid_stages / n_stages)
    pdb.set_trace()
    for data in train_data_loader:

        pdb.set_trace()
        print('hello')