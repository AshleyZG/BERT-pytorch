import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import random
import pdb
import numpy as np
import itertools
# from torch.utils.data import
from .prepare_data import DataReader, Vocab


class VAEDataset(Dataset):
    def __init__(self, graphs, vocab, seq_len, use_sub_token=False, encoding="utf-8", corpus_lines=None, on_memory=True, chunk_size=128, n_neg=5):
        self.vocab = vocab
        self.seq_len = seq_len
        self.n_neg = n_neg
        # self
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        # self.corpus_path = corpus_path
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.use_sub_token = use_sub_token
        graphs = [g for g in graphs if len(g["nodes"]) + 1 <= seq_len]
        # pdb.set_trace()
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
        # print(item)
        # t1, t2, is_next_label = self.random_sent(item)
        # pdb.set_trace()
        graph = self.graphs[item]
        neg_ids = random.sample(range(len(self.graphs)), self.n_neg)
        neg_graphs = [self.graphs[i] for i in neg_ids]
        t1_random, t1_label = self.random_word(graph)
        # print(t1_random)
        # t2_random, t2_label = self.random_word(t2)
        # print('-' * 20)
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random  # + [self.vocab.eos_index]
        # t2 = t2_random + [self.vocab.eos_index]
        # print('*' * 20)

        t1_label = [self.vocab.pad_index] + \
            t1_label  # + [self.vocab.pad_index]
        # t2_label = t2_label + [self.vocab.pad_index]

        # segment_label = ([1 for _ in range(len(t1))] +
        #                  [2 for _ in range(len(t2))])[:self.seq_len]
        segment_label = [1 for _ in range(len(t1))][:self.seq_len]
        # bert_input = (t1 + t2)[:self.seq_len]
        bert_input = t1[:self.seq_len]
        # bert_label = (t1_label + t2_label)[:self.seq_len]
        bert_label = t1_label[:self.seq_len]
        # print('+' * 20)

        # padding = [self.vocab.pad_index for _ in range(
        #     self.seq_len - len(bert_input))]
        # bert_input.extend(padding), bert_label.extend(
        #     padding), segment_label.extend(padding)
        # print('=' * 20)

        adj_mat = np.zeros((len(bert_input), len(bert_input)))
        for i, n in enumerate(graph["nodes"]):
            if "children" not in n:
                continue
            for c in n["children"]:
                adj_mat[i + 1][c + 1] = 1
        adj_mat[:len(graph["nodes"]) + 1, 0] = 1
        adj_mat[0, :len(graph["nodes"]) + 1] = 1
        adj_mat = np.ones((len(bert_input), len(bert_input)))
        # print(adj_mat)
        # print('!' * 20)
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  # "is_next": is_next_label,
                  "adj_mat": adj_mat,
                  "seq_len": len(bert_input)}
        # print(output)
        return output
        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, graph):
        if self.use_sub_token:
            raise NotImplementedError
        else:
            nodes = graph["nodes"]
            tokens = [n["value"] if "value" in n else n["type"] for n in nodes]
            # tokens = [self.vocab.word2idx.get(
            #     t, self.vocab.unk_index) for t in tokens]
        type_tokens = set([n["type"] for n in graph["nodes"]])
        # pdb.set_trace()

        output_label = []

        # for i, token in enumerate(tokens):
        #     if token in type_tokens:
        #         tokens[i] = self.vocab.word2idx.get(
        #             token, self.vocab.unk_index)
        #         output_label.append(self.vocab.pad_index)
        #         continue

        #     prob = random.random()
        #     if prob < 0.15:
        #         prob /= 0.15

        #         # 80% randomly change token to mask token
        #         if prob < 0.8:
        #             tokens[i] = self.vocab.mask_index

        #         # 10% randomly change token to random token
        #         elif prob < 0.9:
        #             tokens[i] = random.randrange(len(self.vocab))

        #         # 10% randomly change token to current token
        #         else:
        #             tokens[i] = self.vocab.word2idx.get(
        #                 token, self.vocab.unk_index)

        #         output_label.append(self.vocab.word2idx.get(
        #             token, self.vocab.unk_index))

        #     else:
        #         tokens[i] = self.vocab.word2idx.get(
        #             token, self.vocab.unk_index)
        #         output_label.append(self.vocab.pad_index)
        tokens = [self.vocab.word2idx.get(
            token, self.vocab.unk_index) for token in tokens]
        output_label = tokens
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
    seq_len = max([item["seq_len"] for item in batch])
    # print(seq_len)
    bert_input = []
    bert_label = []
    segment_label = []
    adj_mat = []
    for item in batch:
        item["bert_input"] += [Vocab.pad_index] * (seq_len - item["seq_len"])
        item["bert_label"] += [Vocab.pad_index] * (seq_len - item["seq_len"])
        item["segment_label"] += [Vocab.pad_index] * \
            (seq_len - item["seq_len"])
        mat = np.zeros((seq_len, seq_len))
        mat[:item["adj_mat"].shape[0], :item["adj_mat"].shape[1]] = item["adj_mat"]
        # mat[:item["adj_mat"].shape[0], :item["adj_mat"].shape[1]
        #     ] = np.ones(item["adj_mat"].shape)
        bert_input.append(item["bert_input"])
        bert_label.append(item["bert_label"])
        segment_label.append(item["segment_label"])
        adj_mat.append(mat)
    return {"bert_input": torch.tensor(bert_input),
            "bert_label": torch.tensor(bert_label),
            "segment_label": torch.tensor(segment_label),
            "adj_mat": torch.tensor(adj_mat)}
    return batch
    # pdb.set_trace()


if __name__ == '__main__':
    data_reader = DataReader(
        "/homes/gws/gezhang/jupyter-notebook-analysis/graphs.txt")
    vocab = Vocab(data_reader.graphs)
    train_dataset = CustomBERTDataset(
        data_reader.graphs[:int(len(data_reader) * 0.8)], vocab, seq_len=64)
    train_dataset.__getitem__(1)
    train_data_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=2, collate_fn=my_collate)
    # for data in train_data_loader:
    #     # pdb.set_trace()
    #     print(data)
    for data in train_data_loader:
        pdb.set_trace()
        print('hello')
