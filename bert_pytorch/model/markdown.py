# coding=utf-8
import torch.nn as nn

import pdb
import torch


class BOW(nn.Module):
    """BOW for markdown header"""
    """
	calculate average embedding of words in header
	padding 2
    """

    def __init__(self, markdown_vocab_size, emb_size, padding_idx):
        super(BOW, self).__init__()
        # self.arg = arg
        self.markdown_vocab_size = markdown_vocab_size
        self.emb_size = emb_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            markdown_vocab_size, emb_size, padding_idx=padding_idx)

    def forward(self, markdown_label, markdown_len):
        # pdb.set_trace()
        token_embedding = self.embedding(markdown_label)
        sum_embedding = torch.sum(token_embedding, dim=1)
        avg_embedding = torch.div(sum_embedding.T, markdown_len).T
        return avg_embedding
        # pdb.set_trace()
        # raise NotImplementedError
