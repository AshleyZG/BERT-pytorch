from .bert_graph import BERTGraph
import torch.nn as nn
from .bert import BERT
import pdb
import torch


class TopicBERT(nn.Module):
    """calculate distribution over topics from code snippet representation"""

    def __init__(self, bert: BERT, vocab_size, n_topics=5):
        super(TopicBERT, self).__init__()
        # self.arg = arg
        self.n_topics = n_topics
        # self.bert = bert
        self.bert_graph = BERTGraph(bert, vocab_size)
        self.dim_reduction = nn.Linear(bert.hidden, self.n_topics)

    def forward(self, x, segment_label, adj_mat):
        graph_vec = self.bert_graph(x, segment_label, adj_mat)
        topic_dist = self.dim_reduction(graph_vec)
        topic_dist = nn.Softmax(dim=1)(topic_dist)
        # pdb.set_trace()
        return topic_dist, graph_vec
        # raise NotImplementedError


class VAE(nn.Module):
    """docstring for VAE"""

    def __init__(self, bert: BERT, vocab_size, n_topics=5, weak_supervise=False):
        super(VAE, self).__init__()
        # self.arg = arg
        self.weak_supervise = weak_supervise
        self.n_topics = n_topics
        # self.bert = bert
        self.bert_graph = BERTGraph(bert, vocab_size)
        self.dim_reduction = nn.Linear(bert.hidden, self.n_topics)

        # self.topic_bert = TopicBERT(bert, vocab_size, n_topics)
        self.reconstruction = nn.Linear(n_topics, bert.hidden, bias=False)
        if weak_supervise:
            self.spv_stage_label = nn.Linear(n_topics, 6)
        # pdb.set_trace()

    def forward(self, x, neg_x, segment_label, neg_segment_label, adj_mat, neg_adj_mat, train):
        # topic_dist, graph_vec = self.topic_bert(x, segment_label, adj_mat)
        graph_vec = self.bert_graph(x, segment_label, adj_mat, train)
        topic_dist = self.dim_reduction(graph_vec)
        if self.weak_supervise:
            stage_vec = self.spv_stage_label(topic_dist)
        else:
            stage_vec = None
        topic_dist = nn.Softmax(dim=1)(topic_dist)

        reconstructed_vec = self.reconstruction(topic_dist)
        neg_graph_vec = self.bert_graph(
            neg_x, neg_segment_label, neg_adj_mat, train)
        return reconstructed_vec, graph_vec, neg_graph_vec, topic_dist, stage_vec
