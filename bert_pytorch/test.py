import argparse

from torch.utils.data import DataLoader

from model import BERT
from trainer import BERTTrainer, ReconstructionBERTTrainer
from dataset import BERTDataset, WordVocab, DataReader, Vocab, my_collate, CustomBERTDataset
import pdb
import os
import json


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--dataset", required=True,
                    type=str, help="dataset")
parser.add_argument("--model_path", required=True,
                    type=str, help="ex)output/bert.model")

parser.add_argument("-hs", "--hidden", type=int,
                    default=256, help="hidden size of transformer model")
parser.add_argument("-l", "--layers", type=int,
                    default=8, help="number of layers")
parser.add_argument("-a", "--attn_heads", type=int,
                    default=8, help="number of attention heads")
parser.add_argument("-s", "--seq_len", type=int,
                    default=64, help="maximum sequence len")

parser.add_argument("-b", "--batch_size", type=int,
                    default=64, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int,
                    default=10, help="number of epochs")
parser.add_argument("-w", "--num_workers", type=int,
                    default=5, help="dataloader worker size")
parser.add_argument("--duplicate", type=int,
                    default=5, help="dataloader worker size")

parser.add_argument("--with_cuda", type=bool, default=True,
                    help="training with CUDA: true, or false")
parser.add_argument("--log_freq", type=int, default=10,
                    help="printing loss every n iter: setting n")
parser.add_argument("--corpus_lines", type=int,
                    default=None, help="total number of lines in corpus")
parser.add_argument("--cuda_devices", type=int, nargs='+',
                    default=None, help="CUDA device ids")
parser.add_argument("--on_memory", type=bool, default=True,
                    help="Loading on memory: true or false")

parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate of adam")
parser.add_argument("--adam_weight_decay", type=float,
                    default=0.01, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float,
                    default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float,
                    default=0.999, help="adam first beta value")
parser.add_argument("--weak_supervise", action="store_true")
parser.add_argument("--min_occur", type=int,
                    default=3, help="minimum of occurrence")
parser.add_argument("--use_sub_token", action="store_true")

args = parser.parse_args()

print("Load Data", args.dataset)
data_reader = DataReader(args.dataset, seq_len=args.seq_len,
                         use_sub_token=args.use_sub_token)
neg_data_reader = DataReader(
    args.dataset, graphs=data_reader.graphs, shuffle=True, duplicate=args.duplicate, seq_len=args.seq_len)
# print("Loading Vocab", args.vocab_path)
print("Loading Vocab")
vocab = Vocab(data_reader.graphs, min_occur=args.min_occur,
              use_sub_token=args.use_sub_token)
# vocab = WordVocab.load_vocab(args.vocab_path)
print("Vocab Size: ", len(vocab))

print("Shuffle Data")
'TODO'

print("Loading Train Dataset", args.dataset)
train_dataset = CustomBERTDataset(data_reader.graphs[:int(len(data_reader) * 0.8)], vocab, seq_len=args.seq_len,
                                  on_memory=args.on_memory, use_sub_token=args.use_sub_token)
# pdb.set_trace()
neg_train_dataset = CustomBERTDataset(neg_data_reader.graphs[:args.duplicate * len(train_dataset)], vocab, seq_len=args.seq_len,
                                      on_memory=args.on_memory)
# pdb.set_trace()
# assert len(neg_train_dataset) == args.duplicate * len(train_dataset)
# print("Loading Test Dataset", args.test_dataset)
print("Loading Dev Dataset", args.dataset)
test_dataset = CustomBERTDataset(data_reader.graphs[int(len(data_reader) * 0.8):], vocab, seq_len=args.seq_len,
                                 on_memory=args.on_memory, use_sub_token=args.use_sub_token)  # \
print(len(test_dataset))
neg_test_dataset = CustomBERTDataset(neg_data_reader.graphs[-args.duplicate * len(test_dataset):], vocab, seq_len=args.seq_len,
                                     on_memory=args.on_memory)  # \
# assert len(neg_test_dataset) == args.duplicate * len(test_dataset)
# if args.test_dataset is not None else None
# pdb.set_trace()
print("Creating Dataloader")
train_data_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)
neg_train_data_loader = DataLoader(
    neg_train_dataset, batch_size=args.batch_size * args.duplicate, num_workers=args.num_workers, collate_fn=my_collate)

test_data_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=my_collate)  # \
neg_test_data_loader = DataLoader(
    neg_test_dataset, batch_size=args.batch_size * args.duplicate, num_workers=args.num_workers, collate_fn=my_collate)  # \
# if test_dataset is not None else None
# assert False
# dataset is not None else None
# assert False
print("Building BERT model")
bert = BERT(len(vocab), hidden=args.hidden,
            n_layers=args.layers, attn_heads=args.attn_heads)

print("Creating BERT Trainer")
# trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
#                       lr=args.lr, betas=(
#                           args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
#                       with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, pad_index=vocab.pad_index)
trainer = ReconstructionBERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader, neg_train_dataloader=neg_train_data_loader, neg_test_dataloader=neg_test_data_loader,
                                    lr=args.lr, betas=(
    args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
    with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, pad_index=vocab.pad_index, model_path=args.model_path, weak_supervise=args.weak_supervise)
# raise NotImplementedError
# pdb.set_trace()
ids, stages = trainer.api()
print(stages)

stage2code = {}
for i, g in enumerate(test_dataset.graphs):
    if stages[i] not in stage2code:
        stage2code[stages[i]] = []
    stage2code[stages[i]].append(g)
    # try:
    #     if stages[i] == 5:
    #         print('=' * 20)
    #         print(g["context"])
    # except:
    #     pass
    # pdb.set_trace()
with open('./stage2code_subtoken_same_data.json', 'w') as fout:
    json.dump(stage2code, fout, ensure_ascii=False)
assert False
print("Training Start")
for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch, args.output_path)

    if test_data_loader is not None:
        trainer.test(epoch)
