import jieba
import numpy as np
import pandas as pd
import pickle
import random
import re
import torch
import tqdm
from gensim.models.keyedvectors import KeyedVectors, Vocab
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer


def _init_fn(worker_id):
    np.random.seed(2021)


def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data


def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks


class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    # def load_data(self, path, shuffle):
    #     print("进入")
    #     self.data = df_filter(read_pkl(path))
    #     content = self.data['content'].to_numpy()
    #     label = torch.tensor(self.data['label'].astype(int).to_numpy())
    #     category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
    #     content_token_ids, content_masks = word2input(content, self.vocab_file, self.max_len)
    #     dataset = TensorDataset(content_token_ids,
    #                             content_masks,
    #                             label,
    #                             category
    #                             )
    #     dataloader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #         shuffle=shuffle,
    #         worker_init_fn=_init_fn
    #     )
    #     return dataloader

    def load_data(self, path, shuffle):
        print("读取数据")
        tweets_file_path = '/Users/daithyren/Downloads/MDFEND-Weibo21/rumor_detection_acl2017/twitter15/source_tweets.txt'
        labels_file_path = '/Users/daithyren/Downloads/MDFEND-Weibo21/rumor_detection_acl2017/twitter15/label.txt'

        tweets_df = pd.read_csv(tweets_file_path, sep='\t', header=None, names=['id', 'sentence'])
        labels_df = pd.read_csv(labels_file_path, sep=':', header=None, names=['label', 'id'])
        print("labels_df.shape:\n", labels_df.shape)

        valid_labels = ['unverified', 'non-rumor', 'true', 'false']
        labels_df = labels_df[labels_df['label'].isin(valid_labels)]

        # labels to integers
        label_mapping = {'unverified': 0, 'non-rumor': 1, 'true': 2, 'false': 3}
        labels_df['label_int'] = labels_df['label'].map(label_mapping)

        content = tweets_df['sentence'].to_numpy()
        label = torch.tensor(labels_df['label_int'].to_numpy())
        tweets_df['category'] = '文体娱乐'
        print(self.category_dict)
        category = torch.tensor(tweets_df['category'].apply(lambda c: self.category_dict[c]).to_numpy())

        content_token_ids, content_masks = word2input(content, self.vocab_file, self.max_len)

        print("content_token_ids.shape:\n", content_token_ids.shape)
        print("content_masks.shape:\n", content_masks.shape)
        print("label.shape:\n", label.shape)
        print("category.shape:\n", category.shape)

        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                label,
                                category
                                )
        # 8:1:1
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
        print("get 3 dataloader")
        return train_loader, val_loader, test_loader


class w2v_data():
    def __init__(self, max_len, batch_size, emb_dim, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.vocab_file = vocab_file
        self.category_dict = category_dict
        self.num_workers = num_workers

    def tokenization(self, content):
        pattern = "&nbsp;|展开全文|秒拍视频|O网页链接|网页链接"
        repl = ""
        tokens = []
        for c in content:
            c = re.sub(pattern, repl, c, count=0)
            cut_c = jieba.cut(c, cut_all=False)
            words = [word for word in cut_c]
            tokens.append(words)
        return tokens

    def get_mask(self, tokens):
        masks = []
        for token in tokens:
            if (len(token) < self.max_len):
                masks.append([1] * len(token) + [0] * (self.max_len - len(token)))
            else:
                masks.append([1] * self.max_len)

        return torch.tensor(masks)

    def encode(self, token_ids):
        w2v_model = KeyedVectors.load(self.vocab_file)
        embedding = []
        for token_id in token_ids:
            words = [w for w in token_id[: self.max_len]]
            words_vec = []
            for word in words:
                words_vec.append(w2v_model[word] if word in w2v_model else np.zeros([self.emb_dim]))
            for i in range(len(words_vec), self.max_len):
                words_vec.append(np.zeros([self.emb_dim]))
            embedding.append(words_vec)
        return torch.tensor(np.array(embedding, dtype=np.float32))

    # def load_data(self, path, shuffle = False):
    #     self.data = df_filter(read_pkl(path))
    #     content = self.data['content'].to_numpy()
    #     label = torch.tensor(self.data['label'].astype(int).to_numpy())
    #     category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
    #
    #     content_token_ids = self.tokenization(content)
    #     content_masks = self.get_mask(content_token_ids)
    #     emb_content = self.encode(content_token_ids)
    #
    #     dataset = TensorDataset(emb_content,
    #                             content_masks,
    #                             label,
    #                             category
    #                             )
    #     dataloader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #         shuffle=shuffle
    #     )
    #     return dataloader

    # 新增
    def load_data(self, path, shuffle=False):
        print("读取数据")
        tweets_file_path = '/Users/daithyren/Downloads/MDFEND-Weibo21/rumor_detection_acl2017/twitter15/source_tweets.txt'
        labels_file_path = '/Users/daithyren/Downloads/MDFEND-Weibo21/rumor_detection_acl2017/twitter15/label.txt'

        tweets_df = pd.read_csv(tweets_file_path, sep='\t', header=None, names=['id', 'sentence'])
        labels_df = pd.read_csv(labels_file_path, sep=':', header=None, names=['label', 'id'])

        valid_labels = ['unverified', 'non-rumor', 'true']
        labels_df = labels_df[labels_df['label'].isin(valid_labels)]

        # labels to integers
        label_mapping = {'unverified': 0, 'non-rumor': 1, 'true': 2}
        labels_df['label_int'] = labels_df['label'].map(label_mapping)

        content = tweets_df.to_numpy()
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        tweets_df['category'] = '通用'
        category = torch.tensor(tweets_df['category'].apply(lambda c: self.category_dict[c]).to_numpy())

        content_token_ids = self.tokenization(content)
        content_masks = self.get_mask(content_token_ids)
        emb_content = self.encode(content_token_ids)

        dataset = TensorDataset(emb_content,
                                content_masks,
                                label,
                                category
                                )

        # 8:1:1
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
        print("get dataloader")
        return train_loader, val_loader, test_loader
