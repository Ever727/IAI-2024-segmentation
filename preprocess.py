import os
import gensim
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset


def get_file_list(data_dir: str, exclude_file: str = "validation") -> list[str]:
    """
    返回数据集中除了指定文件之外的所有文本文件路径列表。

    Args:
        data_dir (str): 数据集所在目录。
        exclude_file (str, optional): 需要排除的文件名前缀。默认值为 "validation"。

    Returns:
        list[str]: 所有文本文件的完整路径列表。
    """
    files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt") and not filename.startswith(exclude_file):
            files.append(str(Path(data_dir) / filename))
    return files


def get_word2id() -> dict:
    """
    构建训练集和验证集中所有单词到ID的映射字典。

    Returns:
        dict: 单词到ID的映射字典。
    """
    data_files = get_file_list("Dataset")
    word2id = {}
    for file_path in data_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                sentence = line.strip().split()
                for word in sentence[1:]:
                    if word not in word2id:
                        word2id[word] = len(word2id)
    return word2id


def get_word2vec(
    word2id: dict, word2vec_file: str = "./Dataset/wiki_word2vec_50.bin"
) -> np.ndarray:
    """
    从预训练的词向量文件中,构建单词到向量的映射矩阵。

    Args:
        word2vec_file (str): 预训练词向量文件路径。
        word2id (dict): 单词到ID的映射字典。

    Returns:
        np.ndarray: 单词向量矩阵。
    """
    pre_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_file, binary=True
    )
    word2vecs = np.zeros((len(word2id) + 1, pre_model.vector_size))
    for word, idx in word2id.items():
        try:
            word2vecs[idx] = pre_model[word]
        except KeyError:
            pass
    return word2vecs


def get_corpus(
    data_file: str, word2id: dict, max_length: int = 70
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从给定的数据文件中读取语料库,并为每个样本生成输入文本和标签的 NumPy 数组。

    Args:
        data_file (str): 样本语料库的文件路径。
        word2id (dict): 单词到ID的映射字典。
        max_length (int, optional): 文本序列的最大长度。默认为 70。

    Returns:
        (np.ndarray, np.ndarray): 输入文本序列和对应的标签的 NumPy 数组。
    """
    contents, labels = [], []
    with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                parts = line.strip().split()
                label = int(parts[0])
                content = [word2id.get(word, 0) for word in parts[1:]][:max_length]
                padding = max(max_length - len(content), 0)
                content = np.pad(content, (0, padding), "constant", constant_values=0)
                contents.append(content)
                labels.append(label)
            except (ValueError, KeyError):
                continue

    return np.array(contents, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_data(max_length: int, batch_size: int):
    """
    加载训练集、验证集和测试集的 DataLoader。

    Args:
        max_length (int): 文本序列的最大长度。
        batch_size (int): DataLoader 的批大小。

    Returns:
        (DataLoader, DataLoader, DataLoader): 训练集、验证集和测试集的 DataLoader。
    """
    word2id = get_word2id()

    train_content, train_labels = get_corpus("./Dataset/train.txt", word2id, max_length)
    val_content, val_labels = get_corpus(
        "./Dataset/validation.txt", word2id, max_length
    )
    test_content, test_labels = get_corpus("./Dataset/test.txt", word2id, max_length)

    train_dataset = TensorDataset(
        torch.from_numpy(train_content), torch.from_numpy(train_labels)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_content), torch.from_numpy(val_labels)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_content), torch.from_numpy(test_labels)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
