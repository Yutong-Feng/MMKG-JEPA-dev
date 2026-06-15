import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data


# 消除 PyG 的自动偏移机制
class KGData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return 0  
        return super().__inc__(key, value, *args, **kwargs)


class KGDataset(Dataset):
    def __init__(self, data_dir="./data", dataset_name="DB15K", mode="train"):
        """
        初始化 DataLoader
        :param data_dir: 数据根目录
        :param dataset_name: 数据集名称 (如 'DB15K', 'MKG-W', 'MKG-Y')
        :param mode: 'train', 'valid', 或 'test'
        """
        self.file_path = os.path.join(data_dir, dataset_name, f"{mode}.txt")
        self.triples = []

        # 使用邻接表存储整张图的拓扑结构
        # adj[node] = [(neighbor_1, relation_1), (neighbor_2, relation_2), ...]
        self.adj = defaultdict(list)

        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    self.triples.append((h, r, t))
                    self.adj[h].append((t, r))
                    self.adj[t].append((h, r))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]

        # 获取 head_neighbor 时，去掉和 tail 实体连接的边（避免数据泄露）
        h_neighbors = [(n, rel) for n, rel in self.adj[h] if n != t]

        # 获取 tail_neighbor 时，去掉和 head 实体连接的边
        t_neighbors = [(n, rel) for n, rel in self.adj[t] if n != h]

        return (h, r, t), h_neighbors, t_neighbors


def kg_collate_fn(batch):
    """
    自定义的 collate_fn，用于将 batch 个三元组和变长的邻居列表打包成 PyG 支持的 Batch 格式。
    """
    triples = []
    head_graphs = []
    tail_graphs = []

    for (h, r, t), h_neigh, t_neigh in batch:
        # 1. 记录三元组
        triples.append([h, r, t])

        # 2. 构建 Head 邻居的 PyG Data 对象 (星型子图)
        if len(h_neigh) > 0:
            h_n, h_r = zip(*h_neigh)
            # edge_index 的 shape 为 [2, num_edges]
            # 这里构建中心节点 h 到邻居节点 h_n 的边
            edge_index = torch.tensor(
                [[h] * len(h_n), list(h_n)],  # Source nodes  # Target nodes
                dtype=torch.long,
            )
            edge_attr = torch.tensor(list(h_r), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        # 构建 head 时：
        head_graphs.append(
            KGData(
                edge_index=edge_index,
                edge_attr=edge_attr,
                center_node=torch.tensor([h]),
            )
        )

        # 3. 构建 Tail 邻居的 PyG Data 对象 (星型子图)
        if len(t_neigh) > 0:
            t_n, t_r = zip(*t_neigh)
            edge_index = torch.tensor([[t] * len(t_n), list(t_n)], dtype=torch.long)
            edge_attr = torch.tensor(list(t_r), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        tail_graphs.append(
            KGData(
                edge_index=edge_index,
                edge_attr=edge_attr,
                center_node=torch.tensor([t]),
            )
        )

    # 聚合
    batch_triples = torch.tensor(triples, dtype=torch.long)
    # 使用 PyG 的 Batch 将 list of Data 打包成一个大断开图，自动处理 node index offset
    head_batch = Batch.from_data_list(head_graphs)
    tail_batch = Batch.from_data_list(tail_graphs)

    return batch_triples, head_batch, tail_batch


if __name__ == "__main__":
    # 假设你的当前目录下有 ./data/DB15K/train.txt
    dataset = KGDataset(data_dir="./data", dataset_name="MKG-Y", mode="train")

    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=kg_collate_fn
    )

    # 迭代第一个 Batch 看看效果
    for batch_triples, head_batch, tail_batch in dataloader:
        print("=== Batch Triples <h, r, t> ===")
        print(batch_triples)
        print(f"Shape: {batch_triples.shape}\n")  # Expected: [batch_size, 3]

        print("=== Head Neighbors (PyG Batch Object) ===")
        print(head_batch)
        print(f"Edge Index: \n{head_batch.edge_index}")
        print(f"Edge Relations: \n{head_batch.edge_attr}")
        print(
            f"Batch Vector: \n{head_batch.batch}\n"
        )  # 指示每条边/节点属于 batch 中的哪一个图

        print("=== Tail Neighbors (PyG Batch Object) ===")
        print(tail_batch)

        break  # 仅测试一次
