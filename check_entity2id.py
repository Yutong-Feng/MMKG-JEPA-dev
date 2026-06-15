import os
import random
import requests
import time


def load_entity2id(filepath):
    id2entity = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                parts = line.strip().split(" ")  # 兼容空格分隔
            if len(parts) >= 2:
                # entity 格式为 <http://dbpedia.org/resource/...>
                entity, eid = parts[0], int(parts[-1])
                id2entity[eid] = entity
    return id2entity


def check_entity_mapping(data_dir, data_files, entity2id_file, sample_size):
    print("1. 加载 entity2id.txt...")
    id2entity = load_entity2id(os.path.join(data_dir, entity2id_file))
    print(f"共加载了 {len(id2entity)} 个实体映射。")

    dataset_entity_ids = set()
    sample_triplets = []

    print("2. 扫描数据集文件...")
    for file in [os.path.join(data_dir, f) for f in data_files]:
        if not os.path.exists(file):
            continue
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                h, r, t = map(int, line.strip().split())
                dataset_entity_ids.add(h)
                dataset_entity_ids.add(t)
                sample_triplets.append((h, r, t))

    # 基础检查：ID是否越界或缺失
    missing_ids = dataset_entity_ids - set(id2entity.keys())
    if missing_ids:
        print(f"⚠️ 警告: 数据集中有 {len(missing_ids)} 个ID在 entity2id.txt 中找不到！")
    else:
        print("✅ 基础检查通过: 数据集中的所有实体ID都在 entity2id.txt 中。")

    # 语义检查：通过 DBpedia SPARQL 端点验证
    print("\n3. 开始语义采样验证 (连接 DBpedia SPARQL)...")
    random.seed(42)
    samples = random.sample(sample_triplets, min(sample_size, len(sample_triplets)))

    valid_count = 0
    url = "http://dbpedia.org/sparql"

    for h, r, t in samples:
        h_uri = id2entity[h]
        t_uri = id2entity[t]

        # 构建 SPARQL 查询，查询两个实体之间的任意关系
        query = f"SELECT ?p WHERE {{ {h_uri} ?p {t_uri} }}"
        params = {"query": query, "format": "json"}

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            results = data.get("results", {}).get("bindings", [])

            if results:
                valid_count += 1
                relations = [res["p"]["value"] for res in results]
                print(
                    f"[匹配成功] ID({h}->{t}) 对应实体间存在关系 (例如: {relations[0]})"
                )
            else:
                print(f"[未找关系] ID({h}->{t}) -> {h_uri} 和 {t_uri} 之间查无关系。")

        except Exception as e:
            print(f"网络请求出错: {e}")

        time.sleep(0.5)  # 防止请求过快被 DBpedia 封IP

    success_rate = valid_count / len(samples)
    print(
        f"\n👉 验证结果: 抽样 {len(samples)} 个三元组，其中 {valid_count} 个在真实DBpedia中存在联系。"
    )
    if success_rate > 0.5:
        print("结论: 映射很可能是【正确】的。知识图谱通常是不完备的，不需要达到100%。")
    else:
        print("结论: 命中率过低，entity2id.txt 映射很可能是【错误/错位】的！")


# 运行检查 (请确保文件在同级目录或修改路径)
if __name__ == "__main__":
    check_entity_mapping(
        "data/DB15K",
        ["train.txt", "valid.txt", "test.txt"],
        "entity2id.txt",
        sample_size=100,
    )
