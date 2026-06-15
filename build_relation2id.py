import os
from collections import Counter
import requests
import time

# ============================================================
# 噪声关系黑名单：这些关系在DBpedia中极度泛化，对推断无意义
# ============================================================
NOISE_PATTERNS = [
    "wikiPageWikiLink",
    "wikiPageRedirects",
    "wikiPageDisambiguates",
    "wikiPageExternalLink",
    "wikiPageInterLanguageLink",
    "isPrimaryTopicOf",
    "primaryTopic",
    "sameAs",
    "seeAlso",
    "subject",  # dcterms:subject 太宽泛
    "type",  # rdf:type 是类型而非关系
    "22-rdf-syntax",  # rdf内部
]


def is_noise(relation_uri: str) -> bool:
    # 拆解出 URI 最后的 predicate 部分
    predicate = relation_uri.split("/")[-1].split("#")[-1]

    # 针对特别容易误伤的词做精确匹配
    EXACT_NOISE = {"type", "subject", "sameAs", "seeAlso"}
    if predicate in EXACT_NOISE:
        return True

    # 其他长模式保留子串匹配
    return any(p in relation_uri for p in NOISE_PATTERNS if p not in EXACT_NOISE)


def load_entity2id(filepath):
    id2entity = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                parts = line.strip().split(" ")
            if len(parts) >= 2:
                entity, eid = parts[0], int(parts[-1])
                id2entity[eid] = entity
    return id2entity


def get_dbpedia_relations(h_uri, t_uri, timeout=10):
    # 防御性编程：确保 URI 被尖括号包裹
    if not h_uri.startswith("<"):
        h_uri = f"<{h_uri}>"
    if not t_uri.startswith("<"):
        t_uri = f"<{t_uri}>"

    url = "http://dbpedia.org/sparql"
    query = f"SELECT ?p WHERE {{ {h_uri} ?p {t_uri} }}"
    params = {"query": query, "format": "json"}
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
        relations = {res["p"]["value"] for res in bindings}
        # 过滤噪声，只保留有语义的关系
        clean = {r for r in relations if not is_noise(r)}
        return clean
    except Exception:
        return None  # None 表示请求本身失败，与空集区分


def infer_relation(
    r_id, pairs, id2entity, max_samples, min_support_ratio, retry_limit=3
):
    """
    对单个 relation ID 推断其真实URI。

    策略：
      - 优先选择URI较短的实体对（更可能是DBpedia核心实体）
      - 对每对实体查询语义关系（已过滤噪声）
      - 用加权投票：某关系在 >= min_support_ratio 比例的成功样本中出现，则纳入候选
      - 票数最高者胜出

    """
    # 按URI总长度升序排列，优先尝试"短URI"实体对
    valid_pairs = [(h, t) for h, t in set(pairs) if h in id2entity and t in id2entity]
    valid_pairs.sort(key=lambda p: len(id2entity[p[0]]) + len(id2entity[p[1]]))

    sample_pairs = valid_pairs[:max_samples]

    all_found = Counter()  # 关系 -> 出现的成功样本数
    success_count = 0  # 成功返回（非None）的查询数

    for h, t in sample_pairs:
        h_uri, t_uri = id2entity[h], id2entity[t]

        # 带重试的查询
        result = None
        for attempt in range(retry_limit):
            result = get_dbpedia_relations(h_uri, t_uri)
            if result is not None:
                break
            time.sleep(1.0)

        if result is None:
            # 网络持续失败，跳过这对
            continue

        success_count += 1
        for rel in result:
            all_found[rel] += 1

        time.sleep(0.4)

    if success_count == 0:
        return None, 0.0, 0.0, success_count  # 无有效查询，无法推断

    # 筛选出现比例超过阈值的候选
    threshold = max(1, int(success_count * min_support_ratio))
    candidates = [(rel, cnt) for rel, cnt in all_found.items() if cnt >= threshold]

    if not candidates:
        return None, 0.0, 0, success_count

    # 按出现次数降序排列，取最高票
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_rel, best_cnt = candidates[0]
    confidence = best_cnt / success_count

    return best_rel, confidence, best_cnt, success_count


def recover_relation2id(
    data_dir,
    entity2id_filename="entity2id.txt",
    output_filename="relation2id_recovered.txt",
    max_samples=50,
    min_support_ratio=0.4,
):
    entity2id_path = os.path.join(data_dir, entity2id_filename)
    output_path = os.path.join(data_dir, output_filename)

    print(f"1. 加载实体映射: {entity2id_path}")
    id2entity = load_entity2id(entity2id_path)
    print(f"   共加载 {len(id2entity)} 个实体。\n")

    rel_pairs = {}
    data_files = ["train.txt", "valid.txt", "test.txt"]

    print(f"2. 读取并合并三元组数据...")
    for filename in data_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"   ⚠️  找不到 {filename}，跳过。")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    h, r, t = map(int, parts[:3])
                    if r not in rel_pairs:
                        rel_pairs[r] = []
                    rel_pairs[r].append((h, t))

    total = len(rel_pairs)
    print(f"   发现 {total} 种 relation。\n")
    print(f"3. 开始逐一推断（每个relation最多查询{max_samples}对实体）...\n")

    results = {}  # r_id -> (uri, confidence, hit, total_queries, pairs_count)
    failed = []

    for idx, (r_id, pairs) in enumerate(sorted(rel_pairs.items()), 1):
        # 【新增】计算这个关系下有多少对不重复的实体
        pairs_count = len(set(pairs))

        uri, conf, hit, total_queries = infer_relation(
            r_id,
            pairs,
            id2entity,
            max_samples=max_samples,
            min_support_ratio=min_support_ratio,
        )

        if uri:
            formatted = f"<{uri}>"
            # 【修改】将 pairs_count 存入 results
            results[r_id] = (formatted, conf, hit, total_queries, pairs_count)
            flag = "✅" if conf >= 0.5 else "⚠️ "

            # 【修改】在控制台打印中显示 可用实体对数量
            print(
                f"  {flag} [{idx:3d}/{total}] Relation {r_id:3d} (可用实体对: {pairs_count:4d}) "
                f"-> {formatted}  "
                f"(hit {hit}/{total_queries}, confidence {conf:.0%})"
            )
        else:
            # 【修改】将 pairs_count 存入 results
            results[r_id] = ("<UNKNOWN>", 0.0, 0, 0, pairs_count)
            failed.append(r_id)
            # 【修改】在控制台打印中显示 可用实体对数量
            print(
                f"  ❌ [{idx:3d}/{total}] Relation {r_id:3d} (可用实体对: {pairs_count:4d}) -> 推断失败"
            )

    print(f"\n4. 保存结果到 {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        # 【修改】在表头增加 available_pairs 列
        f.write(
            "# relation_uri\trelation_id\tconfidence\thit/queried\tavailable_pairs\n"
        )
        for r_id in sorted(results.keys()):
            # 【修改】解包提取出 pairs_count 并写入文件
            uri, conf, hit, total_queries, pairs_count = results[r_id]
            f.write(
                f"{uri}\t{r_id}\t{conf:.4f}\t{hit}/{total_queries}\t{pairs_count}\n"
            )

    success_count = total - len(failed)
    print(f"\n🎉 完成！成功推断 {success_count}/{total} 个relation。")

    if failed:
        print(f"   以下 {len(failed)} 个 relation 推断失败，已标记为 <UNKNOWN>：")
        print(f"   {failed}")
    print(f"\n   💡 提示：输出文件中 confidence < 0.5 的条目建议人工复核。")


if __name__ == "__main__":
    DATASET_DIR = "data/DB15K"
    recover_relation2id(
        data_dir=DATASET_DIR,
        max_samples=50,  # 每个relation最多查询50对实体
        min_support_ratio=0.4,  # 关系在40%以上的成功样本中出现才纳入候选
    )
