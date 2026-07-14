import argparse
from collections import deque
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency in some envs
    Chem = None
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold

from waveas_model import MODEL_REGISTRY, build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


def read_table(path):
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Only support .csv / .xlsx / .xls, got: {path}")


def infer_column(df, candidates, file_path):
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(f"None of columns {candidates} found in {file_path}. Available: {list(df.columns)}")


def infer_dataset_from_file_path(file_path):
    normalized_path = os.path.normpath(file_path)
    stem = os.path.splitext(os.path.basename(normalized_path))[0]
    lower_stem = stem.lower()
    parent = os.path.basename(os.path.dirname(normalized_path))
    lower_parent = parent.lower()

    if lower_stem.endswith("_dti_list"):
        return stem[:-len("_dti_list")]
    if lower_stem.endswith("_dtilist"):
        return stem[:-len("_dtilist")]

    if lower_stem in {"dti", "dti_list"} and lower_parent not in {"data", "dti_lists", "out", "new_out"}:
        return parent

    if lower_parent not in {"data", "dti_lists", "out", "new_out"}:
        return parent

    return stem


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_SET = set(AMINO_ACIDS)
ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se", "other"
]
ATOM_SYMBOL_INDEX = {symbol: idx for idx, symbol in enumerate(ATOM_SYMBOLS)}
HYBRIDIZATION_TYPES = []
if Chem is not None:
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]


def resolve_dataset_entity_file_paths(dataset, file_path):
    dataset_name = dataset or infer_dataset_from_file_path(file_path)
    candidates = []
    if dataset_name:
        candidates.extend(
            [
                os.path.join("Data", dataset_name),
                os.path.join("Data", "dti_lists", dataset_name),
            ]
        )

    dti_dir = os.path.dirname(os.path.abspath(file_path))
    candidates.append(dti_dir)
    if os.path.basename(dti_dir).lower() == "dti_lists":
        parent = os.path.dirname(dti_dir)
        if parent:
            candidates.append(parent)

    checked = []
    for base_dir in candidates:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        drug_candidates = [
            os.path.join(base_dir, "drugs.xlsx"),
            os.path.join(base_dir, "drugs.xls"),
            os.path.join(base_dir, "drug.csv"),
        ]
        target_candidates = [
            os.path.join(base_dir, "targets.xlsx"),
            os.path.join(base_dir, "targets.xls"),
            os.path.join(base_dir, "target.csv"),
        ]
        drug_path = next((p for p in drug_candidates if os.path.exists(p)), None)
        target_path = next((p for p in target_candidates if os.path.exists(p)), None)
        checked.append((base_dir, drug_path, target_path))
        if drug_path and target_path:
            return drug_path, target_path

    raise FileNotFoundError(
        "Cannot resolve drugs/targets table paths for online entity-graph encoding. "
        f"Checked: {[item[0] for item in checked]}"
    )


def first_non_empty_string(series):
    for value in series:
        if pd.notna(value):
            text = str(value).strip()
            if text:
                return text
    return ""


def _load_entity_text_table(path, id_candidates, text_candidates, output_id_name, output_text_name):
    df = read_table(path)
    text_col = infer_column(df, text_candidates, path)
    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    id_col = None
    for candidate in id_candidates:
        if candidate.lower() in normalized_cols:
            id_col = normalized_cols[candidate.lower()]
            break

    texts = df[text_col].fillna("").astype(str)
    if id_col is None:
        out = pd.DataFrame(
            {output_id_name: np.arange(len(df), dtype=np.int64), output_text_name: texts}
        )
    else:
        ids = df[id_col].fillna("").astype(str).str.strip()
        tmp = pd.DataFrame({output_id_name: ids, output_text_name: texts})
        out = (
            tmp.groupby(output_id_name, as_index=False)[output_text_name]
            .agg(first_non_empty_string)
            .reset_index(drop=True)
        )
    out[output_text_name] = out[output_text_name].fillna("").astype(str)
    return out


def load_drug_smiles_map(drug_path):
    table = _load_entity_text_table(
        path=drug_path,
        id_candidates=["drug_id", "drugid"],
        text_candidates=["SMILES", "smiles"],
        output_id_name="drug_id",
        output_text_name="SMILES",
    )
    return {int(k): str(v) for k, v in zip(table["drug_id"].tolist(), table["SMILES"].tolist())}


def load_target_fasta_map(target_path):
    table = _load_entity_text_table(
        path=target_path,
        id_candidates=["target_id", "targetid", "protein_id"],
        text_candidates=["protein_fastas", "FASTA", "TargetSequence", "target_fasta", "fastas"],
        output_id_name="target_id",
        output_text_name="FASTA",
    )
    return {int(k): str(v) for k, v in zip(table["target_id"].tolist(), table["FASTA"].tolist())}


def clean_protein_sequence(seq):
    seq = (seq or "").upper()
    return "".join(ch for ch in seq if ch in AMINO_ACID_SET)


def featurize_atom_for_graph(atom):
    symbol = atom.GetSymbol()
    symbol = symbol if symbol in ATOM_SYMBOL_INDEX else "other"
    symbol_onehot = np.zeros((len(ATOM_SYMBOLS),), dtype=np.float32)
    symbol_onehot[ATOM_SYMBOL_INDEX[symbol]] = 1.0

    degree_onehot = np.zeros((6,), dtype=np.float32)
    degree_onehot[min(int(atom.GetDegree()), 5)] = 1.0

    hyb_onehot = np.zeros((len(HYBRIDIZATION_TYPES) + 1,), dtype=np.float32)
    hyb = atom.GetHybridization()
    if hyb in HYBRIDIZATION_TYPES:
        hyb_onehot[HYBRIDIZATION_TYPES.index(hyb)] = 1.0
    else:
        hyb_onehot[-1] = 1.0

    numeric = np.asarray(
        [
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
            float(atom.GetMass()) / 200.0,
            float(atom.IsInRing()),
            float(atom.GetChiralTag()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([symbol_onehot, degree_onehot, hyb_onehot, numeric], axis=0).astype(np.float32)


def build_single_drug_entity_graph(smiles):
    if Chem is None:
        x = torch.zeros((1, 1), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return {"x": x, "edge_index": edge_index}
    smiles = (smiles or "").strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        x = torch.zeros((1, 1), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return {"x": x, "edge_index": edge_index}

    x = torch.from_numpy(np.stack([featurize_atom_for_graph(a) for a in mol.GetAtoms()], axis=0).astype(np.float32))
    edges = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        edges.append((i, j))
        edges.append((j, i))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return {"x": x, "edge_index": edge_index}


def build_single_target_entity_graph(
    seq,
    max_length=512,
    mode="contact",
    topk=8,
    contact_map=None,
    contact_threshold=0.5,
):
    seq = clean_protein_sequence(seq)
    if max_length and max_length > 0:
        seq = seq[:max_length]
    if not seq:
        x = torch.zeros((1, len(AMINO_ACIDS) + 1), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return {"x": x, "edge_index": edge_index}

    n = len(seq)
    x = np.zeros((n, len(AMINO_ACIDS) + 1), dtype=np.float32)
    aa_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for i, aa in enumerate(seq):
        x[i, aa_index[aa]] = 1.0
        x[i, -1] = float(i) / max(1, n - 1)

    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    graph_mode = (mode or "contact").lower()
    if graph_mode == "contact":
        if contact_map is None:
            raise ValueError("target graph mode=contact requires contact_map, but got None.")
        cm = np.asarray(contact_map, dtype=np.float32)
        cm = cm[:n, :n]
        mask = cm >= float(contact_threshold)
        rows, cols = np.where(mask)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i == j:
                continue
            edges.append((int(i), int(j)))
    if graph_mode == "approx":
        effective_topk = min(max(int(topk), 0), max(n - 1, 0))
        if effective_topk > 0:
            sims = x[:, :-1] @ x[:, :-1].T
            np.fill_diagonal(sims, -np.inf)
            for i in range(n):
                nbrs = np.argpartition(-sims[i], effective_topk)[:effective_topk]
                for j in nbrs.tolist():
                    if j == i:
                        continue
                    edges.append((i, int(j)))

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return {"x": torch.from_numpy(x), "edge_index": edge_index}


def build_entity_graphs_for_training(
    file_path,
    dataset,
    drug_id_map,
    target_id_map,
    target_mode="contact",
    target_topk=8,
    target_max_length=512,
    target_contact_lookup=None,
    target_contact_threshold=0.5,
):
    drug_path, target_path = resolve_dataset_entity_file_paths(dataset=dataset, file_path=file_path)
    drug_map = load_drug_smiles_map(drug_path)
    target_map = load_target_fasta_map(target_path)
    ordered_drug_ids = [drug_id for drug_id, _ in sorted(drug_id_map.items(), key=lambda kv: kv[1])]
    ordered_target_ids = [target_id for target_id, _ in sorted(target_id_map.items(), key=lambda kv: kv[1])]

    drug_graphs = [build_single_drug_entity_graph(drug_map.get(int(did), "")) for did in ordered_drug_ids]
    target_graphs = [
        build_single_target_entity_graph(
            target_map.get(int(tid), ""),
            max_length=target_max_length,
            mode=target_mode,
            topk=target_topk,
            contact_map=None if target_contact_lookup is None else target_contact_lookup.get(int(tid), None),
            contact_threshold=target_contact_threshold,
        )
        for tid in ordered_target_ids
    ]
    return drug_graphs, target_graphs, drug_path, target_path


def resolve_node_feature_path(node_feature_path, dataset, file_path, node_feature_dir):
    if node_feature_path:
        if not os.path.exists(node_feature_path):
            raise FileNotFoundError(f"Node feature file not found: {node_feature_path}")
        resolved_dataset = dataset or infer_dataset_from_file_path(file_path)
        return node_feature_path, resolved_dataset

    resolved_dataset = dataset or infer_dataset_from_file_path(file_path)
    if not resolved_dataset:
        raise ValueError("Cannot infer dataset name. Please provide --dataset or --node_feature_path.")

    if not os.path.isdir(node_feature_dir):
        raise FileNotFoundError(f"Node feature directory not found: {node_feature_dir}")

    all_npz = [name for name in os.listdir(node_feature_dir) if name.lower().endswith(".npz")]
    preferred = [f"{resolved_dataset}.npz", f"node_features_{resolved_dataset}.npz"]
    preferred_lower = {name.lower() for name in preferred}
    matched = [os.path.join(node_feature_dir, name) for name in all_npz if name.lower() in preferred_lower]

    if len(matched) == 1:
        return matched[0], resolved_dataset
    if len(matched) > 1:
        raise ValueError(
            f"Multiple node feature files match dataset '{resolved_dataset}' under {node_feature_dir}: {matched}"
        )

    raise FileNotFoundError(
        f"Cannot find node feature npz for dataset '{resolved_dataset}' under {node_feature_dir}. "
        f"Tried names: {preferred}. Available: {all_npz}"
    )


def load_positive_edges_flexible(file_path):
    df = read_table(file_path)
    drug_col = infer_column(df, ["drug_id", "drugid"], file_path)
    target_col = infer_column(df, ["target_id", "targetid", "protein_id"], file_path)

    label_col = None
    for candidate in ["label", "binding"]:
        key = candidate.lower()
        if key in {str(col).strip().lower() for col in df.columns}:
            label_col = infer_column(df, [candidate], file_path)
            break
    if label_col is not None:
        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
        df = df[labels == 1]

    df_unique = df[[drug_col, target_col]].drop_duplicates().reset_index(drop=True)
    drug_ids = sorted(df_unique[drug_col].astype(int).unique())
    target_ids = sorted(df_unique[target_col].astype(int).unique())
    drug_id_map = {drug_id: i for i, drug_id in enumerate(drug_ids)}
    target_id_map = {target_id: i for i, target_id in enumerate(target_ids)}

    pos_edges = []
    for _, row in df_unique.iterrows():
        pos_edges.append((drug_id_map[int(row[drug_col])], target_id_map[int(row[target_col])]))

    return pos_edges, drug_id_map, target_id_map, len(drug_ids), len(target_ids), df_unique


def load_precomputed_node_features(node_feature_path):
    data = np.load(node_feature_path, allow_pickle=True)
    if "drug_features" not in data or "target_features" not in data:
        raise ValueError(f"drug_features/target_features keys not found in {node_feature_path}")
    if "drug_ids" not in data or "target_ids" not in data:
        raise ValueError(f"drug_ids/target_ids keys not found in {node_feature_path}")

    drug_features = data["drug_features"].astype(np.float32)
    target_features = data["target_features"].astype(np.float32)
    drug_ids = data["drug_ids"]
    target_ids = data["target_ids"]
    return drug_features, target_features, drug_ids, target_ids


def load_precomputed_node_feature_bundle(node_feature_path):
    data = np.load(node_feature_path, allow_pickle=True)
    bundle = {}
    for key in data.files:
        bundle[key] = data[key]
    return bundle


def build_node_features_by_strategy(
    strategy,
    aligned_drug,
    aligned_target,
    seed,
):
    strategy = str(strategy).strip().lower()
    if strategy == "precomputed":
        final_drug = aligned_drug.astype(np.float32)
        final_target = aligned_target.astype(np.float32)
    elif strategy == "random":
        rng = np.random.default_rng(int(seed))
        final_drug = rng.standard_normal(aligned_drug.shape, dtype=np.float32)
        final_target = rng.standard_normal(aligned_target.shape, dtype=np.float32)
    elif strategy == "zero":
        final_drug = np.zeros_like(aligned_drug, dtype=np.float32)
        final_target = np.zeros_like(aligned_target, dtype=np.float32)
    else:
        raise ValueError("node_init_strategy must be one of: precomputed, random, zero")

    node_features = np.concatenate([final_drug, final_target], axis=0).astype(np.float32)
    return final_drug, final_target, node_features


def align_precomputed_features_to_dti(
    drug_id_map,
    target_id_map,
    drug_features,
    target_features,
    drug_ids,
    target_ids,
):
    if len(drug_ids) != drug_features.shape[0]:
        raise ValueError("drug_ids length does not match drug_features rows.")
    if len(target_ids) != target_features.shape[0]:
        raise ValueError("target_ids length does not match target_features rows.")

    drug_lookup = {int(entity_id): drug_features[idx] for idx, entity_id in enumerate(drug_ids)}
    target_lookup = {int(entity_id): target_features[idx] for idx, entity_id in enumerate(target_ids)}

    ordered_drug_ids = [drug_id for drug_id, _ in sorted(drug_id_map.items(), key=lambda item: item[1])]
    ordered_target_ids = [target_id for target_id, _ in sorted(target_id_map.items(), key=lambda item: item[1])]

    missing_drugs = [int(drug_id) for drug_id in ordered_drug_ids if int(drug_id) not in drug_lookup]
    missing_targets = [int(target_id) for target_id in ordered_target_ids if int(target_id) not in target_lookup]
    if missing_drugs:
        raise ValueError(f"Missing {len(missing_drugs)} drug ids in node features. Example: {missing_drugs[:5]}")
    if missing_targets:
        raise ValueError(f"Missing {len(missing_targets)} target ids in node features. Example: {missing_targets[:5]}")

    aligned_drug = np.stack([drug_lookup[int(drug_id)] for drug_id in ordered_drug_ids], axis=0).astype(np.float32)
    aligned_target = np.stack([target_lookup[int(target_id)] for target_id in ordered_target_ids], axis=0).astype(
        np.float32
    )
    if aligned_drug.shape[1] != aligned_target.shape[1]:
        raise ValueError(
            f"Drug feature dim ({aligned_drug.shape[1]}) and target feature dim ({aligned_target.shape[1]}) must match."
        )

    node_features = np.concatenate([aligned_drug, aligned_target], axis=0).astype(np.float32)
    return aligned_drug, aligned_target, node_features


def build_target_contact_lookup_from_bundle(bundle):
    if "target_contact_maps" not in bundle or "target_ids" not in bundle:
        return None
    target_ids = bundle["target_ids"]
    contact_maps = bundle["target_contact_maps"]
    if len(target_ids) != len(contact_maps):
        return None
    lookup = {}
    for idx, tid in enumerate(target_ids):
        try:
            key = int(tid)
        except Exception:
            continue
        lookup[key] = np.asarray(contact_maps[idx], dtype=np.float32)
    return lookup


def build_drug_dissimilarity_neighbors_from_features(drug_features, topk=10):
    num_drug = int(drug_features.shape[0])
    if num_drug <= 1:
        return np.zeros((num_drug, 0), dtype=np.int64)

    topk = int(max(1, min(topk, num_drug - 1)))
    normalized = l2_normalize_rows(drug_features.astype(np.float32))
    sim = np.matmul(normalized, normalized.T).astype(np.float32)
    np.fill_diagonal(sim, np.inf)

    candidate_cols = np.argpartition(sim, kth=topk - 1, axis=1)[:, :topk]
    candidate_scores = np.take_along_axis(sim, candidate_cols, axis=1)
    order = np.argsort(candidate_scores, axis=1)
    return np.take_along_axis(candidate_cols, order, axis=1).astype(np.int64)


def gmjrl_negative_mask_from_positive_mask(pos_mask, drug_dissimmat):
    neg_mask = np.zeros_like(pos_mask, dtype=bool)
    pos_drug_idx, pos_target_idx = np.where(pos_mask)
    for d, t in zip(pos_drug_idx.tolist(), pos_target_idx.tolist()):
        neg_mask[drug_dissimmat[d], t] = True
    return neg_mask


def make_kfold_edge_splits_gmjrl_style(pos_edges, num_drug, num_target, drug_dissimmat, n_splits=10, seed=42):
    pos_edges_arr = np.asarray(pos_edges, dtype=np.int64)
    if pos_edges_arr.ndim != 2 or pos_edges_arr.shape[1] != 2:
        raise ValueError("pos_edges should be a list/array of shape [num_pos, 2].")

    all_pos_mask = np.zeros((num_drug, num_target), dtype=bool)
    all_pos_mask[pos_edges_arr[:, 0], pos_edges_arr[:, 1]] = True
    unknown_mask = ~all_pos_mask

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(pos_edges_arr):
        train_pos = pos_edges_arr[train_idx]
        test_pos = pos_edges_arr[test_idx]

        train_pos_mask = np.zeros_like(all_pos_mask, dtype=bool)
        test_pos_mask = np.zeros_like(all_pos_mask, dtype=bool)
        train_pos_mask[train_pos[:, 0], train_pos[:, 1]] = True
        test_pos_mask[test_pos[:, 0], test_pos[:, 1]] = True

        train_neg_mask_candidate = gmjrl_negative_mask_from_positive_mask(train_pos_mask, drug_dissimmat)
        train_neg_mask = train_neg_mask_candidate & unknown_mask

        test_neg_mask_candidate = gmjrl_negative_mask_from_positive_mask(test_pos_mask, drug_dissimmat)
        test_neg_mask = test_neg_mask_candidate & unknown_mask & (~train_neg_mask)

        train_neg_drug, train_neg_target = np.where(train_neg_mask)
        test_neg_drug, test_neg_target = np.where(test_neg_mask)

        train_pos_edges = [tuple(edge) for edge in train_pos.tolist()]
        test_pos_edges = [tuple(edge) for edge in test_pos.tolist()]
        train_neg_edges = list(zip(train_neg_drug.tolist(), train_neg_target.tolist()))
        test_neg_edges = list(zip(test_neg_drug.tolist(), test_neg_target.tolist()))
        folds.append((train_pos_edges, test_pos_edges, train_neg_edges, test_neg_edges))
    return folds


def l2_normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return features / norms


def build_similarity_edges_from_features(features, topk=10, threshold=0.0):
    n = int(features.shape[0])
    if n <= 1:
        return []

    normalized = l2_normalize_rows(features.astype(np.float32))
    sim = np.matmul(normalized, normalized.T).astype(np.float32)
    np.fill_diagonal(sim, -np.inf)
    edge_to_weight = {}

    if topk is None or topk <= 0 or topk >= n:
        rows, cols = np.where(sim >= threshold)
        for i, j in zip(rows, cols):
            if i == j:
                continue
            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
            w = float(sim[i, j])
            prev = edge_to_weight.get((a, b), -np.inf)
            if w > prev:
                edge_to_weight[(a, b)] = w
    else:
        topk = min(topk, n - 1)
        candidate_cols = np.argpartition(sim, -topk, axis=1)[:, -topk:]
        for i in range(n):
            for j in candidate_cols[i]:
                score = float(sim[i, j])
                if score < threshold or i == j:
                    continue
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                prev = edge_to_weight.get((a, b), -np.inf)
                if score > prev:
                    edge_to_weight[(a, b)] = score

    edges = [(i, j, w) for (i, j), w in sorted(edge_to_weight.items())]
    return edges


def build_entity_graph_edge_index_from_features(features, topk=10, threshold=0.0):
    edges = build_similarity_edges_from_features(features, topk=topk, threshold=threshold)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    directed_edges = []
    for i, j, _ in edges:
        directed_edges.append((int(i), int(j)))
        directed_edges.append((int(j), int(i)))
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_training_adjacency_matrix(
    num_drug,
    num_target,
    train_pos_edges,
    drug_sim_edges=None,
    target_sim_edges=None,
    add_self_loops=True,
):
    num_nodes = num_drug + num_target
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for d, t in train_pos_edges:
        tg = t + num_drug
        adj[d, tg] = 1.0
        adj[tg, d] = 1.0

    for d1, d2, w in drug_sim_edges or []:
        if d1 == d2:
            continue
        ww = float(max(0.0, min(1.0, w)))
        if ww > adj[d1, d2]:
            adj[d1, d2] = ww
            adj[d2, d1] = ww

    for t1, t2, w in target_sim_edges or []:
        if t1 == t2:
            continue
        tg1 = t1 + num_drug
        tg2 = t2 + num_drug
        ww = float(max(0.0, min(1.0, w)))
        if ww > adj[tg1, tg2]:
            adj[tg1, tg2] = ww
            adj[tg2, tg1] = ww

    if add_self_loops:
        np.fill_diagonal(adj, 1.0)

    return torch.tensor(adj, dtype=torch.float32)


def adjacency_matrix_to_edge_index(adjacency_matrix):
    edge_index = torch.nonzero(adjacency_matrix > 0, as_tuple=False).t().contiguous()
    return edge_index


def compute_spectral_context_from_adjacency(adjacency_matrix, topk=16):
    if topk is None or topk <= 0:
        return None, None

    adjacency = adjacency_matrix.detach().cpu().numpy().astype(np.float64)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree)
    nonzero_mask = degree > 0
    inv_sqrt_degree[nonzero_mask] = 1.0 / np.sqrt(degree[nonzero_mask])
    normalized = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    laplacian = np.eye(adjacency.shape[0], dtype=np.float64) - normalized

    eigvals, eigvecs = np.linalg.eigh(laplacian)
    topk = int(min(topk, eigvals.shape[0]))
    eigvals = eigvals[:topk].astype(np.float32)
    eigvecs = eigvecs[:, :topk].astype(np.float32)
    return torch.from_numpy(eigvals), torch.from_numpy(eigvecs)


def compute_binary_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    try:
        aupr = average_precision_score(y_true, probs)
    except ValueError:
        aupr = float("nan")

    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sen": sensitivity,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": auc,
        "aupr": aupr,
        "recall": recall,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": probs,
        "roc_curve": roc_curve(y_true, probs),
        "pr_curve": precision_recall_curve(y_true, probs),
        "confusion_matrix": (tn, fp, fn, tp),
    }


def evaluate_model(model, edge_index, eval_edges, eval_labels, num_drug, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(edge_index, eval_edges, num_drug, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = eval_labels.cpu().numpy()
    return compute_binary_metrics(y_true=y_true, probs=probs, threshold=threshold)


def compute_relation_spectral_profiles(
    num_drug,
    num_target,
    train_pos_edges,
    drug_sim_edges,
    target_sim_edges,
    eigvals,
    eigvecs,
):
    if eigvals is None or eigvecs is None:
        return None

    num_nodes = num_drug + num_target
    relation_signals = {
        "drug-drug": np.zeros(num_nodes, dtype=np.float32),
        "target-target": np.zeros(num_nodes, dtype=np.float32),
        "drug-target": np.zeros(num_nodes, dtype=np.float32),
    }

    for d1, d2, w in drug_sim_edges or []:
        ww = float(max(0.0, w))
        relation_signals["drug-drug"][d1] += ww
        relation_signals["drug-drug"][d2] += ww

    for t1, t2, w in target_sim_edges or []:
        ww = float(max(0.0, w))
        relation_signals["target-target"][t1 + num_drug] += ww
        relation_signals["target-target"][t2 + num_drug] += ww

    for d, t in train_pos_edges:
        relation_signals["drug-target"][d] += 1.0
        relation_signals["drug-target"][t + num_drug] += 1.0

    eigvals_np = eigvals.detach().cpu().numpy().astype(np.float32)
    eigvecs_np = eigvecs.detach().cpu().numpy().astype(np.float32)
    energies = []
    cumulative = []
    labels = []

    for label, signal in relation_signals.items():
        signal = signal - signal.mean()
        norm = float(np.linalg.norm(signal))
        if norm > 0:
            signal = signal / norm
        coeff = eigvecs_np.T @ signal
        energy = np.square(coeff).astype(np.float32)
        total = float(energy.sum())
        if total > 0:
            energy = energy / total
        energies.append(energy)
        cumulative.append(np.cumsum(energy))
        labels.append(label)

    return {
        "eigvals": eigvals_np,
        "labels": labels,
        "energies": np.stack(energies, axis=0),
        "cumulative": np.stack(cumulative, axis=0),
    }


def compute_unweighted_shortest_path_distances(adjacency_matrix):
    adjacency = np.asarray(adjacency_matrix, dtype=np.float32)
    num_nodes = int(adjacency.shape[0])
    neighbor_lists = []
    for node_idx in range(num_nodes):
        neighbors = np.flatnonzero(adjacency[node_idx] > 0).astype(np.int64)
        neighbors = neighbors[neighbors != node_idx]
        neighbor_lists.append(neighbors.tolist())

    distances = np.full((num_nodes, num_nodes), float(num_nodes), dtype=np.float32)
    for source in range(num_nodes):
        distances[source, source] = 0.0
        queue = deque([source])
        while queue:
            current = queue.popleft()
            next_distance = distances[source, current] + 1.0
            for neighbor in neighbor_lists[current]:
                if distances[source, neighbor] <= next_distance:
                    continue
                distances[source, neighbor] = next_distance
                queue.append(neighbor)
    return distances


def compute_node_hop_response_matrix(filters, eigvecs, shortest_path_distances, max_hops=4):
    filter_bank = np.asarray(filters, dtype=np.float32)
    eigvecs_np = np.asarray(eigvecs, dtype=np.float32)
    dist_np = np.asarray(shortest_path_distances, dtype=np.float32)
    max_hops = int(max(1, max_hops))
    hop_profiles = []

    for filter_diag in filter_bank:
        kernel = eigvecs_np @ (filter_diag[:, None] * eigvecs_np.T)
        response_strength = np.abs(kernel).astype(np.float32)
        response_sum = response_strength.sum(axis=1)
        node_hops = np.zeros((response_strength.shape[0], max_hops + 2), dtype=np.float32)
        for hop_idx in range(0, max_hops + 2):
            if hop_idx == max_hops + 1:
                hop_mask = dist_np >= float(hop_idx)
            else:
                hop_mask = dist_np == float(hop_idx)
            hop_response = (response_strength * hop_mask.astype(np.float32)).sum(axis=1)
            node_hops[:, hop_idx] = hop_response / np.maximum(response_sum, 1e-8)
        hop_profiles.append(node_hops)

    if not hop_profiles:
        return np.zeros((0, dist_np.shape[0], max_hops + 2), dtype=np.float32)
    return np.stack(hop_profiles, axis=0)


def iterate_edge_batches(edge_pairs, labels, batch_size, shuffle=True):
    num_samples = len(edge_pairs)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_edges = [edge_pairs[i] for i in batch_indices]
        batch_labels = labels[batch_indices]
        yield batch_edges, batch_labels


def plot_cv_curves(fold_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    pr_path = os.path.join(output_dir, "pr_curve.png")

    plt.figure(figsize=(8, 6))
    for idx, result in enumerate(fold_results, start=1):
        fpr, tpr, _ = result["roc_curve"]
        plt.plot(fpr, tpr, lw=1.5, label=f"Fold {idx} (AUC={result['auc']:.4f})")

    all_y_true = np.concatenate([result["y_true"] for result in fold_results])
    all_y_score = np.concatenate([result["y_score"] for result in fold_results])
    overall_fpr, overall_tpr, _ = roc_curve(all_y_true, all_y_score)
    overall_auc = roc_auc_score(all_y_true, all_y_score)

    plt.plot(overall_fpr, overall_tpr, color="black", lw=2.5, label=f"Overall (AUC={overall_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for idx, result in enumerate(fold_results, start=1):
        precision_curve, recall_curve, _ = result["pr_curve"]
        plt.plot(recall_curve, precision_curve, lw=1.5, label=f"Fold {idx} (AUPR={result['aupr']:.4f})")

    overall_precision, overall_recall, _ = precision_recall_curve(all_y_true, all_y_score)
    overall_aupr = average_precision_score(all_y_true, all_y_score)

    plt.plot(overall_recall, overall_precision, color="black", lw=2.5, label=f"Overall (AUPR={overall_aupr:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()
    return roc_path, pr_path


def save_fold_results_csv(fold_results, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for fold_idx, result in enumerate(fold_results, start=1):
        rows.append(
            {
                "fold": fold_idx,
                "model_name": model_name,
                "best_epoch": result["best_epoch"],
                "best_selection_score": result["best_selection_score"],
                "selection_metric": result["selection_metric"],
                "num_train_pos": result["num_train_pos"],
                "num_train_neg": result["num_train_neg"],
                "num_test_pos": result["num_test_pos"],
                "num_test_neg": result["num_test_neg"],
                "acc": result["acc"],
                "f1": result["f1"],
                "precision": result["precision"],
                "sen": result["sen"],
                "mcc": result["mcc"],
                "auc": result["auc"],
                "aupr": result["aupr"],
                "recall": result["recall"],
            }
        )

    csv_path = os.path.join(output_dir, f"{model_name}_fold_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def save_metric_summary_csv(metric_summary, fold_results, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for metric_name, values in metric_summary.items():
        rows.append(
            {
                "model_name": model_name,
                "metric": metric_name,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }
        )

    rows.append(
        {
            "model_name": model_name,
            "metric": "best_epoch",
            "mean": float(np.mean([result["best_epoch"] for result in fold_results])),
            "std": float(np.std([result["best_epoch"] for result in fold_results])),
        }
    )

    csv_path = os.path.join(output_dir, f"{model_name}_metric_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def save_model_comparison_csv(comparison_summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for model_name, summary in comparison_summary.items():
        row = {"model_name": model_name}
        row.update(summary)
        rows.append(row)

    csv_path = os.path.join(output_dir, "model_comparison_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def save_run_config(output_dir, cli_args, runtime_context):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "cli_args": dict(cli_args),
        "runtime": dict(runtime_context),
    }
    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return config_path


def save_model_checkpoint(path, model, optimizer, epoch, metrics, model_config, extra_state=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "model_config": model_config,
    }
    if extra_state:
        checkpoint.update(extra_state)
    torch.save(checkpoint, path)
    return path


def save_waveas_spectral_visualizations(
    trace_records,
    output_dir,
    relation_profiles=None,
    node_influence_context=None,
    case_selection_context=None,
    influence_layout="spring",
    influence_layout_k=0.12,
    influence_layout_iterations=200,
    influence_layout_seed=42,
):
    if not trace_records:
        return {}

    def _piecewise_axis_transform(x, detail_max=0.56, detail_scale=2.8, tail_scale=0.55):
        x = np.asarray(x, dtype=np.float32)
        transformed = np.empty_like(x)
        low_mask = x <= detail_max
        transformed[low_mask] = x[low_mask] * detail_scale
        transformed[~low_mask] = detail_max * detail_scale + (x[~low_mask] - detail_max) * tail_scale
        return transformed

    def _smooth_curve(y, kernel=None):
        y = np.asarray(y, dtype=np.float32)
        if y.size < 5:
            return y
        if kernel is None:
            kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel = kernel / kernel.sum()
        padded = np.pad(y, (2, 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _compressed_heatmap_data(values, eigvals, detail_max=0.56, detail_step=0.1, high_bins=20):
        eigvals = np.asarray(eigvals)
        values = np.asarray(values)
        segments = []
        segment_centers = []

        detail_edges = list(np.arange(0.0, detail_max + 1e-8, detail_step))
        if len(detail_edges) == 0 or detail_edges[-1] < detail_max:
            detail_edges.append(detail_max)
        if len(detail_edges) == 1:
            detail_edges = [0.0, detail_max]

        for start, end in zip(detail_edges[:-1], detail_edges[1:]):
            if end == detail_edges[-1]:
                mask = (eigvals >= start) & (eigvals <= end)
            else:
                mask = (eigvals >= start) & (eigvals < end)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            segments.append(values[:, idx].mean(axis=1))
            segment_centers.append(float(eigvals[idx].mean()))

        high_mask = eigvals > detail_max
        if np.any(high_mask):
            high_idx = np.where(high_mask)[0]
            high_bins_eff = min(high_bins, len(high_idx))
            for chunk in np.array_split(high_idx, high_bins_eff):
                if len(chunk) == 0:
                    continue
                segments.append(values[:, chunk].mean(axis=1))
                segment_centers.append(float(eigvals[chunk].mean()))

        if not segments:
            return values, eigvals

        compressed = np.stack(segments, axis=1)
        centers = np.asarray(segment_centers, dtype=np.float32)
        return compressed, centers

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    epochs = np.asarray([record["epoch"] for record in trace_records], dtype=np.int64)
    eigvals = np.asarray(trace_records[-1]["eigvals"], dtype=np.float32)
    lambda_hat = np.asarray(trace_records[-1]["lambda_hat"], dtype=np.float32)
    final_filters = np.asarray(trace_records[-1]["filters"], dtype=np.float32)
    continuous_lambda = np.asarray(trace_records[-1]["continuous_lambda"], dtype=np.float32)
    continuous_filters = np.asarray(trace_records[-1]["continuous_filters"], dtype=np.float32)
    scaling_logits = np.asarray(trace_records[-1]["scaling_logits"], dtype=np.float32)
    wavelet_logits = np.asarray(trace_records[-1]["wavelet_logits"], dtype=np.float32)
    scaling_coeffs = np.asarray(trace_records[-1]["scaling_coeffs"], dtype=np.float32)
    wavelet_coeffs = np.asarray(trace_records[-1]["wavelet_coeffs"], dtype=np.float32)
    wavelet_scales = np.asarray(trace_records[-1]["wavelet_scales"], dtype=np.float32)
    learned_pre_s = np.asarray(trace_records[-1]["learned_pre_s"], dtype=np.float32)
    input_spectral_energy = np.asarray(trace_records[-1]["input_spectral_energy"], dtype=np.float32)
    output_spectral_energy = np.asarray(trace_records[-1]["output_spectral_energy"], dtype=np.float32)
    branch_output_norms = np.asarray(trace_records[-1]["branch_output_norms"], dtype=np.float32)

    np.savez(
        os.path.join(vis_dir, "waveas_spectral_trace.npz"),
        epochs=epochs,
        eigvals=eigvals,
        lambda_hat=lambda_hat,
        final_filters=final_filters,
        continuous_lambda=continuous_lambda,
        continuous_filters=continuous_filters,
        scaling_logits=scaling_logits,
        wavelet_logits=wavelet_logits,
        scaling_coeffs=scaling_coeffs,
        wavelet_coeffs=wavelet_coeffs,
        wavelet_scales=wavelet_scales,
        learned_pre_s=learned_pre_s,
        input_spectral_energy=input_spectral_energy,
        output_spectral_energy=output_spectral_energy,
    )

    lowfreq_zoom_path = os.path.join(vis_dir, "spectral_filter_curves_lowfreq_zoom.png")
    normalized_filters = final_filters / np.maximum(final_filters.max(axis=1, keepdims=True), 1e-8)
    labels = ["scaling"] + [f"wavelet_{idx}" for idx in range(1, final_filters.shape[0])]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    transformed_x = _piecewise_axis_transform(eigvals)
    if len(eigvals) > 1:
        dense_x = np.linspace(float(eigvals.min()), float(eigvals.max()), 400)
        dense_tx = _piecewise_axis_transform(dense_x)
    else:
        dense_x = eigvals.copy()
        dense_tx = transformed_x.copy()
    for idx, label in enumerate(labels):
        filter_y = normalized_filters[idx]
        if len(eigvals) > 1:
            dense_y = np.interp(dense_x, eigvals, filter_y)
            dense_y = _smooth_curve(dense_y)
        else:
            dense_y = filter_y
        ax.plot(dense_tx, dense_y, lw=2.0, label=label, alpha=0.95)
        ax.scatter(transformed_x, filter_y, s=16, alpha=0.65, zorder=3)

    tick_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.56, 0.76, 0.9, 1.0, 1.08]
    tick_values = [tick for tick in tick_values if float(eigvals.min()) <= tick <= float(eigvals.max())]
    if tick_values:
        ax.set_xticks(_piecewise_axis_transform(np.asarray(tick_values, dtype=np.float32)))
        ax.set_xticklabels([f"{tick:.2f}" if tick not in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.0} else f"{tick:.1f}" for tick in tick_values])
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Row-normalized response")
    ax.set_title("Row-normalized Spectral Filter Responses")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(lowfreq_zoom_path, dpi=160)
    plt.close()

    continuous_filter_path = os.path.join(vis_dir, "spectral_filter_continuous.png")
    normalized_continuous_filters = continuous_filters / np.maximum(continuous_filters.max(axis=1, keepdims=True), 1e-8)
    plt.figure(figsize=(8.8, 4.8))
    for idx, label in enumerate(labels):
        plt.plot(continuous_lambda, normalized_continuous_filters[idx], lw=2.0, label=label)
    plt.xlabel("Eigenvalue")
    plt.ylabel("Row-normalized response")
    plt.title("Continuous Spectral Filter Responses")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(continuous_filter_path, dpi=160)
    plt.close()

    continuous_filter_actual_range_path = os.path.join(vis_dir, "spectral_filter_continuous_actual_range.png")
    actual_max_eig = float(np.max(eigvals))
    actual_mask = continuous_lambda <= actual_max_eig + 1e-8
    plt.figure(figsize=(8.8, 4.8))
    for idx, label in enumerate(labels):
        plt.plot(
            continuous_lambda[actual_mask],
            normalized_continuous_filters[idx][actual_mask],
            lw=2.0,
            label=label,
        )
    plt.xlabel("Eigenvalue")
    plt.ylabel("Row-normalized response")
    plt.title(f"Continuous Spectral Filter Responses (up to max eig={actual_max_eig:.2f})")
    plt.xlim(0.0, actual_max_eig)
    plt.legend(frameon=False, fontsize=8)
    plt.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(continuous_filter_actual_range_path, dpi=160)
    plt.close()

    coeff_heatmap_path = os.path.join(vis_dir, "wavelet_coeff_heatmap.png")
    if wavelet_coeffs.size > 0:
        plt.figure(figsize=(7, 4))
        plt.imshow(wavelet_coeffs, aspect="auto", cmap="magma", interpolation="nearest")
        plt.colorbar(label="Coefficient")
        plt.yticks(range(wavelet_coeffs.shape[0]), [f"wavelet_{idx}" for idx in range(1, wavelet_coeffs.shape[0] + 1)])
        plt.xticks(range(wavelet_coeffs.shape[1]), [f"order_{idx}" for idx in range(wavelet_coeffs.shape[1])], rotation=30, ha="right")
        plt.title("Wavelet Chebyshev Coefficients")
        plt.tight_layout()
        plt.savefig(coeff_heatmap_path, dpi=160)
        plt.close()
    else:
        coeff_heatmap_path = None

    logits_path = os.path.join(vis_dir, "generator_logits_and_scales.png")
    plt.figure(figsize=(8, 5))
    x_scaling = np.arange(len(scaling_logits))
    plt.plot(x_scaling, scaling_logits, marker="o", lw=1.5, label="scaling logits")
    plt.plot(x_scaling, scaling_coeffs, marker="s", lw=1.5, label="scaling coeffs")
    if wavelet_logits.size > 0:
        offset = np.arange(wavelet_logits.shape[1])
        for idx in range(wavelet_logits.shape[0]):
            plt.plot(offset, wavelet_logits[idx], lw=1.2, linestyle="--", label=f"wavelet_{idx + 1} logits")
            plt.plot(offset, wavelet_coeffs[idx], lw=1.5, label=f"wavelet_{idx + 1} coeffs")
    if wavelet_scales.size > 0:
        plt.plot(np.arange(len(wavelet_scales)), wavelet_scales, marker="^", lw=1.6, label="wavelet scales")
    if learned_pre_s.size > 0:
        plt.plot(np.arange(len(learned_pre_s)), learned_pre_s, marker="d", lw=1.6, label="learned pre_s")
    plt.xlabel("Coefficient index")
    plt.ylabel("Value")
    plt.title("Generator Logits, Coefficients, and Wavelet Scales")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(logits_path, dpi=160)
    plt.close()

    heatmap_path = os.path.join(vis_dir, "spectral_filter_heatmap.png")
    compressed_filters, compressed_eigvals = _compressed_heatmap_data(normalized_filters, eigvals)
    plt.figure(figsize=(8, 4))
    plt.imshow(compressed_filters, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Row-normalized response")
    plt.yticks(range(compressed_filters.shape[0]), labels)
    if len(compressed_eigvals) <= 12:
        tick_idx = np.arange(len(compressed_eigvals))
    else:
        tick_idx = np.linspace(0, len(compressed_eigvals) - 1, num=12, dtype=int)
        tick_idx = np.unique(tick_idx)
    plt.xticks(tick_idx, [f"{compressed_eigvals[idx]:.2f}" for idx in tick_idx], rotation=45, ha="right")
    plt.xlabel("Eigenvalue")
    plt.title("Spectral Filter Heatmap (Row-normalized, Low-frequency Expanded)")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=160)
    plt.close()

    row_normalized_heatmap_path = os.path.join(vis_dir, "spectral_filter_heatmap_row_normalized.png")
    row_normalized_filters = normalized_filters
    compressed_row_filters, compressed_row_eigvals = _compressed_heatmap_data(row_normalized_filters, eigvals)
    plt.figure(figsize=(8, 4))
    plt.imshow(compressed_row_filters, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Row-normalized response")
    plt.yticks(range(compressed_row_filters.shape[0]), labels)
    if len(compressed_row_eigvals) <= 12:
        row_tick_idx = np.arange(len(compressed_row_eigvals))
    else:
        row_tick_idx = np.linspace(0, len(compressed_row_eigvals) - 1, num=12, dtype=int)
        row_tick_idx = np.unique(row_tick_idx)
    plt.xticks(row_tick_idx, [f"{compressed_row_eigvals[idx]:.2f}" for idx in row_tick_idx], rotation=45, ha="right")
    plt.xlabel("Eigenvalue")
    plt.title("Row-normalized Spectral Filter Heatmap (Low-frequency Expanded)")
    plt.tight_layout()
    plt.savefig(row_normalized_heatmap_path, dpi=160)
    plt.close()

    spectral_energy_path = os.path.join(vis_dir, "spectral_energy_before_after.png")
    plt.figure(figsize=(8, 5))
    plt.plot(eigvals, input_spectral_energy, marker="o", lw=1.6, label="input spectral energy")
    plt.plot(eigvals, output_spectral_energy, marker="s", lw=1.6, label="output spectral energy")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Energy")
    plt.title("Spectral Energy Before and After WaveAS Spectral Block")
    plt.legend()
    plt.tight_layout()
    plt.savefig(spectral_energy_path, dpi=160)
    plt.close()

    relation_curve_path = None
    relation_heatmap_path = None
    relation_cumulative_path = None
    relation_curve_nozero_path = None
    relation_heatmap_row_path = None
    node_influence_path = None
    node_kernel_matrix_path = None
    node_frequency_response_path = None
    node_frequency_basis_path = None
    node_frequency_group_curve_path = None
    node_frequency_centroid_path = None
    node_filter_preference_path = None
    node_filter_preference_group_path = None
    node_filter_preference_diff_path = None
    node_centroid_alignment_path = None
    node_hop_npz_path = None
    node_hop_filter_paths = {}
    node_filter_mean_hop_heatmap_path = None
    node_filter_mean_hop_relative_heatmap_path = None
    if relation_profiles is not None:
        rel_eigvals = relation_profiles["eigvals"]
        rel_labels = relation_profiles["labels"]
        rel_energies = relation_profiles["energies"]
        rel_cumulative = relation_profiles["cumulative"]

        relation_curve_path = os.path.join(vis_dir, "relation_spectral_energy_curves.png")
        plt.figure(figsize=(8.2, 4.8))
        for idx, label in enumerate(rel_labels):
            plt.plot(rel_eigvals, rel_energies[idx], marker="o", lw=1.6, label=label)
        plt.xlabel("Eigenvalue")
        plt.ylabel("Normalized spectral energy")
        plt.title("Relation-specific Spectral Energy")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(relation_curve_path, dpi=160)
        plt.close()

        nonzero_mask = rel_eigvals > 1e-8
        if np.any(nonzero_mask):
            relation_curve_nozero_path = os.path.join(vis_dir, "relation_spectral_energy_curves_no_zero.png")
            plt.figure(figsize=(8.2, 4.8))
            for idx, label in enumerate(rel_labels):
                plt.plot(
                    rel_eigvals[nonzero_mask],
                    rel_energies[idx][nonzero_mask],
                    marker="o",
                    lw=1.6,
                    label=label,
                )
            plt.xlabel("Eigenvalue")
            plt.ylabel("Normalized spectral energy")
            plt.title("Relation-specific Spectral Energy (Without λ=0)")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(relation_curve_nozero_path, dpi=160)
            plt.close()

        relation_heatmap_path = os.path.join(vis_dir, "relation_spectral_energy_heatmap.png")
        compressed_relation, compressed_relation_eigvals = _compressed_heatmap_data(rel_energies, rel_eigvals)
        plt.figure(figsize=(8, 3.8))
        plt.imshow(compressed_relation, aspect="auto", cmap="magma", interpolation="nearest")
        plt.colorbar(label="Normalized spectral energy")
        plt.yticks(range(len(rel_labels)), rel_labels)
        if len(compressed_relation_eigvals) <= 12:
            rel_tick_idx = np.arange(len(compressed_relation_eigvals))
        else:
            rel_tick_idx = np.linspace(0, len(compressed_relation_eigvals) - 1, num=12, dtype=int)
            rel_tick_idx = np.unique(rel_tick_idx)
        plt.xticks(rel_tick_idx, [f"{compressed_relation_eigvals[idx]:.2f}" for idx in rel_tick_idx], rotation=45, ha="right")
        plt.xlabel("Eigenvalue")
        plt.title("Relation-specific Spectral Energy Heatmap")
        plt.tight_layout()
        plt.savefig(relation_heatmap_path, dpi=160)
        plt.close()

        relation_heatmap_row_path = os.path.join(vis_dir, "relation_spectral_energy_heatmap_row_normalized.png")
        row_relation = rel_energies / np.maximum(rel_energies.max(axis=1, keepdims=True), 1e-8)
        compressed_row_relation, compressed_row_relation_eigvals = _compressed_heatmap_data(row_relation, rel_eigvals)
        plt.figure(figsize=(8, 3.8))
        plt.imshow(
            compressed_row_relation,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(label="Row-normalized spectral energy")
        plt.yticks(range(len(rel_labels)), rel_labels)
        if len(compressed_row_relation_eigvals) <= 12:
            row_rel_tick_idx = np.arange(len(compressed_row_relation_eigvals))
        else:
            row_rel_tick_idx = np.linspace(0, len(compressed_row_relation_eigvals) - 1, num=12, dtype=int)
            row_rel_tick_idx = np.unique(row_rel_tick_idx)
        plt.xticks(
            row_rel_tick_idx,
            [f"{compressed_row_relation_eigvals[idx]:.2f}" for idx in row_rel_tick_idx],
            rotation=45,
            ha="right",
        )
        plt.xlabel("Eigenvalue")
        plt.title("Relation-specific Spectral Energy Heatmap (Row-normalized)")
        plt.tight_layout()
        plt.savefig(relation_heatmap_row_path, dpi=160)
        plt.close()

        relation_cumulative_path = os.path.join(vis_dir, "relation_spectral_cumulative_energy.png")
        plt.figure(figsize=(8.2, 4.8))
        for idx, label in enumerate(rel_labels):
            plt.plot(rel_eigvals, rel_cumulative[idx], lw=1.8, label=label)
        plt.xlabel("Eigenvalue")
        plt.ylabel("Cumulative energy ratio")
        plt.title("Relation-specific Cumulative Spectral Energy")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(relation_cumulative_path, dpi=160)
        plt.close()

    if node_influence_context is not None:
        adjacency = node_influence_context["adjacency"]
        layout_adjacency = node_influence_context.get("layout_adjacency", adjacency)
        influence_filters = final_filters
        filter_titles = labels
        num_nodes = adjacency.shape[0]
        full_x = None
        full_y = None
        try:
            import networkx as nx

            graph = nx.Graph()
            edge_rows_all, edge_cols_all = np.where(np.triu(layout_adjacency > 0, k=1))
            graph.add_nodes_from(range(num_nodes))
            graph.add_edges_from(zip(edge_rows_all.tolist(), edge_cols_all.tolist()))
            if influence_layout == "kamada":
                pos = nx.kamada_kawai_layout(graph)
            else:
                pos = nx.spring_layout(
                    graph,
                    seed=int(influence_layout_seed),
                    k=float(influence_layout_k),
                    iterations=int(influence_layout_iterations),
                )
            coords = np.asarray([pos[idx] for idx in range(num_nodes)], dtype=np.float32)
            full_x = coords[:, 0]
            full_y = coords[:, 1]
        except Exception:
            influence_eigvecs = node_influence_context["eigvecs"]
            if influence_eigvecs.shape[1] > 1:
                full_x = influence_eigvecs[:, 1]
            else:
                full_x = np.linspace(-1.0, 1.0, num_nodes, dtype=np.float32)
            if influence_eigvecs.shape[1] > 2:
                full_y = influence_eigvecs[:, 2]
            elif influence_eigvecs.shape[1] > 1:
                full_y = influence_eigvecs[:, 0]
            else:
                full_y = np.zeros(num_nodes, dtype=np.float32)

        coords = np.stack([full_x, full_y], axis=1)
        center = coords.mean(axis=0, keepdims=True)
        radius = np.linalg.norm(coords - center, axis=1)
        candidate_threshold = np.quantile(radius, 0.8)
        candidate_nodes = np.where(radius >= candidate_threshold)[0]
        if candidate_nodes.size == 0:
            candidate_nodes = np.asarray([int(np.argmax(radius))], dtype=np.int64)
        source_node = int(candidate_nodes[len(candidate_nodes) // 2])

        cols = 2
        rows = int(np.ceil(len(filter_titles) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 4.8 * rows))
        axes = np.asarray(axes).reshape(-1)

        influence_eigvecs = node_influence_context["eigvecs"]
        impulse_coeff = influence_eigvecs[source_node]
        for idx, title in enumerate(filter_titles):
            ax = axes[idx]
            influence = influence_eigvecs @ (influence_filters[idx] * impulse_coeff)
            vmax = float(np.max(np.abs(influence)))
            vmax = max(vmax, 1e-6)
            abs_influence = np.abs(influence) / vmax
            strong_threshold = np.quantile(abs_influence, 0.975)
            medium_threshold = np.quantile(abs_influence, 0.925)
            strong_nodes = np.where(abs_influence >= strong_threshold)[0]
            medium_nodes = np.where((abs_influence >= medium_threshold) & (abs_influence < strong_threshold))[0]
            selected_nodes = np.unique(np.concatenate([strong_nodes, medium_nodes, np.asarray([source_node], dtype=np.int64)]))

            edge_rows, edge_cols = np.where(np.triu(adjacency > 0, k=1))
            for r, c in zip(edge_rows.tolist(), edge_cols.tolist()):
                ax.plot(
                    [full_x[r], full_x[c]],
                    [full_y[r], full_y[c]],
                    color="#8b8ba7",
                    alpha=0.035,
                    lw=0.5,
                    zorder=0,
                )
            ax.scatter(
                full_x,
                full_y,
                c="#5b567a",
                s=5,
                alpha=0.08,
                linewidths=0.0,
                zorder=0,
            )
            selected_set = set(selected_nodes.tolist())
            local_edges = [(r, c) for r, c in zip(edge_rows.tolist(), edge_cols.tolist()) if r in selected_set and c in selected_set]

            for r, c in local_edges:
                ax.plot(
                    [full_x[r], full_x[c]],
                    [full_y[r], full_y[c]],
                    color="#9089aa",
                    alpha=0.16,
                    lw=0.8,
                    zorder=1,
                )

            weak_nodes = np.setdiff1d(selected_nodes, np.union1d(strong_nodes, np.asarray([source_node], dtype=np.int64)))
            ax.scatter(
                full_x[weak_nodes],
                full_y[weak_nodes],
                c=influence[weak_nodes],
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                s=26,
                alpha=0.45,
                linewidths=0.0,
                zorder=2,
            )
            scatter = ax.scatter(
                full_x[strong_nodes],
                full_y[strong_nodes],
                c=influence[strong_nodes],
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                s=70,
                alpha=0.95,
                linewidths=0.15,
                edgecolors="#3a3a3a",
                zorder=3,
            )
            if weak_nodes.size > 0:
                ax.scatter(
                    full_x[weak_nodes],
                    full_y[weak_nodes],
                    s=52,
                    facecolors="none",
                    edgecolors="#58506b",
                    linewidths=0.45,
                    alpha=0.55,
                    zorder=2,
                )
            if strong_nodes.size > 0:
                ax.scatter(
                    full_x[strong_nodes],
                    full_y[strong_nodes],
                    s=108,
                    facecolors="none",
                    edgecolors="#2d273d",
                    linewidths=0.85,
                    alpha=0.95,
                    zorder=3,
                )
            ax.scatter(
                [full_x[source_node]],
                [full_y[source_node]],
                s=120,
                c="#ff4b2b",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        for idx in range(len(filter_titles), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(f"Single-source Node Influence Maps (source={source_node})", fontsize=15)
        plt.tight_layout()
        node_influence_path = os.path.join(vis_dir, "node_influence_maps.png")
        plt.savefig(node_influence_path, dpi=180)
        plt.close()

        eigvecs_np = node_influence_context["eigvecs"]
        sort_index = np.argsort(eigvecs_np[:, 0])
        max_nodes = min(140, eigvecs_np.shape[0])
        sort_index = sort_index[:max_nodes]
        cols = 2
        rows = int(np.ceil(len(filter_titles) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 4.8 * rows))
        axes = np.asarray(axes).reshape(-1)
        for idx, title in enumerate(filter_titles):
            ax = axes[idx]
            filter_diag = np.asarray(influence_filters[idx], dtype=np.float32)
            kernel = eigvecs_np @ (filter_diag[:, None] * eigvecs_np.T)
            kernel_show = kernel[np.ix_(sort_index, sort_index)]
            vmax = float(np.max(np.abs(kernel_show)))
            vmax = max(vmax, 1e-6)
            im = ax.imshow(
                kernel_show,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        for idx in range(len(filter_titles), len(axes)):
            axes[idx].axis("off")
        fig.suptitle("Node-to-node Spectral Kernel Matrices", fontsize=15)
        plt.tight_layout()
        node_kernel_matrix_path = os.path.join(vis_dir, "node_kernel_matrices.png")
        plt.savefig(node_kernel_matrix_path, dpi=180)
        plt.close()

        num_drug = int(node_influence_context.get("num_drug", 0))
        shortest_path_distances = compute_unweighted_shortest_path_distances(layout_adjacency)
        node_hop_matrix = compute_node_hop_response_matrix(
            filters=influence_filters,
            eigvecs=eigvecs_np,
            shortest_path_distances=shortest_path_distances,
            max_hops=4,
        )
        node_hop_mean = node_hop_matrix.mean(axis=(0, 2))
        if num_drug > 0 and num_drug < num_nodes:
            drug_order = np.argsort(node_hop_mean[:num_drug])
            target_order = np.argsort(node_hop_mean[num_drug:]) + num_drug
            node_hop_order = np.concatenate([drug_order, target_order])
        else:
            node_hop_order = np.argsort(node_hop_mean)

        ordered_hop_matrix = node_hop_matrix[:, node_hop_order, :]
        hop_vmin = 0.0
        hop_vmax = float(ordered_hop_matrix.max())
        if hop_vmax <= hop_vmin:
            hop_vmax = hop_vmin + 1e-6
        hop_labels = ["1-hop", "2-hop", "3-hop", "4+-hop"]
        hop_values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        node_filter_mean_hop = (node_hop_matrix * hop_values.reshape(1, 1, -1)).sum(axis=2)
        ordered_node_filter_mean_hop = node_filter_mean_hop[:, node_hop_order].T
        mean_hop_vmin = float(ordered_node_filter_mean_hop.min())
        mean_hop_vmax = float(ordered_node_filter_mean_hop.max())
        if mean_hop_vmax <= mean_hop_vmin:
            mean_hop_vmax = mean_hop_vmin + 1e-6
        ordered_node_filter_mean_hop_relative = (
            ordered_node_filter_mean_hop - ordered_node_filter_mean_hop.mean(axis=1, keepdims=True)
        )
        mean_hop_rel_absmax = float(np.max(np.abs(ordered_node_filter_mean_hop_relative)))
        mean_hop_rel_absmax = max(mean_hop_rel_absmax, 1e-6)

        node_hop_npz_path = os.path.join(vis_dir, "node_hop_response_matrices.npz")
        np.savez(
            node_hop_npz_path,
            hop_response_matrix=node_hop_matrix.astype(np.float32),
            ordered_hop_response_matrix=ordered_hop_matrix.astype(np.float32),
            node_filter_mean_hop=node_filter_mean_hop.astype(np.float32),
            ordered_node_filter_mean_hop=ordered_node_filter_mean_hop.astype(np.float32),
            ordered_node_filter_mean_hop_relative=ordered_node_filter_mean_hop_relative.astype(np.float32),
            node_order=node_hop_order.astype(np.int64),
            filter_labels=np.asarray(filter_titles),
            hop_labels=np.asarray(hop_labels),
            shortest_path_distances=shortest_path_distances.astype(np.float32),
        )

        plt.figure(figsize=(min(11.7, max(8.8, 0.012 * num_nodes + 6.4)), 3.8))
        im = plt.imshow(
            ordered_node_filter_mean_hop.T,
            aspect="auto",
            cmap="coolwarm",
            interpolation="nearest",
            vmin=mean_hop_vmin,
            vmax=mean_hop_vmax,
        )
        plt.colorbar(im, label="Average hop", fraction=0.022, pad=0.018, aspect=35)
        plt.yticks(range(len(filter_titles)), filter_titles)
        if num_drug > 0 and num_drug < num_nodes:
            plt.axvline(num_drug - 0.5, color="white", lw=1.0, alpha=0.9)
            plt.xticks(
                [max((num_drug - 1) / 2.0, 0.0), num_drug + max((num_nodes - num_drug - 1) / 2.0, 0.0)],
                ["drug", "target"],
            )
        else:
            tick_idx = np.linspace(0, num_nodes - 1, num=min(8, num_nodes), dtype=int)
            tick_idx = np.unique(tick_idx)
            plt.xticks(tick_idx, [str(node_hop_order[idx]) for idx in tick_idx], rotation=25, ha="right")
        plt.xlabel("Nodes")
        plt.ylabel("Filter branch")
        plt.title("Node-filter Average Hop Heatmap")
        plt.tight_layout()
        node_filter_mean_hop_heatmap_path = os.path.join(vis_dir, "node_filter_mean_hop_heatmap.png")
        plt.savefig(node_filter_mean_hop_heatmap_path, dpi=180)
        plt.close()

        plt.figure(figsize=(min(11.7, max(8.8, 0.012 * num_nodes + 6.4)), 3.8))
        im = plt.imshow(
            ordered_node_filter_mean_hop_relative.T,
            aspect="auto",
            cmap="coolwarm",
            interpolation="nearest",
            vmin=-mean_hop_rel_absmax,
            vmax=mean_hop_rel_absmax,
        )
        plt.colorbar(im, label="Row-centered average hop", fraction=0.022, pad=0.018, aspect=35)
        plt.yticks(range(len(filter_titles)), filter_titles)
        if num_drug > 0 and num_drug < num_nodes:
            plt.axvline(num_drug - 0.5, color="white", lw=1.0, alpha=0.9)
            plt.xticks(
                [max((num_drug - 1) / 2.0, 0.0), num_drug + max((num_nodes - num_drug - 1) / 2.0, 0.0)],
                ["drug", "target"],
            )
        else:
            tick_idx = np.linspace(0, num_nodes - 1, num=min(8, num_nodes), dtype=int)
            tick_idx = np.unique(tick_idx)
            plt.xticks(tick_idx, [str(node_hop_order[idx]) for idx in tick_idx], rotation=25, ha="right")
        plt.xlabel("Nodes")
        plt.ylabel("Filter branch")
        plt.title("Relative Node-filter Average Hop Heatmap")
        plt.tight_layout()
        node_filter_mean_hop_relative_heatmap_path = os.path.join(
            vis_dir,
            "node_filter_mean_hop_relative_heatmap.png",
        )
        plt.savefig(node_filter_mean_hop_relative_heatmap_path, dpi=180)
        plt.close()

        for filter_idx, title in enumerate(filter_titles):
            safe_title = str(title).strip().lower().replace(" ", "_").replace("-", "_")
            filter_values = ordered_hop_matrix[filter_idx]
            row_min = filter_values.min(axis=1, keepdims=True)
            row_max = filter_values.max(axis=1, keepdims=True)
            filter_values_display = (filter_values - row_min) / np.maximum(row_max - row_min, 1e-8)
            plt.figure(figsize=(5.4, max(5.2, min(16.0, 0.018 * num_nodes + 3.8))))
            im = plt.imshow(
                filter_values_display,
                aspect="auto",
                cmap="coolwarm",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
            plt.colorbar(im, label="Row-wise normalized hop response", fraction=0.022, pad=0.018, aspect=35)
            plt.xticks(range(len(hop_labels)), hop_labels, rotation=25, ha="right")
            if num_drug > 0 and num_drug < num_nodes:
                plt.axhline(num_drug - 0.5, color="white", lw=1.0, alpha=0.9)
                plt.yticks(
                    [max((num_drug - 1) / 2.0, 0.0), num_drug + max((num_nodes - num_drug - 1) / 2.0, 0.0)],
                    ["drug", "target"],
                )
            else:
                tick_idx = np.linspace(0, num_nodes - 1, num=min(10, num_nodes), dtype=int)
                tick_idx = np.unique(tick_idx)
                plt.yticks(tick_idx, [str(node_hop_order[idx]) for idx in tick_idx])
            plt.ylabel("Nodes")
            plt.xlabel("Hop bucket")
            plt.title(f"Node-hop Response Heatmap: {title}")
            plt.tight_layout()
            filter_path = os.path.join(vis_dir, f"node_hop_response_heatmap_{safe_title}.png")
            plt.savefig(filter_path, dpi=180)
            plt.close()
            node_hop_filter_paths[title] = filter_path

        basis_energy = np.square(eigvecs_np.T).astype(np.float32)
        basis_energy = basis_energy / np.maximum(basis_energy.max(axis=1, keepdims=True), 1e-8)

        def _display_map(values, gamma=0.55):
            values = np.asarray(values, dtype=np.float32)
            values = np.clip(values, 0.0, 1.0)
            return np.power(values, gamma)

        plt.figure(figsize=(10.5, 5.8))
        basis_show = _display_map(basis_energy)
        im = plt.imshow(
            basis_show,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(im, label="Enhanced normalized node-frequency response")
        if num_drug > 0 and num_drug < basis_show.shape[1]:
            plt.axvline(num_drug - 0.5, color="white", lw=1.0, alpha=0.8)
            tick_pos = [num_drug / 2.0, num_drug + (basis_show.shape[1] - num_drug) / 2.0]
            plt.xticks(tick_pos, ["drug", "target"])
        else:
            tick_idx = np.linspace(0, basis_show.shape[1] - 1, num=8, dtype=int)
            tick_idx = np.unique(tick_idx)
            plt.xticks(tick_idx, [str(idx_val) for idx_val in tick_idx])
        if eigvals.size <= 10:
            y_idx = np.arange(eigvals.size)
        else:
            y_idx = np.linspace(0, eigvals.size - 1, num=10, dtype=int)
            y_idx = np.unique(y_idx)
        plt.yticks(y_idx, [f"{eigvals[idy]:.2f}" for idy in y_idx])
        plt.xlabel("Nodes")
        plt.ylabel("Eigenvalue")
        plt.title("Node-frequency Basis Response Heatmap")
        plt.tight_layout()
        node_frequency_basis_path = os.path.join(vis_dir, "node_frequency_basis_heatmap.png")
        plt.savefig(node_frequency_basis_path, dpi=180)
        plt.close()

        if num_drug > 0 and num_drug < basis_energy.shape[1]:
            drug_basis = basis_energy[:, :num_drug]
            target_basis = basis_energy[:, num_drug:]

            drug_mean_curve = drug_basis.mean(axis=1)
            target_mean_curve = target_basis.mean(axis=1)
            curve_scale = max(float(drug_mean_curve.max()), float(target_mean_curve.max()), 1e-8)
            plt.figure(figsize=(8.6, 4.8))
            plt.plot(eigvals, drug_mean_curve / curve_scale, lw=2.0, label="drug")
            plt.plot(eigvals, target_mean_curve / curve_scale, lw=2.0, label="target")
            plt.xlabel("Eigenvalue")
            plt.ylabel("Normalized mean response")
            plt.title("Average Node-frequency Response by Node Type")
            plt.legend(frameon=False)
            plt.grid(alpha=0.18)
            plt.tight_layout()
            node_frequency_group_curve_path = os.path.join(vis_dir, "node_frequency_group_curves.png")
            plt.savefig(node_frequency_group_curve_path, dpi=180)
            plt.close()

            eigvals_col = eigvals.reshape(-1, 1)
            node_den = basis_energy.sum(axis=0) + 1e-8
            node_centroids = (eigvals_col * basis_energy).sum(axis=0) / node_den
            drug_centroids = node_centroids[:num_drug]
            target_centroids = node_centroids[num_drug:]
            bins = np.linspace(float(eigvals.min()), float(eigvals.max()), 22)
            plt.figure(figsize=(8.6, 4.8))
            plt.hist(drug_centroids, bins=bins, alpha=0.55, label="drug", color="#4c78a8", density=True)
            plt.hist(target_centroids, bins=bins, alpha=0.55, label="target", color="#f58518", density=True)
            plt.axvline(float(drug_centroids.mean()), color="#4c78a8", lw=1.8, linestyle="--")
            plt.axvline(float(target_centroids.mean()), color="#f58518", lw=1.8, linestyle="--")
            plt.xlabel("Frequency centroid")
            plt.ylabel("Density")
            plt.title("Node-frequency Centroid Distribution")
            plt.legend(frameon=False)
            plt.tight_layout()
            node_frequency_centroid_path = os.path.join(vis_dir, "node_frequency_centroid_histogram.png")
            plt.savefig(node_frequency_centroid_path, dpi=180)
            plt.close()

            branch_scores = np.asarray(branch_output_norms, dtype=np.float32)
            node_filter_scores = branch_scores / np.maximum(branch_scores.sum(axis=0, keepdims=True), 1e-8)

            drug_order = np.argsort(drug_centroids)
            target_order = np.argsort(target_centroids) + num_drug
            node_display_order = np.concatenate([drug_order, target_order])
            display_scores = node_filter_scores[:, node_display_order]
            display_vmin = float(np.quantile(display_scores, 0.02))
            display_vmax = float(np.quantile(display_scores, 0.98))
            if display_vmax <= display_vmin:
                display_vmin = float(display_scores.min())
                display_vmax = float(display_scores.max()) + 1e-8
            plt.figure(figsize=(10.5, 4.6))
            im = plt.imshow(
                display_scores,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
                vmin=display_vmin,
                vmax=display_vmax,
            )
            plt.colorbar(im, label="Filter preference probability (display-scaled)")
            plt.yticks(range(len(filter_titles)), filter_titles)
            plt.axvline(num_drug - 0.5, color="white", lw=1.0, alpha=0.8)
            tick_pos = [num_drug / 2.0, num_drug + (display_scores.shape[1] - num_drug) / 2.0]
            plt.xticks(tick_pos, ["drug", "target"])
            plt.xlabel("Nodes")
            plt.title("Node-to-filter Preference Heatmap (nodes sorted by raw centroid)")
            plt.tight_layout()
            node_filter_preference_path = os.path.join(vis_dir, "node_filter_preference_heatmap.png")
            plt.savefig(node_filter_preference_path, dpi=180)
            plt.close()

            drug_pref_mean = node_filter_scores[:, :num_drug].mean(axis=1)
            target_pref_mean = node_filter_scores[:, num_drug:].mean(axis=1)

            plt.figure(figsize=(7.6, 4.8))
            x = np.arange(len(filter_titles))
            width = 0.36
            plt.bar(x - width / 2, drug_pref_mean, width=width, label="drug", color="#4c78a8")
            plt.bar(x + width / 2, target_pref_mean, width=width, label="target", color="#f58518")
            plt.xticks(x, filter_titles)
            plt.ylabel("Mean filter preference")
            plt.title("Mean Filter Preference by Node Type")
            plt.legend(frameon=False)
            plt.tight_layout()
            node_filter_preference_group_path = os.path.join(vis_dir, "node_filter_preference_group_means.png")
            plt.savefig(node_filter_preference_group_path, dpi=180)
            plt.close()

            pref_diff = (drug_pref_mean - target_pref_mean).reshape(-1, 1)
            diff_absmax = float(np.max(np.abs(pref_diff)))
            diff_absmax = max(diff_absmax, 1e-6)
            plt.figure(figsize=(3.8, 4.6))
            im = plt.imshow(
                pref_diff,
                aspect="auto",
                cmap="coolwarm",
                interpolation="nearest",
                vmin=-diff_absmax,
                vmax=diff_absmax,
            )
            plt.colorbar(im, label="Drug mean - Target mean")
            plt.yticks(range(len(filter_titles)), filter_titles)
            plt.xticks([0], ["difference"])
            plt.title("Filter Preference Difference")
            plt.tight_layout()
            node_filter_preference_diff_path = os.path.join(vis_dir, "node_filter_preference_difference.png")
            plt.savefig(node_filter_preference_diff_path, dpi=180)
            plt.close()

            filter_centroids = []
            for filter_idx in range(influence_filters.shape[0]):
                filter_w = np.abs(np.asarray(influence_filters[filter_idx], dtype=np.float32))
                denom = float(filter_w.sum()) + 1e-8
                filter_centroids.append(float((eigvals * filter_w).sum() / denom))
            filter_centroids = np.asarray(filter_centroids, dtype=np.float32)
            learned_centroids = (filter_centroids.reshape(-1, 1) * node_filter_scores).sum(axis=0)
            plt.figure(figsize=(7.2, 5.8))
            plt.scatter(
                drug_centroids,
                learned_centroids[:num_drug],
                s=16,
                alpha=0.55,
                label="drug",
                c="#4c78a8",
            )
            plt.scatter(
                target_centroids,
                learned_centroids[num_drug:],
                s=16,
                alpha=0.55,
                label="target",
                c="#f58518",
            )
            diag_min = float(min(node_centroids.min(), learned_centroids.min()))
            diag_max = float(max(node_centroids.max(), learned_centroids.max()))
            plt.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="gray", lw=1.2)
            plt.xlabel("Raw frequency centroid")
            plt.ylabel("Model-learned frequency centroid")
            plt.title("Raw vs Learned Node Frequency Centroids")
            plt.legend(frameon=False)
            plt.tight_layout()
            node_centroid_alignment_path = os.path.join(vis_dir, "node_centroid_alignment.png")
            plt.savefig(node_centroid_alignment_path, dpi=180)
            plt.close()

        cols = 2
        rows = int(np.ceil(len(filter_titles) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(11.5, 4.8 * rows))
        axes = np.asarray(axes).reshape(-1)
        for idx, title in enumerate(filter_titles):
            ax = axes[idx]
            filter_response = np.asarray(influence_filters[idx], dtype=np.float32).reshape(-1, 1)
            node_frequency_response = np.abs(filter_response) * basis_energy
            node_frequency_response = node_frequency_response / np.maximum(
                node_frequency_response.max(axis=1, keepdims=True),
                1e-8,
            )
            node_frequency_show = _display_map(node_frequency_response)
            im = ax.imshow(
                node_frequency_show,
                aspect="auto",
                cmap="magma",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
            if num_drug > 0 and num_drug < node_frequency_response.shape[1]:
                ax.axvline(num_drug - 0.5, color="white", lw=1.0, alpha=0.8)
                tick_pos = [num_drug / 2.0, num_drug + (node_frequency_response.shape[1] - num_drug) / 2.0]
                ax.set_xticks(tick_pos)
                ax.set_xticklabels(["drug", "target"])
            else:
                tick_idx = np.linspace(0, node_frequency_response.shape[1] - 1, num=8, dtype=int)
                tick_idx = np.unique(tick_idx)
                ax.set_xticks(tick_idx)
                ax.set_xticklabels([str(idx_val) for idx_val in tick_idx], fontsize=8)
            if eigvals.size <= 10:
                y_idx = np.arange(eigvals.size)
            else:
                y_idx = np.linspace(0, eigvals.size - 1, num=10, dtype=int)
                y_idx = np.unique(y_idx)
            ax.set_yticks(y_idx)
            ax.set_yticklabels([f"{eigvals[idy]:.2f}" for idy in y_idx], fontsize=8)
            ax.set_xlabel("Nodes")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(title)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        for idx in range(len(filter_titles), len(axes)):
            axes[idx].axis("off")
        fig.suptitle("Node-frequency Response Heatmaps per Filter", fontsize=15)
        plt.tight_layout()
        node_frequency_response_path = os.path.join(vis_dir, "node_frequency_response_heatmap.png")
        plt.savefig(node_frequency_response_path, dpi=180)
        plt.close()

    return {
        "trace_npz": os.path.join(vis_dir, "waveas_spectral_trace.npz"),
        "filter_curves_lowfreq_zoom": lowfreq_zoom_path,
        "filter_curves_continuous": continuous_filter_path,
        "filter_curves_continuous_actual_range": continuous_filter_actual_range_path,
        "wavelet_coeff_heatmap": coeff_heatmap_path,
        "generator_logits_and_scales": logits_path,
        "filter_heatmap": heatmap_path,
        "filter_heatmap_row_normalized": row_normalized_heatmap_path,
        "spectral_energy": spectral_energy_path,
        "relation_spectral_energy_curves": relation_curve_path,
        "relation_spectral_energy_heatmap": relation_heatmap_path,
        "relation_spectral_energy_curves_no_zero": relation_curve_nozero_path,
        "relation_spectral_energy_heatmap_row_normalized": relation_heatmap_row_path,
        "relation_spectral_cumulative_energy": relation_cumulative_path,
        "node_influence_maps": node_influence_path,
        "node_kernel_matrices": node_kernel_matrix_path,
        "node_frequency_basis_heatmap": node_frequency_basis_path,
        "node_frequency_group_curves": node_frequency_group_curve_path,
        "node_frequency_centroid_histogram": node_frequency_centroid_path,
        "node_filter_preference_heatmap": node_filter_preference_path,
        "node_filter_preference_group_means": node_filter_preference_group_path,
        "node_filter_preference_difference": node_filter_preference_diff_path,
        "node_centroid_alignment": node_centroid_alignment_path,
        "node_frequency_response_heatmap": node_frequency_response_path,
        "node_hop_response_npz": node_hop_npz_path,
        "node_filter_mean_hop_heatmap": node_filter_mean_hop_heatmap_path,
        "node_filter_mean_hop_relative_heatmap": node_filter_mean_hop_relative_heatmap_path,
        "node_hop_response_filter_heatmaps": node_hop_filter_paths,
    }


def save_waveas_analysis_figures(
    trace_records,
    output_dir,
    node_influence_context=None,
    case_selection_context=None,
    influence_layout="spring",
    influence_layout_k=0.12,
    influence_layout_iterations=200,
    influence_layout_seed=42,
):
    if not trace_records:
        return {}

    def _smooth_curve(y):
        y = np.asarray(y, dtype=np.float32)
        if y.size < 5:
            return y
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel = kernel / kernel.sum()
        padded = np.pad(y, (2, 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for existing_name in os.listdir(vis_dir):
        existing_path = os.path.join(vis_dir, existing_name)
        if os.path.isfile(existing_path):
            os.remove(existing_path)

    epochs = np.asarray([record["epoch"] for record in trace_records], dtype=np.int64)
    eigvals = np.asarray(trace_records[-1]["eigvals"], dtype=np.float32)
    lambda_hat = np.asarray(trace_records[-1]["lambda_hat"], dtype=np.float32)
    final_filters = np.asarray(trace_records[-1]["filters"], dtype=np.float32)
    continuous_lambda = np.asarray(trace_records[-1]["continuous_lambda"], dtype=np.float32)
    continuous_filters = np.asarray(trace_records[-1]["continuous_filters"], dtype=np.float32)
    scaling_coeffs = np.asarray(trace_records[-1]["scaling_coeffs"], dtype=np.float32)
    wavelet_coeffs = np.asarray(trace_records[-1]["wavelet_coeffs"], dtype=np.float32)
    wavelet_scales = np.asarray(trace_records[-1]["wavelet_scales"], dtype=np.float32)
    input_spectral_energy = np.asarray(trace_records[-1]["input_spectral_energy"], dtype=np.float32)
    output_spectral_energy = np.asarray(trace_records[-1]["output_spectral_energy"], dtype=np.float32)
    labels = ["scaling"] + [f"wavelet_{idx}" for idx in range(1, final_filters.shape[0])]

    trace_npz_path = os.path.join(vis_dir, "waveas_trace.npz")
    np.savez(
        trace_npz_path,
        epochs=epochs,
        eigvals=eigvals,
        lambda_hat=lambda_hat,
        final_filters=final_filters,
        continuous_lambda=continuous_lambda,
        continuous_filters=continuous_filters,
        scaling_coeffs=scaling_coeffs,
        wavelet_coeffs=wavelet_coeffs,
        wavelet_scales=wavelet_scales,
        input_spectral_energy=input_spectral_energy,
        output_spectral_energy=output_spectral_energy,
    )

    spectral_response_path = os.path.join(vis_dir, "B_spectral_response_curves.png")
    normalized_continuous_filters = continuous_filters / np.maximum(
        continuous_filters.max(axis=1, keepdims=True),
        1e-8,
    )
    actual_max_eig = float(np.max(eigvals))
    actual_mask = continuous_lambda <= actual_max_eig + 1e-8
    plt.figure(figsize=(8.8, 4.8))
    for idx, label in enumerate(labels):
        plt.plot(
            continuous_lambda[actual_mask],
            normalized_continuous_filters[idx][actual_mask],
            lw=2.2,
            label=label,
        )
    plt.xlabel("Eigenvalue")
    plt.ylabel("Normalized spectral response")
    plt.title("B. Spectral Response Curves")
    plt.xlim(0.0, actual_max_eig)
    plt.legend(frameon=False, fontsize=8)
    plt.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(spectral_response_path, dpi=180)
    plt.close()

    if node_influence_context is None:
        return {
            "trace_npz": trace_npz_path,
            "B_spectral_response_curves": spectral_response_path,
        }

    adjacency = np.asarray(node_influence_context["adjacency"], dtype=np.float32)
    layout_adjacency = np.asarray(node_influence_context.get("layout_adjacency", adjacency), dtype=np.float32)
    eigvecs_np = np.asarray(node_influence_context["eigvecs"], dtype=np.float32)
    num_nodes = int(layout_adjacency.shape[0])
    num_drug = int(node_influence_context.get("num_drug", 0))
    shortest_path_distances = compute_unweighted_shortest_path_distances(layout_adjacency)
    hop_max = 4
    hop_matrix = compute_node_hop_response_matrix(
        filters=final_filters,
        eigvecs=eigvecs_np,
        shortest_path_distances=shortest_path_distances,
        max_hops=hop_max,
    )
    hop_labels = [f"{idx}-hop" for idx in range(0, hop_max + 1)] + [f"{hop_max + 1}+-hop"]
    hop_values = np.arange(hop_matrix.shape[-1], dtype=np.float32)
    avg_hop = (hop_matrix * hop_values.reshape(1, 1, -1)).sum(axis=2)
    hop95 = np.argmax(np.cumsum(hop_matrix, axis=2) >= 0.95, axis=2).astype(np.float32)
    degree = layout_adjacency.sum(axis=1) - np.diag(layout_adjacency)
    kernel_bank = np.asarray(
        [eigvecs_np @ (filter_diag[:, None] * eigvecs_np.T) for filter_diag in final_filters],
        dtype=np.float32,
    )

    hop_npz_path = os.path.join(vis_dir, "A_C_hop_statistics.npz")
    np.savez(
        hop_npz_path,
        hop_response_matrix=hop_matrix.astype(np.float32),
        average_hop=avg_hop.astype(np.float32),
        hop95=hop95.astype(np.float32),
        shortest_path_distances=shortest_path_distances.astype(np.float32),
        filter_labels=np.asarray(labels),
        hop_labels=np.asarray(hop_labels),
    )

    avg_hop_profile = hop_matrix.mean(axis=1)
    hop_profile_path = os.path.join(vis_dir, "A_hopwise_response_profile.png")
    plt.figure(figsize=(8.6, 4.8))
    x_hop = np.arange(len(hop_labels), dtype=np.float32)
    dense_x = np.linspace(0.0, float(len(hop_labels) - 1), 300)
    for idx, label in enumerate(labels):
        dense_y = np.interp(dense_x, x_hop, avg_hop_profile[idx])
        dense_y = _smooth_curve(dense_y)
        plt.plot(dense_x, dense_y, lw=2.2, label=label)
        plt.scatter(x_hop, avg_hop_profile[idx], s=20, alpha=0.7)
    plt.xticks(x_hop, hop_labels, rotation=20, ha="right")
    plt.xlabel("Hop bucket")
    plt.ylabel("Average normalized response")
    plt.title("A. Hop-wise Response Profile")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(hop_profile_path, dpi=180)
    plt.close()

    propagation_path = os.path.join(vis_dir, "C_node_type_specific_propagation.png")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    drug_slice = slice(0, num_drug)
    target_slice = slice(num_drug, num_nodes)
    box_colors = ["#4c78a8", "#f58518"]
    for ax_idx, (metric_matrix, metric_name) in enumerate(
        [(avg_hop, "Average propagation radius"), (hop95, "95% cumulative hop")]
    ):
        positions = []
        values = []
        colors = []
        pos_cursor = 1.0
        for filter_idx in range(metric_matrix.shape[0]):
            positions.extend([pos_cursor, pos_cursor + 0.32])
            values.extend([metric_matrix[filter_idx, drug_slice], metric_matrix[filter_idx, target_slice]])
            colors.extend(box_colors)
            pos_cursor += 1.0
        bp = axes[ax_idx].boxplot(
            values,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        tick_positions = [1.16 + idx for idx in range(len(labels))]
        axes[ax_idx].set_xticks(tick_positions)
        axes[ax_idx].set_xticklabels(labels, rotation=20, ha="right")
        axes[ax_idx].set_ylabel(metric_name)
        axes[ax_idx].set_title(metric_name)
        axes[ax_idx].grid(alpha=0.16, axis="y")
    handles = [
        plt.Line2D([0], [0], color=box_colors[0], lw=6),
        plt.Line2D([0], [0], color=box_colors[1], lw=6),
    ]
    fig.legend(handles, ["drug", "target"], loc="upper center", ncol=2, frameon=False)
    fig.suptitle("C. Node-type Specific Propagation", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(propagation_path, dpi=180)
    plt.close()

    candidate_csv_path = None
    selected_case_json_path = None
    case_study_path = None
    if case_selection_context is not None:
        eval_edges = [tuple(edge) for edge in case_selection_context["eval_edges"]]
        y_true = np.asarray(case_selection_context["y_true"]).astype(np.int64)
        y_pred = np.asarray(case_selection_context["y_pred"]).astype(np.int64)
        y_score = np.asarray(case_selection_context["y_score"]).astype(np.float32)
        candidate_rows = []
        correct_positive_idx = np.where((y_true == 1) & (y_pred == 1))[0]
        active_idx = correct_positive_idx if correct_positive_idx.size > 0 else np.where(y_true == 1)[0]
        for idx in active_idx.tolist():
            drug_id, target_id = eval_edges[idx]
            target_node = int(target_id + num_drug)
            drug_degree = float(degree[drug_id])
            target_degree = float(degree[target_node])
            center_node = int(drug_id if drug_degree >= target_degree else target_node)
            center_type = "drug" if center_node < num_drug else "target"
            counterpart_node = int(target_node if center_node == drug_id else drug_id)
            branch_ranges = avg_hop[:, center_node].astype(np.float32)
            spread = float(branch_ranges.max() - branch_ranges.min())
            sub_nodes = np.where(shortest_path_distances[center_node] <= 3.0)[0]
            if counterpart_node not in sub_nodes:
                sub_nodes = np.unique(np.concatenate([sub_nodes, np.asarray([counterpart_node], dtype=np.int64)]))
            candidate_rows.append(
                {
                    "drug_id": int(drug_id),
                    "target_id": int(target_id),
                    "label": int(y_true[idx]),
                    "prediction_score": float(y_score[idx]),
                    "drug_degree": drug_degree,
                    "target_degree": target_degree,
                    "subgraph_nodes": int(sub_nodes.size),
                    "center_node": center_node,
                    "center_type": center_type,
                    "counterpart_node": counterpart_node,
                    "range_scaling": float(branch_ranges[0]),
                    "range_wavelet_1": float(branch_ranges[1]) if len(branch_ranges) > 1 else np.nan,
                    "range_wavelet_2": float(branch_ranges[2]) if len(branch_ranges) > 2 else np.nan,
                    "range_wavelet_3": float(branch_ranges[3]) if len(branch_ranges) > 3 else np.nan,
                    "range_spread": spread,
                    "ranking_score": float(
                        y_score[idx]
                        + 0.03 * np.log1p(min(drug_degree, target_degree))
                        + 0.10 * spread
                        + 0.01 * sub_nodes.size
                    ),
                }
            )
        if candidate_rows:
            candidate_df = pd.DataFrame(candidate_rows).sort_values(
                by=["ranking_score", "prediction_score", "subgraph_nodes", "range_spread"],
                ascending=[False, False, False, False],
            )
            candidate_csv_path = os.path.join(vis_dir, "D_subgraph_case_top5_candidates.csv")
            candidate_df.head(5).to_csv(candidate_csv_path, index=False)

            best_case = {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in candidate_df.iloc[0].to_dict().items()
            }
            selected_case_json_path = os.path.join(vis_dir, "D_selected_case.json")
            with open(selected_case_json_path, "w", encoding="utf-8") as f:
                json.dump(best_case, f, ensure_ascii=False, indent=2)

            center_node = int(best_case["center_node"])
            local_nodes = np.where(shortest_path_distances[center_node] <= 3.0)[0]
            local_adj = layout_adjacency[np.ix_(local_nodes, local_nodes)]
            try:
                import networkx as nx

                graph = nx.Graph()
                graph.add_nodes_from(range(local_nodes.size))
                edge_rows, edge_cols = np.where(np.triu(local_adj > 0, k=1))
                graph.add_edges_from(zip(edge_rows.tolist(), edge_cols.tolist()))
                pos = nx.spring_layout(
                    graph,
                    seed=int(influence_layout_seed),
                    k=float(influence_layout_k),
                    iterations=int(influence_layout_iterations),
                )
                local_coords = np.asarray([pos[idx] for idx in range(local_nodes.size)], dtype=np.float32)
            except Exception:
                if eigvecs_np.shape[1] >= 2:
                    local_coords = eigvecs_np[local_nodes, :2].astype(np.float32)
                else:
                    local_coords = np.stack(
                        [np.linspace(-1.0, 1.0, local_nodes.size), np.zeros(local_nodes.size)],
                        axis=1,
                    ).astype(np.float32)

            local_x = local_coords[:, 0]
            local_y = local_coords[:, 1]
            center_local_idx = int(np.where(local_nodes == center_node)[0][0])
            local_edge_rows, local_edge_cols = np.where(np.triu(local_adj > 0, k=1))

            case_study_path = os.path.join(vis_dir, "D_single_source_propagation_influence.png")
            cols = 2
            rows = int(np.ceil(len(labels) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(11.2, 4.8 * rows))
            axes = np.asarray(axes).reshape(-1)
            for filter_idx, label in enumerate(labels):
                ax = axes[filter_idx]
                local_response = kernel_bank[filter_idx, center_node, local_nodes]
                vmax = float(np.max(np.abs(local_response)))
                vmax = max(vmax, 1e-6)
                abs_response = np.abs(local_response) / vmax
                node_sizes = 22.0 + 160.0 * np.power(abs_response, 0.85)
                for r, c in zip(local_edge_rows.tolist(), local_edge_cols.tolist()):
                    ax.plot(
                        [local_x[r], local_x[c]],
                        [local_y[r], local_y[c]],
                        color="#b4bcc8",
                        alpha=0.35,
                        lw=0.8,
                        zorder=0,
                    )
                scatter = ax.scatter(
                    local_x,
                    local_y,
                    c=local_response,
                    cmap="coolwarm",
                    vmin=-vmax,
                    vmax=vmax,
                    s=node_sizes,
                    alpha=0.92,
                    edgecolors="#2f3640",
                    linewidths=0.25,
                    zorder=2,
                )
                ax.scatter(
                    [local_x[center_local_idx]],
                    [local_y[center_local_idx]],
                    s=180,
                    c="#111111",
                    marker="*",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=4,
                )
                ax.set_title(f"{label} | avg hop={avg_hop[filter_idx, center_node]:.2f}")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
                cbar = plt.colorbar(scatter, ax=ax, fraction=0.04, pad=0.03)
                cbar.ax.tick_params(labelsize=8)
            for idx in range(len(labels), len(axes)):
                axes[idx].axis("off")
            fig.suptitle(
                f"D. Single-source Propagation Influence | source={best_case['center_type']} {center_node}",
                fontsize=13,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(case_study_path, dpi=180)
            plt.close()

    return {
        "trace_npz": trace_npz_path,
        "A_hopwise_response_profile": hop_profile_path,
        "A_C_hop_statistics_npz": hop_npz_path,
        "B_spectral_response_curves": spectral_response_path,
        "C_node_type_specific_propagation": propagation_path,
        "D_case_top5_candidates_csv": candidate_csv_path,
        "D_selected_case_json": selected_case_json_path,
        "D_single_source_propagation_influence": case_study_path,
    }


def train_one_fold(
    train_pos_edges,
    test_pos_edges,
    train_neg_edges,
    test_neg_edges,
    num_drug,
    num_target,
    node_features,
    drug_sim_edges,
    target_sim_edges,
    device="cuda",
    hidden_dim=128,
    dropout=0.2,
    num_scales=4,
    num_layers=2,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=None,
    shuffle=True,
    verbose=True,
    threshold=0.5,
    model_name="hpgnn",
    selection_metric="aupr",
    add_self_loops=True,
    use_spectral_context=False,
    spectral_topk=16,
    waveas_diversity_lambda=1e-2,
    waveas_wavelet_energy_lambda=1e-2,
    waveas_wavelet_balance_lambda=1e-1,
    waveas_wavelet_target_ratio=0.25,
    use_eigen_encoding=True,
    use_local_mpnn=False,
    use_trainable_graph_branch=False,
    graph_branch_layers=2,
    graph_branch_dropout=0.1,
    disable_post_filter_mlp=False,
    graph_branch_drug_graphs=None,
    graph_branch_target_graphs=None,
    visualization_output_dir=None,
    visualize_waveas_spectral=False,
    visualize_every=5,
    influence_layout="spring",
    influence_layout_k=0.12,
    influence_layout_iterations=200,
    influence_layout_seed=42,
):
    train_pos_edges = [tuple(edge) for edge in train_pos_edges]
    test_pos_edges = [tuple(edge) for edge in test_pos_edges]
    train_neg_edges = [tuple(edge) for edge in train_neg_edges]
    test_neg_edges = [tuple(edge) for edge in test_neg_edges]

    train_adj_matrix = build_training_adjacency_matrix(
        num_drug=num_drug,
        num_target=num_target,
        train_pos_edges=train_pos_edges,
        drug_sim_edges=drug_sim_edges,
        target_sim_edges=target_sim_edges,
        add_self_loops=add_self_loops,
    )
    train_edge_index = adjacency_matrix_to_edge_index(train_adj_matrix)
    spectral_eigvals, spectral_eigvecs = None, None
    relation_profiles = None
    node_influence_context = None
    if use_spectral_context and model_name == "WaveAS":
        spectral_eigvals, spectral_eigvecs = compute_spectral_context_from_adjacency(
            train_adj_matrix,
            topk=spectral_topk,
        )
        if visualize_waveas_spectral and model_name == "WaveAS":
            relation_profiles = compute_relation_spectral_profiles(
                num_drug=num_drug,
                num_target=num_target,
                train_pos_edges=train_pos_edges,
                drug_sim_edges=drug_sim_edges,
                target_sim_edges=target_sim_edges,
                eigvals=spectral_eigvals,
                eigvecs=spectral_eigvecs,
            )
            dti_adjacency = build_training_adjacency_matrix(
                num_drug=num_drug,
                num_target=num_target,
                train_pos_edges=train_pos_edges,
                drug_sim_edges=None,
                target_sim_edges=None,
                add_self_loops=False,
            ).detach().cpu().numpy().astype(np.float32)
            adjacency_np = train_adj_matrix.detach().cpu().numpy().astype(np.float32)
            degree = adjacency_np.sum(axis=1) - np.diag(adjacency_np)
            source_node = int(np.argmax(degree))
            node_influence_context = {
                "eigvecs": spectral_eigvecs.detach().cpu().numpy().astype(np.float32),
                "adjacency": dti_adjacency,
                "layout_adjacency": adjacency_np,
                "source_node": source_node,
                "num_drug": num_drug,
            }
    train_adj_matrix = train_adj_matrix.to(device)
    train_edge_index = train_edge_index.to(device)

    train_edges = train_pos_edges + train_neg_edges
    test_edges = test_pos_edges + test_neg_edges
    train_labels = torch.tensor(
        [1] * len(train_pos_edges) + [0] * len(train_neg_edges),
        dtype=torch.float32,
        device=device,
    )
    test_labels = torch.tensor(
        [1] * len(test_pos_edges) + [0] * len(test_neg_edges),
        dtype=torch.float32,
        device=device,
    )

    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
    emb_dim = node_features_tensor.shape[1]
    model = build_model(
        model_name=model_name,
        num_nodes=num_drug + num_target,
        node_features=node_features_tensor,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_scales=num_scales,
        num_layers=num_layers,
        node_feature_mode="random",
        use_eigen_encoding=use_eigen_encoding,
        use_local_mpnn=use_local_mpnn,
        use_trainable_graph_branch=use_trainable_graph_branch,
        graph_branch_layers=graph_branch_layers,
        graph_branch_dropout=graph_branch_dropout,
        disable_post_filter_mlp=disable_post_filter_mlp,
    ).to(device)
    if use_spectral_context and hasattr(model, "set_spectral_context"):
        model.set_spectral_context(
            eigvals=spectral_eigvals.to(device) if spectral_eigvals is not None else None,
            eigvecs=spectral_eigvecs.to(device) if spectral_eigvecs is not None else None,
        )
    if use_trainable_graph_branch and hasattr(model, "set_entity_graphs"):
        if graph_branch_drug_graphs is None or graph_branch_target_graphs is None:
            raise ValueError("graph_branch_drug_graphs and graph_branch_target_graphs are required.")
        model.set_entity_graphs(
            num_drug=num_drug,
            num_target=num_target,
            drug_graphs=graph_branch_drug_graphs,
            target_graphs=graph_branch_target_graphs,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    fold_output_dir = visualization_output_dir
    checkpoint_paths = {}
    model_config = {
        "model_name": model_name,
        "num_nodes": num_drug + num_target,
        "num_drug": num_drug,
        "num_target": num_target,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "num_scales": num_scales,
        "num_layers": num_layers,
        "use_spectral_context": use_spectral_context,
        "spectral_topk": spectral_topk,
        "use_eigen_encoding": use_eigen_encoding,
        "use_local_mpnn": use_local_mpnn,
        "use_trainable_graph_branch": use_trainable_graph_branch,
        "graph_branch_layers": graph_branch_layers,
        "graph_branch_dropout": graph_branch_dropout,
        "disable_post_filter_mlp": disable_post_filter_mlp,
    }

    if selection_metric not in {"auc", "aupr"}:
        raise ValueError("selection_metric must be one of: 'auc', 'aupr'")

    best_metric_value = -1
    best_epoch = -1
    best_state_dict = None
    spectral_trace_records = []

    for epoch in range(1, epochs + 1):
        model.train()
        if batch_size is None or batch_size >= len(train_edges):
            optimizer.zero_grad()
            logits = model(train_edge_index, train_edges, num_drug, device)
            loss = criterion(logits, train_labels)
            if hasattr(model, "diversity_regularization") and waveas_diversity_lambda > 0:
                loss = loss + waveas_diversity_lambda * model.diversity_regularization()
            if hasattr(model, "wavelet_energy_regularization") and waveas_wavelet_energy_lambda > 0:
                loss = loss + waveas_wavelet_energy_lambda * model.wavelet_energy_regularization()
            if hasattr(model, "wavelet_scaling_balance_regularization") and waveas_wavelet_balance_lambda > 0:
                loss = loss + waveas_wavelet_balance_lambda * model.wavelet_scaling_balance_regularization(
                    target_ratio=waveas_wavelet_target_ratio
                )
            loss.backward()
            optimizer.step()
        else:
            epoch_loss = 0.0
            num_seen = 0
            for batch_edges, batch_labels in iterate_edge_batches(
                train_edges,
                train_labels,
                batch_size=batch_size,
                shuffle=shuffle,
            ):
                optimizer.zero_grad()
                logits = model(train_edge_index, batch_edges, num_drug, device)
                batch_loss = criterion(logits, batch_labels)
                if hasattr(model, "diversity_regularization") and waveas_diversity_lambda > 0:
                    batch_loss = batch_loss + waveas_diversity_lambda * model.diversity_regularization()
                if hasattr(model, "wavelet_energy_regularization") and waveas_wavelet_energy_lambda > 0:
                    batch_loss = batch_loss + waveas_wavelet_energy_lambda * model.wavelet_energy_regularization()
                if hasattr(model, "wavelet_scaling_balance_regularization") and waveas_wavelet_balance_lambda > 0:
                    batch_loss = batch_loss + waveas_wavelet_balance_lambda * model.wavelet_scaling_balance_regularization(
                        target_ratio=waveas_wavelet_target_ratio
                    )
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item() * len(batch_edges)
                num_seen += len(batch_edges)
            loss = torch.tensor(epoch_loss / num_seen, dtype=torch.float32)

        eval_result = evaluate_model(
            model=model,
            edge_index=train_edge_index,
            eval_edges=test_edges,
            eval_labels=test_labels,
            num_drug=num_drug,
            device=device,
            threshold=threshold,
        )

        current_metric = eval_result[selection_metric]
        if np.isnan(current_metric):
            current_metric = -1
        if current_metric > best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (
            visualize_waveas_spectral
            and model_name == "WaveAS"
            and hasattr(model, "get_visualization_state")
            and (epoch == 1 or epoch % max(1, visualize_every) == 0 or epoch == epochs)
        ):
            vis_state = model.get_visualization_state()
            vis_state["epoch"] = epoch
            spectral_trace_records.append(vis_state)

        if verbose and (epoch == 1 or epoch % 1 == 0 or epoch == epochs):
            print(
                f"Epoch [{epoch:03d}/{epochs}] "
                f"Loss={loss.item():.4f} "
                f"AUC={eval_result['auc']:.4f} "
                f"AUPR={eval_result['aupr']:.4f} "
                f"F1={eval_result['f1']:.4f} "
                f"MCC={eval_result['mcc']:.4f}"
            )

    last_eval = evaluate_model(
        model=model,
        edge_index=train_edge_index,
        eval_edges=test_edges,
        eval_labels=test_labels,
        num_drug=num_drug,
        device=device,
        threshold=threshold,
    )

    if fold_output_dir:
        last_metrics = {"epoch": int(epochs)}
        last_metrics.update(last_eval)
        checkpoint_paths["last_model"] = save_model_checkpoint(
            path=os.path.join(fold_output_dir, "last_model.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epochs,
            metrics=last_metrics,
            model_config=model_config,
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_eval = evaluate_model(
        model=model,
        edge_index=train_edge_index,
        eval_edges=test_edges,
        eval_labels=test_labels,
        num_drug=num_drug,
        device=device,
        threshold=threshold,
    )

    if fold_output_dir:
        best_metrics = {
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_metric_value),
            "selection_metric": selection_metric,
        }
        best_metrics.update(final_eval)
        checkpoint_paths["best_model"] = save_model_checkpoint(
            path=os.path.join(fold_output_dir, "best_model.pt"),
            model=model,
            optimizer=optimizer,
            epoch=best_epoch,
            metrics=best_metrics,
            model_config=model_config,
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    result = {
        "best_epoch": best_epoch,
        "best_selection_score": best_metric_value,
        "selection_metric": selection_metric,
        "model_name": model_name,
        "num_train_pos": len(train_pos_edges),
        "num_train_neg": len(train_neg_edges),
        "num_test_pos": len(test_pos_edges),
        "num_test_neg": len(test_neg_edges),
    }
    if fold_output_dir:
        result["checkpoint_paths"] = checkpoint_paths
    if visualize_waveas_spectral and model_name == "WaveAS" and visualization_output_dir:
        result["visualization_paths"] = save_waveas_analysis_figures(
            spectral_trace_records,
            visualization_output_dir,
            node_influence_context=node_influence_context,
            case_selection_context={
                "eval_edges": list(test_edges),
                "y_true": np.asarray(final_eval["y_true"]).astype(np.int64),
                "y_pred": np.asarray(final_eval["y_pred"]).astype(np.int64),
                "y_score": np.asarray(final_eval["y_score"]).astype(np.float32),
                "num_drug": int(num_drug),
                "num_target": int(num_target),
                "threshold": float(threshold),
            },
            influence_layout=influence_layout,
            influence_layout_k=influence_layout_k,
            influence_layout_iterations=influence_layout_iterations,
            influence_layout_seed=influence_layout_seed,
        )
    result.update(final_eval)
    return result


def run_cv_with_full_graph(
    file_path,
    node_feature_path,
    dataset,
    model_name,
    device="cuda",
    seed=42,
    hidden_dim=128,
    dropout=0.2,
    num_scales=4,
    num_layers=2,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=None,
    shuffle=True,
    n_splits=10,
    threshold=0.5,
    output_dir="outputs",
    selection_metric="aupr",
    gmjrl_neg_topk=10,
    include_drug_sim=True,
    include_target_sim=True,
    drug_sim_topk=10,
    drug_sim_threshold=0.3,
    target_sim_topk=10,
    target_sim_threshold=0.3,
    add_self_loops=True,
    use_spectral_context=False,
    spectral_topk=16,
    waveas_diversity_lambda=1e-2,
    waveas_wavelet_energy_lambda=1e-2,
    waveas_wavelet_balance_lambda=1e-1,
    waveas_wavelet_target_ratio=0.25,
    use_eigen_encoding=True,
    use_local_mpnn=False,
    use_trainable_graph_branch=False,
    node_init_strategy="precomputed",
    graph_branch_layers=2,
    graph_branch_dropout=0.1,
    disable_post_filter_mlp=False,
    graph_branch_target_mode="contact",
    graph_branch_target_topk=8,
    graph_branch_target_max_length=512,
    graph_branch_target_contact_threshold=0.5,
    visualize_waveas_spectral=False,
    visualize_every=5,
    influence_layout="spring",
    influence_layout_k=0.12,
    influence_layout_iterations=200,
    influence_layout_seed=42,
):
    set_seed(seed)
    pos_edges, drug_id_map, target_id_map, num_drug, num_target, _ = load_positive_edges_flexible(file_path)
    feature_bundle = load_precomputed_node_feature_bundle(node_feature_path)
    if "drug_features" not in feature_bundle or "target_features" not in feature_bundle:
        raise ValueError(f"drug_features/target_features keys not found in {node_feature_path}")
    if "drug_ids" not in feature_bundle or "target_ids" not in feature_bundle:
        raise ValueError(f"drug_ids/target_ids keys not found in {node_feature_path}")
    drug_features = feature_bundle["drug_features"].astype(np.float32)
    target_features = feature_bundle["target_features"].astype(np.float32)
    drug_ids = feature_bundle["drug_ids"]
    target_ids = feature_bundle["target_ids"]
    aligned_drug, aligned_target, _precomputed_node_features = align_precomputed_features_to_dti(
        drug_id_map=drug_id_map,
        target_id_map=target_id_map,
        drug_features=drug_features,
        target_features=target_features,
        drug_ids=drug_ids,
        target_ids=target_ids,
    )
    aligned_drug_for_model, aligned_target_for_model, base_node_features = build_node_features_by_strategy(
        strategy=node_init_strategy,
        aligned_drug=aligned_drug,
        aligned_target=aligned_target,
        seed=seed,
    )

    expected_num_nodes = num_drug + num_target
    if base_node_features.shape[0] != expected_num_nodes:
        raise ValueError(
            f"node_features rows ({base_node_features.shape[0]}) != num_drug+num_target ({expected_num_nodes})."
        )

    drug_sim_edges = []
    target_sim_edges = []
    if include_drug_sim:
        drug_sim_edges = build_similarity_edges_from_features(
            aligned_drug_for_model, topk=drug_sim_topk, threshold=drug_sim_threshold
        )
    if include_target_sim:
        target_sim_edges = build_similarity_edges_from_features(
            aligned_target_for_model, topk=target_sim_topk, threshold=target_sim_threshold
        )

    online_drug_graphs = None
    online_target_graphs = None
    online_drug_path = ""
    online_target_path = ""
    if use_trainable_graph_branch:
        target_contact_lookup = build_target_contact_lookup_from_bundle(feature_bundle)
        if graph_branch_target_mode == "contact" and target_contact_lookup is None:
            raise ValueError(
                "graph_branch_target_mode=contact requires target_contact_maps in node_feature npz. "
                "Please regenerate node features with node_representation_llm.py so contact maps are saved."
            )
        online_drug_graphs, online_target_graphs, online_drug_path, online_target_path = build_entity_graphs_for_training(
            file_path=file_path,
            dataset=dataset,
            drug_id_map=drug_id_map,
            target_id_map=target_id_map,
            target_mode=graph_branch_target_mode,
            target_topk=graph_branch_target_topk,
            target_max_length=graph_branch_target_max_length,
            target_contact_lookup=target_contact_lookup,
            target_contact_threshold=graph_branch_target_contact_threshold,
        )

    drug_dissimmat = build_drug_dissimilarity_neighbors_from_features(aligned_drug_for_model, topk=gmjrl_neg_topk)
    folds = make_kfold_edge_splits_gmjrl_style(
        pos_edges=pos_edges,
        num_drug=num_drug,
        num_target=num_target,
        drug_dissimmat=drug_dissimmat,
        n_splits=n_splits,
        seed=seed,
    )

    print("=" * 60)
    print("Dataset Info")
    print(f"Positive edges       : {len(pos_edges)}")
    print(f"Number of drugs      : {num_drug}")
    print(f"Number of targets    : {num_target}")
    print(f"Total nodes          : {expected_num_nodes}")
    print(f"Node feature shape   : {base_node_features.shape}")
    print(f"Node init strategy   : {node_init_strategy}")
    print(f"Drug sim edges       : {len(drug_sim_edges)}")
    print(f"Target sim edges     : {len(target_sim_edges)}")
    print("Negative sampling    : GMJRL style")
    print(f"GMJRL neg topk       : {gmjrl_neg_topk}")
    print(f"Model name           : {model_name}")
    print(f"Selection metric     : {selection_metric}")
    if use_trainable_graph_branch:
        print("Online entity graph  : enabled")
        print(f"Drug entity file     : {online_drug_path}")
        print(f"Target entity file   : {online_target_path}")
        print(f"Target graph mode    : {graph_branch_target_mode}")
        if graph_branch_target_mode == "contact":
            print(f"Target contact thr   : {graph_branch_target_contact_threshold:.2f}")
    print("=" * 60)

    fold_results = []
    for fold_idx, (train_pos_edges, test_pos_edges, train_neg_edges, test_neg_edges) in enumerate(folds, start=1):
        print(f"\n{'=' * 25} Fold {fold_idx}/{n_splits} {'=' * 25}")
        result = train_one_fold(
            train_pos_edges=train_pos_edges,
            test_pos_edges=test_pos_edges,
            train_neg_edges=train_neg_edges,
            test_neg_edges=test_neg_edges,
            num_drug=num_drug,
            num_target=num_target,
            node_features=base_node_features,
            drug_sim_edges=drug_sim_edges,
            target_sim_edges=target_sim_edges,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_scales=num_scales,
            num_layers=num_layers,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=True,
            threshold=threshold,
            model_name=model_name,
            selection_metric=selection_metric,
            add_self_loops=add_self_loops,
            use_spectral_context=use_spectral_context,
            spectral_topk=spectral_topk,
            waveas_diversity_lambda=waveas_diversity_lambda,
            waveas_wavelet_energy_lambda=waveas_wavelet_energy_lambda,
            waveas_wavelet_balance_lambda=waveas_wavelet_balance_lambda,
            waveas_wavelet_target_ratio=waveas_wavelet_target_ratio,
            use_eigen_encoding=use_eigen_encoding,
            use_local_mpnn=use_local_mpnn,
            use_trainable_graph_branch=use_trainable_graph_branch,
            graph_branch_layers=graph_branch_layers,
            graph_branch_dropout=graph_branch_dropout,
            disable_post_filter_mlp=disable_post_filter_mlp,
            graph_branch_drug_graphs=online_drug_graphs,
            graph_branch_target_graphs=online_target_graphs,
            visualization_output_dir=os.path.join(output_dir, f"fold_{fold_idx}"),
            visualize_waveas_spectral=visualize_waveas_spectral,
            visualize_every=visualize_every,
            influence_layout=influence_layout,
            influence_layout_k=influence_layout_k,
            influence_layout_iterations=influence_layout_iterations,
            influence_layout_seed=influence_layout_seed,
        )
        fold_results.append(result)
        print(
            f"Fold {fold_idx} Result | "
            f"Best Epoch={result['best_epoch']} | "
            f"AUC={result['auc']:.4f} | "
            f"AUPR={result['aupr']:.4f} | "
            f"TrainPos/Neg={result['num_train_pos']}/{result['num_train_neg']} | "
            f"TestPos/Neg={result['num_test_pos']}/{result['num_test_neg']}"
        )

    metric_names = ["acc", "f1", "precision", "sen", "mcc", "auc", "aupr", "recall"]
    metric_summary = {metric: [result[metric] for result in fold_results] for metric in metric_names}
    roc_path, pr_path = plot_cv_curves(fold_results=fold_results, output_dir=output_dir)
    fold_csv_path = save_fold_results_csv(fold_results=fold_results, output_dir=output_dir, model_name=model_name)
    summary_csv_path = save_metric_summary_csv(
        metric_summary=metric_summary,
        fold_results=fold_results,
        output_dir=output_dir,
        model_name=model_name,
    )

    print("-" * 60)
    for metric in metric_names:
        values = metric_summary[metric]
        print(f"Mean {metric.upper():<9}: {np.nanmean(values):.4f} +/- {np.nanstd(values):.4f}")
    print(f"ROC curve saved to      : {roc_path}")
    print(f"PR curve saved to       : {pr_path}")
    print(f"Fold results saved to   : {fold_csv_path}")
    print(f"Metric summary saved to : {summary_csv_path}")

    return fold_results


def compare_models_with_full_graph(model_names, **kwargs):
    comparison_summary = {}
    output_dir = kwargs["output_dir"]
    base_kwargs = dict(kwargs)
    base_kwargs.pop("output_dir", None)

    for model_name in model_names:
        model_output_dir = os.path.join(output_dir, model_name)
        print("\n" + "#" * 60)
        print(f"Running model: {model_name}")
        print("#" * 60)
        fold_results = run_cv_with_full_graph(
            model_name=model_name,
            output_dir=model_output_dir,
            **base_kwargs,
        )
        comparison_summary[model_name] = {
            "mean_auc": float(np.nanmean([result["auc"] for result in fold_results])),
            "mean_aupr": float(np.nanmean([result["aupr"] for result in fold_results])),
            "std_auc": float(np.nanstd([result["auc"] for result in fold_results])),
            "std_aupr": float(np.nanstd([result["aupr"] for result in fold_results])),
        }

    comparison_csv_path = save_model_comparison_csv(comparison_summary=comparison_summary, output_dir=output_dir)
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    for model_name in model_names:
        summary = comparison_summary[model_name]
        print(
            f"{model_name:<8} | "
            f"Mean AUC={summary['mean_auc']:.4f} +/- {summary['std_auc']:.4f} | "
            f"Mean AUPR={summary['mean_aupr']:.4f} +/- {summary['std_aupr']:.4f}"
        )
    print(f"Comparison summary saved to: {comparison_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end training with full n*n adjacency matrix + node feature matrix. "
            "Drug-drug and target-target edges are built from feature similarity."
        )
    )
    parser.add_argument("--file_path", type=str, required=True, help="Path to DTI file (.csv/.xlsx/.xls).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset name for resolving node feature npz when --node_feature_path is omitted.",
    )
    parser.add_argument(
        "--node_feature_dir",
        type=str,
        default="Data/node_features",
        help="Directory that stores precomputed node features (.npz).",
    )
    parser.add_argument(
        "--node_feature_path",
        type=str,
        default="",
        help="Optional explicit path to node feature npz.",
    )
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_scales", type=int, default=4, help="Number of spectral branches: 1 scaling + (num_scales-1) wavelets.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked WaveAS spectral blocks.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--shuffle", type=str2bool, default=True)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="WaveAS", choices=["WaveAS"])
    parser.add_argument("--selection_metric", type=str, default="aupr", choices=["auc", "aupr"])
    parser.add_argument(
        "--gmjrl_neg_topk",
        type=int,
        default=1,
        help="Top-k dissimilar drugs used in GMJRL-style negative sampling.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="outputs/full_training")

    parser.add_argument("--include_drug_sim", type=str2bool, default=True)
    parser.add_argument("--include_target_sim", type=str2bool, default=True)
    parser.add_argument(
        "--node_init_strategy",
        type=str,
        default="precomputed",
        choices=["precomputed", "random", "zero"],
        help="How to initialize node features before model training.",
    )
    parser.add_argument("--drug_sim_topk", type=int, default=10)
    parser.add_argument("--drug_sim_threshold", type=float, default=0.3)
    parser.add_argument("--target_sim_topk", type=int, default=10)
    parser.add_argument("--target_sim_threshold", type=float, default=0.3)
    parser.add_argument("--add_self_loops", type=str2bool, default=True)
    parser.add_argument(
        "--use_spectral_context",
        type=str2bool,
        default=True,
        help="Use truncated eigenvalues/eigenvectors from each training graph to condition the WaveAS kernel generator.",
    )
    parser.add_argument(
        "--spectral_topk",
        type=int,
        default=16,
        help="Number of smallest Laplacian eigenpairs used as spectral context when enabled.",
    )
    parser.add_argument(
        "--waveas_diversity_lambda",
        type=float,
        default=1e-2,
        help="Regularization weight that discourages different WaveAS wavelet branches from collapsing to the same order distribution.",
    )
    parser.add_argument(
        "--waveas_wavelet_energy_lambda",
        type=float,
        default=1e-2,
        help="Regularization weight that encourages WaveAS spectral wavelet branches to keep non-trivial response magnitude.",
    )
    parser.add_argument(
        "--waveas_wavelet_balance_lambda",
        type=float,
        default=1e-1,
        help="Regularization weight that prevents WaveAS spectral wavelet filters from collapsing to a tiny fraction of the scaling filter.",
    )
    parser.add_argument(
        "--waveas_wavelet_target_ratio",
        type=float,
        default=0.25,
        help="Minimum target ratio between average wavelet response magnitude and average scaling response magnitude.",
    )
    parser.add_argument(
        "--visualize_waveas_spectral",
        type=str2bool,
        default=False,
        help="Record and save spectral filter visualizations during training when model_name=WaveAS.",
    )
    parser.add_argument(
        "--visualize_every",
        type=int,
        default=5,
        help="Save one WaveAS spectral visualization snapshot every N epochs.",
    )
    parser.add_argument(
        "--use_eigen_encoding",
        type=str2bool,
        default=True,
        help="Use the WaveAS-style spectral eigenvalue encoder before decoding scaling/wavelet coefficients and scales.",
    )
    parser.add_argument(
        "--use_local_mpnn",
        type=str2bool,
        default=False,
        help="Enable a local GCN message-passing branch inside each WaveAS spectral block, following the original local+global fusion pattern.",
    )
    parser.add_argument(
        "--disable_post_filter_mlp",
        type=str2bool,
        default=False,
        help="Ablation: remove the post-filter W/MLP transform and keep only spectral filtering.",
    )
    parser.add_argument(
        "--use_trainable_graph_branch",
        type=str2bool,
        default=False,
        help="Enable an online trainable per-entity graph encoder branch (each drug/target has its own graph).",
    )
    parser.add_argument(
        "--graph_branch_target_mode",
        type=str,
        default="contact",
        choices=["contact", "approx", "seq"],
        help="Target entity-graph edge mode in online branch.",
    )
    parser.add_argument(
        "--graph_branch_target_contact_threshold",
        type=float,
        default=0.5,
        help="Contact-map threshold for target entity-graph edges when mode=contact.",
    )
    parser.add_argument(
        "--graph_branch_target_topk",
        type=int,
        default=8,
        help="Top-k approximate residue neighbors used when target graph mode is approx/contact fallback.",
    )
    parser.add_argument(
        "--graph_branch_target_max_length",
        type=int,
        default=512,
        help="Maximum target residue length used for online target entity-graph construction.",
    )
    parser.add_argument(
        "--graph_branch_layers",
        type=int,
        default=2,
        help="Number of GCN layers in each entity graph branch encoder.",
    )
    parser.add_argument(
        "--graph_branch_dropout",
        type=float,
        default=0.1,
        help="Dropout inside trainable entity graph branch encoders.",
    )
    parser.add_argument(
        "--influence_layout",
        type=str,
        default="spring",
        choices=["spring", "kamada"],
        help="Layout used for node influence maps.",
    )
    parser.add_argument(
        "--influence_layout_k",
        type=float,
        default=0.12,
        help="Spring-layout ideal node spacing for node influence maps.",
    )
    parser.add_argument(
        "--influence_layout_iterations",
        type=int,
        default=200,
        help="Number of spring-layout iterations for node influence maps.",
    )
    parser.add_argument(
        "--influence_layout_seed",
        type=int,
        default=42,
        help="Random seed for node influence map layout.",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = None if args.batch_size is not None and args.batch_size <= 0 else args.batch_size
    node_feature_path, resolved_dataset = resolve_node_feature_path(
        node_feature_path=args.node_feature_path,
        dataset=args.dataset,
        file_path=args.file_path,
        node_feature_dir=args.node_feature_dir,
    )

    print("Using device       :", device)
    print("Batch size         :", batch_size)
    print("Shuffle            :", args.shuffle)
    print("Resolved dataset   :", resolved_dataset)
    print("Node feature path  :", node_feature_path)

    common_kwargs = dict(
        file_path=args.file_path,
        node_feature_path=node_feature_path,
        dataset=resolved_dataset,
        device=device,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_scales=args.num_scales,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=batch_size,
        shuffle=args.shuffle,
        n_splits=args.n_splits,
        threshold=args.threshold,
        output_dir=args.output_dir,
        selection_metric=args.selection_metric,
        gmjrl_neg_topk=args.gmjrl_neg_topk,
        include_drug_sim=args.include_drug_sim,
        include_target_sim=args.include_target_sim,
        node_init_strategy=args.node_init_strategy,
        drug_sim_topk=args.drug_sim_topk,
        drug_sim_threshold=args.drug_sim_threshold,
        target_sim_topk=args.target_sim_topk,
        target_sim_threshold=args.target_sim_threshold,
        add_self_loops=args.add_self_loops,
        use_spectral_context=args.use_spectral_context,
        spectral_topk=args.spectral_topk,
        waveas_diversity_lambda=args.waveas_diversity_lambda,
        waveas_wavelet_energy_lambda=args.waveas_wavelet_energy_lambda,
        waveas_wavelet_balance_lambda=args.waveas_wavelet_balance_lambda,
        waveas_wavelet_target_ratio=args.waveas_wavelet_target_ratio,
        use_eigen_encoding=args.use_eigen_encoding,
        use_local_mpnn=args.use_local_mpnn,
        disable_post_filter_mlp=args.disable_post_filter_mlp,
        use_trainable_graph_branch=args.use_trainable_graph_branch,
        graph_branch_layers=args.graph_branch_layers,
        graph_branch_dropout=args.graph_branch_dropout,
        graph_branch_target_mode=args.graph_branch_target_mode,
        graph_branch_target_topk=args.graph_branch_target_topk,
        graph_branch_target_max_length=args.graph_branch_target_max_length,
        graph_branch_target_contact_threshold=args.graph_branch_target_contact_threshold,
        visualize_waveas_spectral=args.visualize_waveas_spectral,
        visualize_every=args.visualize_every,
    )

    selected_models = [args.model_name]
    run_config_path = save_run_config(
        output_dir=args.output_dir,
        cli_args=vars(args),
        runtime_context={
            "resolved_dataset": resolved_dataset,
            "resolved_node_feature_path": node_feature_path,
            "resolved_device": device,
            "resolved_batch_size": batch_size,
            "selected_models": selected_models,
            "cwd": os.getcwd(),
        },
    )
    print("Run config path    :", run_config_path)

    run_cv_with_full_graph(model_name=args.model_name, **common_kwargs)
