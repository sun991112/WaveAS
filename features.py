import argparse
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_SET = set(AMINO_ACIDS)
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
DIPEPTIDE_INDEX = {dp: idx for idx, dp in enumerate(DIPEPTIDES)}
ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se", "other"
]
ATOM_SYMBOL_INDEX = {symbol: idx for idx, symbol in enumerate(ATOM_SYMBOLS)}
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def read_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Only support .csv / .xlsx / .xls, got: {path}")


def infer_column(df, candidates, file_path):
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"None of columns {candidates} found in {file_path}. "
        f"Available columns: {list(df.columns)}"
    )


def first_non_empty_string(series):
    for value in series:
        if pd.notna(value):
            text = str(value).strip()
            if text:
                return text
    return ""


def resolve_feature_paths(drug_path="", target_path="", dataset="", data_dir="Data", dti_lists_dir="Data/dti_lists"):
    if drug_path and target_path:
        return drug_path, target_path, os.path.dirname(os.path.abspath(drug_path))
    if drug_path or target_path:
        raise ValueError("Please provide both --drug_path and --target_path together.")
    if not dataset:
        raise ValueError("Provide either both --drug_path/--target_path or --dataset.")

    candidates = [
        os.path.join(data_dir, dataset),
        os.path.join(dti_lists_dir, dataset),
    ]
    dataset_dir = None
    for candidate in candidates:
        if os.path.isdir(candidate):
            dataset_dir = candidate
            break
    if dataset_dir is None:
        raise FileNotFoundError(
            f"Cannot find dataset directory for '{dataset}'. Tried: {candidates}"
        )

    drug_candidates = [
        os.path.join(dataset_dir, "drugs.xlsx"),
        os.path.join(dataset_dir, "drugs.xls"),
        os.path.join(dataset_dir, "drug.csv"),
    ]
    target_candidates = [
        os.path.join(dataset_dir, "targets.xlsx"),
        os.path.join(dataset_dir, "targets.xls"),
        os.path.join(dataset_dir, "target.csv"),
    ]
    resolved_drug = next((p for p in drug_candidates if os.path.exists(p)), None)
    resolved_target = next((p for p in target_candidates if os.path.exists(p)), None)
    if resolved_drug is None:
        raise FileNotFoundError(f"Cannot find drugs file under {dataset_dir}. Tried: {drug_candidates}")
    if resolved_target is None:
        raise FileNotFoundError(f"Cannot find targets file under {dataset_dir}. Tried: {target_candidates}")
    return resolved_drug, resolved_target, dataset_dir


def make_non_overwriting_path(path):
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{root}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


def build_default_output_path(
    dataset,
    base_dir,
    use_drug_graph_branch,
    use_target_graph_branch,
    use_entity_graph_ae,
    use_static_drug,
    use_static_target,
):
    tags = ["llm"]
    if use_drug_graph_branch or use_target_graph_branch:
        graph_tags = []
        if use_drug_graph_branch:
            graph_tags.append("druggraph")
        if use_target_graph_branch:
            graph_tags.append("targetgraph")
        tags.append("-".join(graph_tags))
    if use_static_drug or use_static_target:
        static_tags = []
        if use_static_drug:
            static_tags.append("drugstatic")
        if use_static_target:
            static_tags.append("targetstatic")
        tags.append("-".join(static_tags))
    if use_entity_graph_ae:
        tags.append("entitygraphae")
    file_name = f"node_features_{dataset}_{'_'.join(tags)}.npz"
    return make_non_overwriting_path(os.path.join(base_dir, file_name))


def build_default_llm_cache_path(dataset, base_dir):
    file_name = f"llm_cache_{dataset}.npz"
    return make_non_overwriting_path(os.path.join(base_dir, file_name))


def load_precomputed_llm_cache(cache_path, drug_ids=None, target_ids=None):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"LLM cache file not found: {cache_path}")
    bundle = np.load(cache_path, allow_pickle=True)
    required_keys = ["drug_ids", "target_ids", "target_residue_embeddings", "target_contact_maps"]
    missing = [key for key in required_keys if key not in bundle]
    if missing:
        raise ValueError(f"LLM cache missing required keys {missing}: {cache_path}")

    cached_drug_ids = bundle["drug_ids"]
    cached_target_ids = bundle["target_ids"]
    if drug_ids is not None and not np.array_equal(np.asarray(drug_ids).astype(str), np.asarray(cached_drug_ids).astype(str)):
        raise ValueError("Drug ids in llm cache do not match current dataset ordering.")
    if target_ids is not None and not np.array_equal(np.asarray(target_ids).astype(str), np.asarray(cached_target_ids).astype(str)):
        raise ValueError("Target ids in llm cache do not match current dataset ordering.")

    if "drug_semantic" in bundle:
        drug_semantic = bundle["drug_semantic"]
    elif "drug_llm" in bundle:
        drug_semantic = bundle["drug_llm"]
    else:
        raise ValueError(f"LLM cache missing drug_semantic/drug_llm: {cache_path}")
    target_semantic = (
        bundle["target_semantic"]
        if "target_semantic" in bundle
        else bundle["target_llm"] if "target_llm" in bundle else np.zeros((len(cached_target_ids), 0), dtype=np.float32)
    )
    return {
        "drug_semantic": np.asarray(drug_semantic, dtype=np.float32),
        "target_semantic": np.asarray(target_semantic, dtype=np.float32),
        "target_residue_embeddings": list(bundle["target_residue_embeddings"]),
        "target_contact_maps": list(bundle["target_contact_maps"]),
        "drug_ids": cached_drug_ids,
        "target_ids": cached_target_ids,
    }


def save_precomputed_llm_cache(
    cache_path,
    drug_ids,
    target_ids,
    drug_semantic,
    target_semantic,
    target_residue_embeddings,
    target_contact_maps,
):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez_compressed(
        cache_path,
        drug_ids=np.asarray(drug_ids),
        target_ids=np.asarray(target_ids),
        drug_semantic=np.asarray(drug_semantic, dtype=np.float32),
        target_semantic=np.asarray(target_semantic, dtype=np.float32),
        target_residue_embeddings=np.asarray(target_residue_embeddings, dtype=object),
        target_contact_maps=np.asarray(target_contact_maps, dtype=object),
    )


def _group_first_non_empty(df, id_col, value_col, output_id_name, output_value_name):
    grouped = (
        df[[id_col, value_col]]
        .rename(columns={id_col: output_id_name, value_col: output_value_name})
        .groupby(output_id_name, as_index=False)[output_value_name]
        .agg(first_non_empty_string)
    )
    return grouped.reset_index(drop=True)


def load_drug_table(drug_path):
    df = read_table(drug_path)
    smiles_col = infer_column(df, ["SMILES", "smiles"], drug_path)
    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    id_col = None
    for candidate in ["drug_id", "drugid"]:
        if candidate in normalized_cols:
            id_col = normalized_cols[candidate]
            break

    smiles_series = df[smiles_col].fillna("").astype(str)
    if id_col is None:
        output = pd.DataFrame({
            "drug_id": np.arange(len(df), dtype=np.int64),
            "SMILES": smiles_series,
        })
    else:
        ids = df[id_col].fillna("").astype(str).str.strip()
        tmp = pd.DataFrame({"drug_id": ids, "SMILES": smiles_series})
        output = _group_first_non_empty(tmp, "drug_id", "SMILES", "drug_id", "SMILES")
    output["SMILES"] = output["SMILES"].fillna("").astype(str)
    return output


def load_target_table(target_path):
    df = read_table(target_path)
    fasta_col = infer_column(
        df,
        ["protein_fastas", "FASTA", "TargetSequence", "target_fasta", "fastas"],
        target_path,
    )
    normalized_cols = {str(col).strip().lower(): col for col in df.columns}
    id_col = None
    for candidate in ["target_id", "targetid", "protein_id"]:
        if candidate in normalized_cols:
            id_col = normalized_cols[candidate]
            break

    fasta_series = df[fasta_col].fillna("").astype(str)
    if id_col is None:
        output = pd.DataFrame({
            "target_id": np.arange(len(df), dtype=np.int64),
            "FASTA": fasta_series,
        })
    else:
        ids = df[id_col].fillna("").astype(str).str.strip()
        tmp = pd.DataFrame({"target_id": ids, "FASTA": fasta_series})
        output = _group_first_non_empty(tmp, "target_id", "FASTA", "target_id", "FASTA")
    output["FASTA"] = output["FASTA"].fillna("").astype(str)
    return output


def bitvect_to_array(fp, length):
    arr = np.zeros((length,), dtype=np.float32)
    if fp is not None:
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_smiles(smiles, morgan_bits=1024, morgan_radius=2):
    if not smiles:
        return np.zeros((morgan_bits + 167 + 8,), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((morgan_bits + 167 + 8,), dtype=np.float32)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    morgan_arr = bitvect_to_array(morgan_fp, morgan_bits)
    maccs_arr = bitvect_to_array(maccs_fp, 167)
    descriptor_values = np.asarray(
        [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.RingCount(mol)),
            float(Descriptors.FractionCSP3(mol)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([morgan_arr, maccs_arr, descriptor_values], axis=0)


def amino_acid_composition(seq):
    seq = seq or ""
    counts = Counter(ch for ch in seq if ch in AMINO_ACID_SET)
    total = sum(counts.values())
    if total == 0:
        return np.zeros((len(AMINO_ACIDS),), dtype=np.float32)
    return np.asarray([counts[aa] / total for aa in AMINO_ACIDS], dtype=np.float32)


def dipeptide_composition(seq):
    seq = "".join(ch for ch in (seq or "") if ch in AMINO_ACID_SET)
    vec = np.zeros((len(DIPEPTIDES),), dtype=np.float32)
    if len(seq) < 2:
        return vec
    total = len(seq) - 1
    for i in range(total):
        dp = seq[i : i + 2]
        idx = DIPEPTIDE_INDEX.get(dp)
        if idx is not None:
            vec[idx] += 1.0
    vec /= float(total)
    return vec


def basic_sequence_physchem(seq):
    seq = "".join(ch for ch in (seq or "") if ch in AMINO_ACID_SET)
    length = float(len(seq))
    if length == 0:
        return np.zeros((8,), dtype=np.float32)
    counts = Counter(seq)
    aromatic = counts["F"] + counts["W"] + counts["Y"] + counts["H"]
    polar = counts["S"] + counts["T"] + counts["N"] + counts["Q"] + counts["C"]
    positive = counts["K"] + counts["R"] + counts["H"]
    negative = counts["D"] + counts["E"]
    aliphatic = counts["A"] + counts["V"] + counts["I"] + counts["L"] + counts["M"]
    gly_pro = counts["G"] + counts["P"]
    return np.asarray(
        [
            length,
            aromatic / length,
            polar / length,
            positive / length,
            negative / length,
            aliphatic / length,
            gly_pro / length,
            counts["C"] / length,
        ],
        dtype=np.float32,
    )


def build_target_tfidf_svd(sequences, svd_dim, seed):
    if svd_dim <= 0:
        return np.zeros((len(sequences), 0), dtype=np.float32)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False)
    tfidf = vectorizer.fit_transform(sequences)
    max_dim = min(svd_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if max_dim <= 0:
        return np.zeros((len(sequences), 0), dtype=np.float32)
    svd = TruncatedSVD(n_components=max_dim, random_state=seed)
    return svd.fit_transform(tfidf).astype(np.float32)


def safe_project(features, output_dim, seed):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")
    if features.shape[1] == 0:
        return np.zeros((features.shape[0], output_dim), dtype=np.float32)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    max_dim = min(output_dim, scaled.shape[0], scaled.shape[1])
    if max_dim <= 0:
        return np.zeros((features.shape[0], output_dim), dtype=np.float32)

    if scaled.shape[1] > output_dim:
        pca = PCA(n_components=max_dim, random_state=seed)
        projected = pca.fit_transform(scaled).astype(np.float32)
    else:
        projected = scaled.astype(np.float32)

    if projected.shape[1] < output_dim:
        pad = np.zeros((projected.shape[0], output_dim - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected, pad], axis=1)
    return projected


def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked_hidden = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return masked_hidden.sum(dim=1) / denom


def masked_token_embeddings(last_hidden_state, attention_mask):
    mask = attention_mask.to(torch.bool)
    token_embeddings = []
    for idx in range(last_hidden_state.size(0)):
        valid = mask[idx].nonzero(as_tuple=False).view(-1)
        if valid.numel() <= 2:
            token_embeddings.append(np.zeros((0, last_hidden_state.size(-1)), dtype=np.float32))
            continue
        # Drop special tokens at both ends to keep residue-level alignment.
        token_slice = last_hidden_state[idx, valid[1:-1], :]
        token_embeddings.append(token_slice.detach().cpu().numpy().astype(np.float32))
    return token_embeddings


def canonicalize_smiles(smiles):
    smiles = (smiles or "").strip()
    if not smiles:
        return "", False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", False
    return Chem.MolToSmiles(mol, canonical=True), True


def clean_protein_sequence(seq, max_length):
    seq = (seq or "").upper()
    seq = "".join(ch for ch in seq if ch in AMINO_ACID_SET)
    if max_length is not None and max_length > 0:
        seq = seq[:max_length]
    return seq


def _load_hf_model(model_name, device):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for node_representation_llm.py. "
            "Please install transformers."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model


def _batched_encode_texts(texts, model_name, batch_size, device):
    tokenizer, model = _load_hf_model(model_name, device=device)
    hidden_size = int(getattr(model.config, "hidden_size", 768))
    embeddings = np.zeros((len(texts), hidden_size), dtype=np.float32)
    valid_indices = [idx for idx, text in enumerate(texts) if text]
    if not valid_indices:
        return embeddings

    total_batches = (len(valid_indices) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_num, start in enumerate(range(0, len(valid_indices), batch_size), start=1):
            batch_indices = valid_indices[start : start + batch_size]
            batch_texts = [texts[idx] for idx in batch_indices]
            print(f"[info] HF encoding batch {batch_num}/{total_batches} ({len(batch_indices)} samples)")
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = masked_mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings[batch_indices] = pooled.detach().cpu().numpy().astype(np.float32)
    return embeddings


def _derive_contact_from_attentions(attentions, attention_mask):
    # attentions: tuple(num_layers) of [B, H, L, L]
    stacked = torch.stack(list(attentions), dim=0)  # [num_layers, B, H, L, L]
    contact = stacked.mean(dim=(0, 2))  # [B, L, L]
    # Keep only valid tokens and normalize to [0, 1] for stable thresholding.
    contact = contact * attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    min_v = contact.amin(dim=(1, 2), keepdim=True)
    max_v = contact.amax(dim=(1, 2), keepdim=True)
    denom = (max_v - min_v).clamp_min(1e-6)
    return (contact - min_v) / denom


def _batched_encode_texts_with_tokens(texts, model_name, batch_size, device, include_contact=False):
    tokenizer, model = _load_hf_model(model_name, device=device)
    hidden_size = int(getattr(model.config, "hidden_size", 768))
    pooled_embeddings = np.zeros((len(texts), hidden_size), dtype=np.float32)
    token_embeddings = [np.zeros((0, hidden_size), dtype=np.float32) for _ in texts]
    contact_maps = [np.zeros((0, 0), dtype=np.float32) for _ in texts]
    valid_indices = [idx for idx, text in enumerate(texts) if text]
    if not valid_indices:
        return pooled_embeddings, token_embeddings

    total_batches = (len(valid_indices) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_num, start in enumerate(range(0, len(valid_indices), batch_size), start=1):
            batch_indices = valid_indices[start : start + batch_size]
            batch_texts = [texts[idx] for idx in batch_indices]
            print(
                f"[info] HF token encoding batch {batch_num}/{total_batches} "
                f"({len(batch_indices)} samples, include_contact={str(bool(include_contact)).lower()})"
            )
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded, output_attentions=include_contact)
            pooled = masked_mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled_embeddings[batch_indices] = pooled.detach().cpu().numpy().astype(np.float32)
            batch_tokens = masked_token_embeddings(outputs.last_hidden_state, encoded["attention_mask"])
            for local_idx, global_idx in enumerate(batch_indices):
                token_embeddings[global_idx] = batch_tokens[local_idx]
            if include_contact:
                if hasattr(outputs, "contacts") and outputs.contacts is not None:
                    raw_contacts = outputs.contacts
                elif getattr(outputs, "attentions", None) is not None:
                    raw_contacts = _derive_contact_from_attentions(outputs.attentions, encoded["attention_mask"])
                else:
                    raw_contacts = None
                for local_idx, global_idx in enumerate(batch_indices):
                    if raw_contacts is None:
                        contact_maps[global_idx] = np.zeros((0, 0), dtype=np.float32)
                        continue
                    valid = encoded["attention_mask"][local_idx].nonzero(as_tuple=False).view(-1)
                    if valid.numel() <= 2:
                        contact_maps[global_idx] = np.zeros((0, 0), dtype=np.float32)
                        continue
                    sliced = raw_contacts[local_idx][valid[1:-1]][:, valid[1:-1]]
                    contact_maps[global_idx] = sliced.detach().cpu().numpy().astype(np.float32)
    return pooled_embeddings, token_embeddings, contact_maps


def encode_drug_smiles(smiles_list, encoder_name, batch_size, device):
    canonical_smiles = []
    invalid_count = 0
    for idx, smiles in enumerate(smiles_list):
        canonical, valid = canonicalize_smiles(smiles)
        if not valid:
            invalid_count += 1
            warnings.warn(f"Invalid or empty SMILES at index {idx}; using zero vector.", RuntimeWarning)
        canonical_smiles.append(canonical)
    if invalid_count > 0:
        print(f"[warning] Drug encoder skipped {invalid_count} invalid/empty SMILES.")
    return _batched_encode_texts(canonical_smiles, model_name=encoder_name, batch_size=batch_size, device=device)


def encode_target_sequences(seq_list, encoder_name, batch_size, device, max_protein_length):
    cleaned_sequences = []
    invalid_count = 0
    for idx, seq in enumerate(seq_list):
        cleaned = clean_protein_sequence(seq, max_length=max_protein_length)
        if not cleaned:
            invalid_count += 1
            warnings.warn(f"Invalid or empty protein sequence at index {idx}; using zero vector.", RuntimeWarning)
        cleaned_sequences.append(cleaned)
    if invalid_count > 0:
        print(f"[warning] Target encoder skipped {invalid_count} invalid/empty sequences.")
    return _batched_encode_texts(cleaned_sequences, model_name=encoder_name, batch_size=batch_size, device=device)


def encode_target_sequences_with_tokens(
    seq_list,
    encoder_name,
    batch_size,
    device,
    max_protein_length,
    include_contact,
):
    cleaned_sequences = []
    invalid_count = 0
    for idx, seq in enumerate(seq_list):
        cleaned = clean_protein_sequence(seq, max_length=max_protein_length)
        if not cleaned:
            invalid_count += 1
            warnings.warn(f"Invalid or empty protein sequence at index {idx}; using zero vector.", RuntimeWarning)
        cleaned_sequences.append(cleaned)
    if invalid_count > 0:
        print(f"[warning] Target encoder skipped {invalid_count} invalid/empty sequences.")
    return _batched_encode_texts_with_tokens(
        cleaned_sequences,
        model_name=encoder_name,
        batch_size=batch_size,
        device=device,
        include_contact=include_contact,
    )


def build_static_drug_features(smiles_list, morgan_bits, morgan_radius):
    return np.stack(
        [featurize_smiles(smiles, morgan_bits=morgan_bits, morgan_radius=morgan_radius) for smiles in smiles_list],
        axis=0,
    ).astype(np.float32)


def build_static_target_features(sequences, target_tfidf_dim, seed):
    cleaned_sequences = [clean_protein_sequence(seq, max_length=None) for seq in sequences]
    target_aac = np.stack([amino_acid_composition(seq) for seq in cleaned_sequences], axis=0)
    target_dipep = np.stack([dipeptide_composition(seq) for seq in cleaned_sequences], axis=0)
    target_phys = np.stack([basic_sequence_physchem(seq) for seq in cleaned_sequences], axis=0)
    target_tfidf = build_target_tfidf_svd(cleaned_sequences, svd_dim=target_tfidf_dim, seed=seed)
    return np.concatenate([target_aac, target_dipep, target_phys, target_tfidf], axis=1).astype(np.float32)


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", RuntimeWarning)
        return "cpu"
    return device


def normalize_weighted_adjacency(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.size == 0:
        return adjacency
    adjacency = adjacency + np.eye(adjacency.shape[0], dtype=np.float32)
    degree = adjacency.sum(axis=1)
    degree = np.clip(degree, a_min=1e-6, a_max=None)
    inv_sqrt_degree = 1.0 / np.sqrt(degree)
    return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


def compute_feature_stats(feature_matrices):
    feature_dim = 0
    for features in feature_matrices:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] > 0:
            feature_dim = arr.shape[1]
            break
    if feature_dim <= 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)

    total_sum = np.zeros((feature_dim,), dtype=np.float64)
    total_sq_sum = np.zeros((feature_dim,), dtype=np.float64)
    total_count = 0
    for features in feature_matrices:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        total_sum += arr.sum(axis=0, dtype=np.float64)
        total_sq_sum += np.square(arr, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_count += int(arr.shape[0])

    if total_count <= 0:
        return np.zeros((feature_dim,), dtype=np.float32), np.ones((feature_dim,), dtype=np.float32)

    mean = total_sum / float(total_count)
    variance = np.maximum(total_sq_sum / float(total_count) - np.square(mean), 1e-6)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_feature_matrix(features, mean, std):
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr.astype(np.float32)
    if mean.size == 0 or std.size == 0:
        return arr.astype(np.float32)
    return ((arr - mean) / np.clip(std, a_min=1e-6, a_max=None)).astype(np.float32)


def weighted_pagerank(adjacency, damping=0.85, max_iter=100, tol=1e-6):
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    num_nodes = adjacency.shape[0]
    transition = adjacency.copy()
    row_sums = transition.sum(axis=1, keepdims=True)
    dangling = row_sums.squeeze(-1) <= 0
    if np.any(dangling):
        transition[dangling] = 1.0
        row_sums = transition.sum(axis=1, keepdims=True)
    transition = transition / np.clip(row_sums, a_min=1e-6, a_max=None)

    ranks = np.full((num_nodes,), 1.0 / float(num_nodes), dtype=np.float32)
    teleport = np.full((num_nodes,), (1.0 - damping) / float(num_nodes), dtype=np.float32)
    for _ in range(int(max_iter)):
        next_ranks = teleport + damping * (transition.T @ ranks)
        if np.linalg.norm(next_ranks - ranks, ord=1) < tol:
            ranks = next_ranks.astype(np.float32)
            break
        ranks = next_ranks.astype(np.float32)
    ranks = np.clip(ranks, a_min=0.0, a_max=None)
    denom = ranks.sum()
    if denom <= 0:
        return np.full((num_nodes,), 1.0 / float(num_nodes), dtype=np.float32)
    return (ranks / denom).astype(np.float32)


def pagerank_weighted_readout(node_embeddings, adjacency):
    node_embeddings = np.asarray(node_embeddings, dtype=np.float32)
    if node_embeddings.ndim != 2 or node_embeddings.shape[0] == 0:
        hidden_dim = node_embeddings.shape[1] if node_embeddings.ndim == 2 else 0
        return np.zeros((hidden_dim,), dtype=np.float32)
    scores = weighted_pagerank(adjacency)
    if scores.size == 0:
        return np.zeros((node_embeddings.shape[1],), dtype=np.float32)
    return (scores[:, None] * node_embeddings).sum(axis=0).astype(np.float32)


class DrugGINEConv(torch.nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.edge_proj = torch.nn.Linear(edge_dim, hidden_dim)
        self.eps = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return self.norm(h + self.mlp((1.0 + self.eps) * h))
        src = edge_index[0]
        dst = edge_index[1]
        messages = torch.relu(h[src] + self.edge_proj(edge_attr))
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)
        updated = self.mlp((1.0 + self.eps) * h + aggregated)
        return self.norm(h + updated)


class DrugStructureEncoder(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [DrugGINEConv(hidden_dim=hidden_dim, edge_dim=edge_dim) for _ in range(int(max(1, num_layers)))]
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, node_features, edge_index, edge_attr):
        h = self.input_proj(node_features)
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr)
        return h

    def forward(self, node_features, edge_index, edge_attr):
        h = self.encode(node_features=node_features, edge_index=edge_index, edge_attr=edge_attr)
        recon = self.decoder(h)
        return h, recon


class ProteinStructureLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.backbone_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.spatial_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attn_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attn_src = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.attn_dst = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.edge_gamma = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, h, backbone_adj, spatial_adj):
        if h.size(0) == 0:
            return h

        backbone_deg = backbone_adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        backbone_msg = torch.matmul(backbone_adj / backbone_deg, self.backbone_linear(h))

        spatial_mask = spatial_adj > 0
        spatial_proj = self.spatial_linear(h)
        attn_hidden = torch.tanh(self.attn_proj(h))
        attn_scores = self.attn_src(attn_hidden) + self.attn_dst(attn_hidden).transpose(0, 1)
        attn_scores = attn_scores + self.edge_gamma * spatial_adj
        attn_scores = attn_scores.masked_fill(~spatial_mask, float("-inf"))
        if torch.any(spatial_mask):
            spatial_alpha = torch.softmax(attn_scores, dim=-1)
            spatial_alpha = torch.nan_to_num(spatial_alpha, nan=0.0, posinf=0.0, neginf=0.0)
            spatial_alpha = spatial_alpha * spatial_mask.float()
            spatial_msg = torch.matmul(spatial_alpha, spatial_proj)
        else:
            spatial_msg = torch.zeros_like(backbone_msg)

        fused = torch.cat([backbone_msg, spatial_msg], dim=-1)
        updated = self.update_mlp(fused)
        return self.norm(h + updated)


class ProteinStructureEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [ProteinStructureLayer(hidden_dim=hidden_dim) for _ in range(int(max(1, num_layers)))]
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, node_features, backbone_adj, spatial_adj):
        h = self.input_proj(node_features)
        for layer in self.layers:
            h = layer(h, backbone_adj=backbone_adj, spatial_adj=spatial_adj)
        return h

    def forward(self, node_features, backbone_adj, spatial_adj):
        h = self.encode(node_features=node_features, backbone_adj=backbone_adj, spatial_adj=spatial_adj)
        recon = self.decoder(h)
        return h, recon


class EntityGraphAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(input_dim, input_dim),
        )

    def forward(self, node_features, normalized_adjacency):
        smoothed = normalized_adjacency @ node_features
        latent = self.encoder(smoothed)
        recon = self.decoder(latent)
        return latent, recon, smoothed


def simple_graph_readout(node_features, adjacency, steps=2):
    node_features = np.asarray(node_features, dtype=np.float32)
    if node_features.ndim != 2 or node_features.shape[0] == 0:
        return np.zeros((node_features.shape[1] * (steps + 1),), dtype=np.float32)
    normalized_adjacency = normalize_weighted_adjacency(adjacency)
    states = [node_features]
    current = node_features
    for _ in range(steps):
        current = normalized_adjacency @ current
        current = np.tanh(current).astype(np.float32)
        states.append(current)
    pooled = [state.mean(axis=0) for state in states]
    return np.concatenate(pooled, axis=0).astype(np.float32)


def build_drug_entity_graph(smiles):
    smiles = (smiles or "").strip()
    if not smiles:
        return np.zeros((1, len(ATOM_SYMBOLS) + 6 + len(HYBRIDIZATION_TYPES) + 1 + 6), dtype=np.float32), np.zeros(
            (1, 1), dtype=np.float32
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros((1, len(ATOM_SYMBOLS) + 6 + len(HYBRIDIZATION_TYPES) + 1 + 6), dtype=np.float32), np.zeros(
            (1, 1), dtype=np.float32
        )

    node_features = np.stack([featurize_atom(atom) for atom in mol.GetAtoms()], axis=0).astype(np.float32)
    adjacency = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        w = float(bond.GetBondTypeAsDouble())
        adjacency[i, j] = w
        adjacency[j, i] = w
    return node_features, adjacency


def build_target_entity_graph(residue_embeddings, topk, graph_mode, contact_map=None, contact_threshold=0.5):
    residue_embeddings = np.asarray(residue_embeddings, dtype=np.float32)
    if residue_embeddings.ndim != 2 or residue_embeddings.shape[0] == 0:
        hidden_dim = residue_embeddings.shape[1] if residue_embeddings.ndim == 2 and residue_embeddings.shape[1] > 0 else 1
        return np.zeros((1, hidden_dim), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    num_residues = residue_embeddings.shape[0]
    adjacency = np.zeros((num_residues, num_residues), dtype=np.float32)
    if num_residues > 1:
        idx = np.arange(num_residues - 1)
        adjacency[idx, idx + 1] = 1.0
        adjacency[idx + 1, idx] = 1.0

    mode = (graph_mode or "contact").lower()
    if mode == "contact":
        if contact_map is None or np.asarray(contact_map).size == 0:
            mode = "approx"
        else:
            cm = np.asarray(contact_map, dtype=np.float32)[:num_residues, :num_residues]
            mask = cm >= float(contact_threshold)
            adjacency = np.maximum(adjacency, np.where(mask, cm, 0.0).astype(np.float32))
    if mode == "approx":
        norms = np.linalg.norm(residue_embeddings, axis=1, keepdims=True)
        normalized = residue_embeddings / np.clip(norms, a_min=1e-6, a_max=None)
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)
        effective_topk = min(max(int(topk), 0), max(num_residues - 1, 0))
        if effective_topk > 0:
            for i in range(num_residues):
                neighbors = np.argpartition(-similarity[i], effective_topk)[:effective_topk]
                for j in neighbors:
                    weight = max(float(similarity[i, j]), 0.0)
                    if weight > 0:
                        adjacency[i, j] = max(adjacency[i, j], weight)
                        adjacency[j, i] = max(adjacency[j, i], weight)
    return residue_embeddings, adjacency


def train_entity_graph_autoencoder_embeddings(
    graph_items,
    latent_dim,
    epochs,
    lr,
    weight_decay,
    device,
    verbose=False,
):
    if not graph_items:
        return np.zeros((0, latent_dim), dtype=np.float32)
    input_dim = int(graph_items[0][0].shape[1])
    if input_dim <= 0:
        return np.zeros((len(graph_items), latent_dim), dtype=np.float32)

    prepared = []
    for node_features, adjacency in graph_items:
        x = np.asarray(node_features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != input_dim or x.shape[0] == 0:
            x = np.zeros((1, input_dim), dtype=np.float32)
        a = np.asarray(adjacency, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] != x.shape[0] or a.shape[1] != x.shape[0]:
            a = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
        prepared.append((x, normalize_weighted_adjacency(a)))

    model = EntityGraphAutoEncoder(input_dim=input_dim, latent_dim=int(max(1, latent_dim))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    epochs = int(max(1, epochs))
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x_np, a_np in prepared:
            x = torch.from_numpy(x_np).to(device)
            a = torch.from_numpy(a_np).to(device)
            optimizer.zero_grad()
            _, recon, target = model(x, a)
            loss = torch.nn.functional.mse_loss(recon, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if verbose and (epoch == 1 or epoch == epochs):
            avg_loss = total_loss / max(len(prepared), 1)
            print(f"[info] Entity graph AE epoch {epoch}/{epochs} recon_loss={avg_loss:.6f}")

    model.eval()
    embeddings = []
    with torch.no_grad():
        for x_np, a_np in prepared:
            x = torch.from_numpy(x_np).to(device)
            a = torch.from_numpy(a_np).to(device)
            latent, _, _ = model(x, a)
            embeddings.append(latent.mean(dim=0).detach().cpu().numpy().astype(np.float32))
    return np.stack(embeddings, axis=0).astype(np.float32)


class TypeAwareProteinGraphLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.seq_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cont_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attn_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attn_src = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.attn_dst = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.gamma = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gate_linear = torch.nn.Linear(hidden_dim * 3, 1)
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, h, seq_adj, cont_adj, cont_weight, node_mask):
        seq_deg = seq_adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        seq_msg = torch.bmm(seq_adj / seq_deg, self.seq_linear(h))

        cont_value = self.cont_linear(h)
        attn_h = torch.tanh(self.attn_proj(h))
        attn_src = self.attn_src(attn_h)
        attn_dst = self.attn_dst(attn_h)
        edge_scores = attn_src + attn_dst.transpose(1, 2) + self.gamma * cont_weight
        edge_scores = edge_scores.masked_fill(cont_adj <= 0, float("-inf"))
        has_neighbor = cont_adj.sum(dim=-1, keepdim=True) > 0
        attn = torch.zeros_like(cont_weight)
        if torch.any(has_neighbor):
            attn = torch.softmax(edge_scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0) * cont_adj
        cont_msg = torch.bmm(attn, cont_value)

        gate_input = torch.cat([h, seq_msg, cont_msg], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        fused = gate * seq_msg + (1.0 - gate) * cont_msg
        updated = self.norm(h + self.update_mlp(fused))
        return updated * node_mask.unsqueeze(-1)


class TypeAwareProteinGraphSSL(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList([TypeAwareProteinGraphLayer(hidden_dim) for _ in range(int(max(1, num_layers)))])
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool_attn_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.pool_attn_score = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.pool_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def encode_from_projected(self, h0, seq_adj, cont_adj, cont_weight, node_mask):
        h = h0 * node_mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, seq_adj=seq_adj, cont_adj=cont_adj, cont_weight=cont_weight, node_mask=node_mask)
        return h

    def project_inputs(self, residue_features):
        return self.input_proj(residue_features)

    def pool_graph(self, h, node_mask):
        attn_logits = self.pool_attn_score(torch.tanh(self.pool_attn_proj(h))).squeeze(-1)
        attn_logits = attn_logits.masked_fill(node_mask <= 0, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0) * node_mask
        z_attn = torch.sum(attn.unsqueeze(-1) * h, dim=1)

        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        z_mean = torch.sum(h * node_mask.unsqueeze(-1), dim=1) / denom

        masked_h = h.masked_fill(node_mask.unsqueeze(-1) <= 0, float("-inf"))
        z_max = masked_h.max(dim=1).values
        z_max = torch.nan_to_num(z_max, nan=0.0, posinf=0.0, neginf=0.0)
        return self.pool_mlp(torch.cat([z_attn, z_mean, z_max], dim=-1))

    def forward(self, residue_features, seq_adj, cont_adj, cont_weight, node_mask):
        h0 = self.project_inputs(residue_features)
        h = self.encode_from_projected(h0, seq_adj=seq_adj, cont_adj=cont_adj, cont_weight=cont_weight, node_mask=node_mask)
        recon = self.decoder(h)
        z = self.pool_graph(h, node_mask=node_mask)
        return h0, h, recon, z


def build_topk_contact_graph(contact_map, length, topk, exclusion_delta, threshold):
    num_nodes = int(max(length, 0))
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if num_nodes <= 1:
        return adjacency
    if contact_map is None or np.asarray(contact_map).size == 0:
        return adjacency
    cm = np.asarray(contact_map, dtype=np.float32)[:num_nodes, :num_nodes]
    exclusion_delta = int(max(exclusion_delta, 0))
    effective_topk = min(max(int(topk), 0), max(num_nodes - 1, 0))
    if effective_topk <= 0:
        return adjacency
    for i in range(num_nodes):
        scores = cm[i].copy()
        scores[i] = -np.inf
        local_left = max(0, i - exclusion_delta)
        local_right = min(num_nodes, i + exclusion_delta + 1)
        scores[local_left:local_right] = -np.inf
        valid_idx = np.where(scores > float(threshold))[0]
        if valid_idx.size == 0:
            continue
        select_k = min(effective_topk, valid_idx.size)
        candidate_scores = scores[valid_idx]
        topk_local = np.argpartition(-candidate_scores, select_k - 1)[:select_k]
        neighbors = valid_idx[topk_local]
        for j in neighbors.tolist():
            weight = max(float(scores[j]), 0.0)
            if weight > 0:
                adjacency[i, j] = max(adjacency[i, j], weight)
                adjacency[j, i] = max(adjacency[j, i], weight)
    return adjacency


def build_sequential_adjacency(length):
    adjacency = np.zeros((length, length), dtype=np.float32)
    if length > 1:
        idx = np.arange(length - 1)
        adjacency[idx, idx + 1] = 1.0
        adjacency[idx + 1, idx] = 1.0
    return adjacency


def prepare_target_ssl_graphs(
    residue_embedding_list,
    contact_map_list,
    max_graph_length,
    contact_topk,
    contact_exclusion_delta,
    contact_threshold,
):
    graphs = []
    for idx, residue_embeddings in enumerate(residue_embedding_list or []):
        residues = np.asarray(residue_embeddings, dtype=np.float32)
        if max_graph_length and max_graph_length > 0:
            residues = residues[:max_graph_length]
        if residues.ndim != 2 or residues.shape[0] == 0:
            hidden_dim = residues.shape[1] if residues.ndim == 2 and residues.shape[1] > 0 else 1
            residues = np.zeros((1, hidden_dim), dtype=np.float32)
        contact_map = None
        if contact_map_list is not None and idx < len(contact_map_list):
            contact_map = contact_map_list[idx]
            if max_graph_length and max_graph_length > 0 and np.asarray(contact_map).size > 0:
                contact_map = np.asarray(contact_map, dtype=np.float32)[: residues.shape[0], : residues.shape[0]]
        if contact_map is None or np.asarray(contact_map).size == 0:
            raise ValueError("Target graph SSL requires contact maps from ESM, but contact_map is unavailable.")
        seq_adj = build_sequential_adjacency(residues.shape[0])
        cont_adj = build_topk_contact_graph(
            contact_map=contact_map,
            length=residues.shape[0],
            topk=contact_topk,
            exclusion_delta=contact_exclusion_delta,
            threshold=contact_threshold,
        )
        graphs.append(
            {
                "residue_features": residues.astype(np.float32),
                "seq_adj": seq_adj.astype(np.float32),
                "cont_adj": cont_adj.astype(np.float32),
                "contact_map": np.asarray(contact_map, dtype=np.float32),
            }
        )
    return graphs


def pad_target_ssl_batch(graph_batch):
    batch_size = len(graph_batch)
    max_len = max(graph["residue_features"].shape[0] for graph in graph_batch)
    feat_dim = graph_batch[0]["residue_features"].shape[1]
    residue_features = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
    seq_adj = np.zeros((batch_size, max_len, max_len), dtype=np.float32)
    cont_adj = np.zeros((batch_size, max_len, max_len), dtype=np.float32)
    node_mask = np.zeros((batch_size, max_len), dtype=np.float32)
    lengths = []
    for batch_idx, graph in enumerate(graph_batch):
        length = graph["residue_features"].shape[0]
        residue_features[batch_idx, :length] = graph["residue_features"]
        seq_adj[batch_idx, :length, :length] = graph["seq_adj"]
        cont_adj[batch_idx, :length, :length] = graph["cont_adj"]
        node_mask[batch_idx, :length] = 1.0
        lengths.append(length)
    return residue_features, seq_adj, cont_adj, node_mask, lengths


def build_ssl_views(node_mask, cont_adj, mask_ratio, contact_dropout, device):
    node_mask = node_mask.to(device)
    base_mask = node_mask.unsqueeze(-1)
    masked_views = []
    cont_views = []
    for _ in range(2):
        rand = torch.rand_like(node_mask)
        visible = ((rand > mask_ratio).float() * node_mask).to(device)
        empty_rows = visible.sum(dim=1) <= 0
        if torch.any(empty_rows):
            first_idx = node_mask.argmax(dim=1)
            visible[empty_rows, first_idx[empty_rows]] = 1.0
        edge_keep = (torch.rand_like(cont_adj) > contact_dropout).float().to(device)
        edge_keep = torch.triu(edge_keep, diagonal=1)
        edge_keep = edge_keep + edge_keep.transpose(1, 2)
        edge_keep = edge_keep * (cont_adj > 0).float()
        masked_views.append(visible.unsqueeze(-1) * base_mask)
        cont_views.append(cont_adj * edge_keep)
    return masked_views, cont_views


def info_nce_loss(z1, z2, temperature):
    z1 = torch.nn.functional.normalize(z1, dim=-1)
    z2 = torch.nn.functional.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.transpose(0, 1)) / float(max(temperature, 1e-6))
    targets = torch.arange(z1.shape[0], device=z1.device)
    return torch.nn.functional.cross_entropy(logits, targets)


def train_target_graph_ssl_embeddings(
    residue_embedding_list,
    contact_map_list,
    output_dim,
    num_layers,
    contact_topk,
    contact_exclusion_delta,
    contact_threshold,
    mask_ratio,
    contact_dropout,
    contrastive_lambda,
    epochs,
    batch_size,
    lr,
    weight_decay,
    max_graph_length,
    device,
    seed,
    verbose=False,
):
    graphs = prepare_target_ssl_graphs(
        residue_embedding_list=residue_embedding_list,
        contact_map_list=contact_map_list,
        max_graph_length=max_graph_length,
        contact_topk=contact_topk,
        contact_exclusion_delta=contact_exclusion_delta,
        contact_threshold=contact_threshold,
    )
    if not graphs:
        return np.zeros((0, output_dim), dtype=np.float32), [], []

    input_dim = graphs[0]["residue_features"].shape[1]
    model = TypeAwareProteinGraphSSL(input_dim=input_dim, hidden_dim=output_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    epochs = int(max(1, epochs))
    batch_size = int(max(1, batch_size))

    model.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(graphs))
        epoch_total = 0.0
        epoch_mask_total = 0.0
        epoch_con_total = 0.0
        batch_count = 0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size].tolist()
            batch_graphs = [graphs[idx] for idx in batch_indices]
            x_np, seq_np, cont_np, node_mask_np, _ = pad_target_ssl_batch(batch_graphs)
            x = torch.from_numpy(x_np).to(device)
            seq_adj = torch.from_numpy(seq_np).to(device)
            cont_adj = torch.from_numpy(cont_np).to(device)
            node_mask = torch.from_numpy(node_mask_np).to(device)

            masked_views, cont_views = build_ssl_views(
                node_mask=node_mask,
                cont_adj=cont_adj,
                mask_ratio=mask_ratio,
                contact_dropout=contact_dropout,
                device=device,
            )

            optimizer.zero_grad()
            h0 = model.project_inputs(x)
            h_view1 = model.encode_from_projected(
                h0 * masked_views[0],
                seq_adj=seq_adj,
                cont_adj=(cont_views[0] > 0).float(),
                cont_weight=cont_views[0],
                node_mask=node_mask,
            )
            recon = model.decoder(h_view1)
            masked_nodes = ((masked_views[0].squeeze(-1) <= 0) & (node_mask > 0)).float()
            recon_diff = ((recon - h0) ** 2).mean(dim=-1)
            mask_loss = (recon_diff * masked_nodes).sum() / masked_nodes.sum().clamp_min(1.0)

            h_view2 = model.encode_from_projected(
                h0 * masked_views[1],
                seq_adj=seq_adj,
                cont_adj=(cont_views[1] > 0).float(),
                cont_weight=cont_views[1],
                node_mask=node_mask,
            )
            z1 = model.pool_graph(h_view1, node_mask=node_mask)
            z2 = model.pool_graph(h_view2, node_mask=node_mask)
            con_loss = 0.5 * (info_nce_loss(z1, z2, temperature=0.2) + info_nce_loss(z2, z1, temperature=0.2))
            loss = mask_loss + float(max(contrastive_lambda, 0.0)) * con_loss
            loss.backward()
            optimizer.step()

            epoch_total += float(loss.item())
            epoch_mask_total += float(mask_loss.item())
            epoch_con_total += float(con_loss.item())
            batch_count += 1
        if verbose:
            denom = max(batch_count, 1)
            print(
                f"[info] Target graph SSL epoch {epoch}/{epochs} "
                f"loss={epoch_total / denom:.6f} "
                f"mask_loss={epoch_mask_total / denom:.6f} "
                f"con_loss={epoch_con_total / denom:.6f}"
            )

    model.eval()
    graph_embeddings = []
    residue_outputs = []
    contact_graphs = []
    with torch.no_grad():
        for graph in graphs:
            x = torch.from_numpy(graph["residue_features"][None, ...]).to(device)
            seq_adj = torch.from_numpy(graph["seq_adj"][None, ...]).to(device)
            cont_adj = torch.from_numpy(graph["cont_adj"][None, ...]).to(device)
            node_mask = torch.ones((1, graph["residue_features"].shape[0]), dtype=torch.float32, device=device)
            _, h, _, z = model(
                residue_features=x,
                seq_adj=seq_adj,
                cont_adj=(cont_adj > 0).float(),
                cont_weight=cont_adj,
                node_mask=node_mask,
            )
            graph_embeddings.append(z.squeeze(0).detach().cpu().numpy().astype(np.float32))
            residue_outputs.append(h.squeeze(0).detach().cpu().numpy().astype(np.float32))
            contact_graphs.append(graph["cont_adj"].astype(np.float32))
    return np.stack(graph_embeddings, axis=0).astype(np.float32), residue_outputs, contact_graphs


def featurize_atom(atom):
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

    charge = np.asarray([float(atom.GetFormalCharge())], dtype=np.float32)
    aromatic = np.asarray([float(atom.GetIsAromatic())], dtype=np.float32)
    hydrogens = np.asarray([float(atom.GetTotalNumHs())], dtype=np.float32)
    mass = np.asarray([float(atom.GetMass()) / 200.0], dtype=np.float32)
    in_ring = np.asarray([float(atom.IsInRing())], dtype=np.float32)
    chirality = np.asarray([float(atom.GetChiralTag())], dtype=np.float32)

    return np.concatenate(
        [
            symbol_onehot,
            degree_onehot,
            hyb_onehot,
            charge,
            aromatic,
            hydrogens,
            mass,
            in_ring,
            chirality,
        ],
        axis=0,
    ).astype(np.float32)


def featurize_bond(bond):
    bond_type = bond.GetBondType()
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type_onehot = np.zeros((len(bond_types) + 1,), dtype=np.float32)
    if bond_type in bond_types:
        bond_type_onehot[bond_types.index(bond_type)] = 1.0
    else:
        bond_type_onehot[-1] = 1.0

    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    stereo_onehot = np.zeros((len(stereo_types) + 1,), dtype=np.float32)
    stereo = bond.GetStereo()
    if stereo in stereo_types:
        stereo_onehot[stereo_types.index(stereo)] = 1.0
    else:
        stereo_onehot[-1] = 1.0

    return np.concatenate(
        [
            bond_type_onehot,
            np.asarray([float(bond.GetIsConjugated())], dtype=np.float32),
            np.asarray([float(bond.IsInRing())], dtype=np.float32),
            stereo_onehot,
        ],
        axis=0,
    ).astype(np.float32)


def build_drug_graph_embedding(smiles, steps):
    smiles = (smiles or "").strip()
    if not smiles:
        return np.zeros((93,), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros((93,), dtype=np.float32)

    atoms = [featurize_atom(atom) for atom in mol.GetAtoms()]
    node_features = np.stack(atoms, axis=0).astype(np.float32)
    adjacency = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        weight = float(bond.GetBondTypeAsDouble())
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    return simple_graph_readout(node_features, adjacency, steps=steps)


def build_drug_graph_embeddings(smiles_list, steps):
    return np.stack([build_drug_graph_embedding(smiles, steps=steps) for smiles in smiles_list], axis=0).astype(np.float32)


def build_edge_index_and_attr(num_nodes, edges):
    if num_nodes <= 0 or not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    edge_index = []
    edge_attr = []
    edge_dim = len(edges[0][2])
    for src, dst, attr in edges:
        attr = np.asarray(attr, dtype=np.float32)
        edge_index.append([src, dst])
        edge_attr.append(attr)
        edge_index.append([dst, src])
        edge_attr.append(attr.copy())
    return np.asarray(edge_index, dtype=np.int64).T, np.asarray(edge_attr, dtype=np.float32).reshape(-1, edge_dim)


def build_spatial_contact_adjacency(contact_map, threshold):
    cm = np.asarray(contact_map, dtype=np.float32)
    if cm.ndim != 2 or cm.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    weighted = np.where(cm >= float(threshold), cm, 0.0).astype(np.float32)
    np.fill_diagonal(weighted, 0.0)
    return np.maximum(weighted, weighted.T).astype(np.float32)


def prepare_drug_structure_graphs(smiles_list):
    graph_items = []
    node_dim = len(ATOM_SYMBOLS) + 6 + len(HYBRIDIZATION_TYPES) + 1 + 6
    edge_dim = 5 + 1 + 1 + 6
    for smiles in smiles_list:
        smiles = (smiles or "").strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or mol.GetNumAtoms() == 0:
            graph_items.append(
                {
                    "node_features": np.zeros((1, node_dim), dtype=np.float32),
                    "edge_index": np.zeros((2, 0), dtype=np.int64),
                    "edge_attr": np.zeros((0, edge_dim), dtype=np.float32),
                    "pagerank_adj": np.zeros((1, 1), dtype=np.float32),
                }
            )
            continue

        node_features = np.stack([featurize_atom(atom) for atom in mol.GetAtoms()], axis=0).astype(np.float32)
        undirected_edges = []
        pagerank_adj = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = featurize_bond(bond)
            undirected_edges.append((i, j, edge_feature))
            pagerank_adj[i, j] = 1.0
            pagerank_adj[j, i] = 1.0
        edge_index, edge_attr = build_edge_index_and_attr(mol.GetNumAtoms(), undirected_edges)
        graph_items.append(
            {
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "pagerank_adj": pagerank_adj.astype(np.float32),
            }
        )
    return graph_items


def prepare_protein_structure_graphs(residue_embedding_list, contact_map_list, max_graph_length, contact_threshold):
    graph_items = []
    for idx, residue_embeddings in enumerate(residue_embedding_list or []):
        residues = np.asarray(residue_embeddings, dtype=np.float32)
        if max_graph_length and max_graph_length > 0:
            residues = residues[:max_graph_length]
        if residues.ndim != 2 or residues.shape[0] == 0:
            hidden_dim = residues.shape[1] if residues.ndim == 2 and residues.shape[1] > 0 else 1
            residues = np.zeros((1, hidden_dim), dtype=np.float32)

        backbone_adj = build_sequential_adjacency(residues.shape[0]).astype(np.float32)
        if contact_map_list is None or idx >= len(contact_map_list):
            raise ValueError("Protein structure encoding requires ESM contact maps, but none were provided.")
        contact_map = np.asarray(contact_map_list[idx], dtype=np.float32)
        if contact_map.size == 0:
            raise ValueError("Protein structure encoding requires non-empty ESM contact maps.")
        if max_graph_length and max_graph_length > 0:
            contact_map = contact_map[: residues.shape[0], : residues.shape[0]]
        spatial_adj = build_spatial_contact_adjacency(contact_map, threshold=contact_threshold)
        pagerank_adj = (backbone_adj + spatial_adj).astype(np.float32)
        graph_items.append(
            {
                "node_features": residues.astype(np.float32),
                "backbone_adj": backbone_adj,
                "spatial_adj": spatial_adj,
                "pagerank_adj": pagerank_adj,
            }
        )
    return graph_items


def train_drug_structure_embeddings(
    graph_items,
    hidden_dim,
    num_layers,
    epochs,
    lr,
    weight_decay,
    device,
    verbose=False,
):
    if not graph_items:
        return np.zeros((0, hidden_dim), dtype=np.float32), []

    node_mean, node_std = compute_feature_stats([item["node_features"] for item in graph_items])
    edge_mean, edge_std = compute_feature_stats([item["edge_attr"] for item in graph_items])
    processed_items = []
    for item in graph_items:
        processed_items.append(
            {
                "node_features": standardize_feature_matrix(item["node_features"], node_mean, node_std),
                "edge_index": item["edge_index"].astype(np.int64),
                "edge_attr": standardize_feature_matrix(item["edge_attr"], edge_mean, edge_std),
                "pagerank_adj": item["pagerank_adj"].astype(np.float32),
            }
        )

    input_dim = processed_items[0]["node_features"].shape[1]
    edge_dim = processed_items[0]["edge_attr"].shape[1] if processed_items[0]["edge_attr"].ndim == 2 else 0
    model = DrugStructureEncoder(
        input_dim=input_dim,
        edge_dim=edge_dim,
        hidden_dim=int(max(1, hidden_dim)),
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(max(1, epochs))
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for item in processed_items:
            x = torch.from_numpy(item["node_features"]).to(device)
            edge_index = torch.from_numpy(item["edge_index"]).to(device)
            edge_attr = torch.from_numpy(item["edge_attr"]).to(device)
            optimizer.zero_grad()
            _, recon = model(node_features=x, edge_index=edge_index, edge_attr=edge_attr)
            loss = torch.nn.functional.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if verbose and (epoch == 1 or epoch == epochs):
            avg_loss = total_loss / max(len(processed_items), 1)
            print(f"[info] Drug structure AE epoch {epoch}/{epochs} recon_loss={avg_loss:.6f}")

    model.eval()
    graph_embeddings = []
    node_outputs = []
    with torch.no_grad():
        for item in processed_items:
            x = torch.from_numpy(item["node_features"]).to(device)
            edge_index = torch.from_numpy(item["edge_index"]).to(device)
            edge_attr = torch.from_numpy(item["edge_attr"]).to(device)
            h = model.encode(node_features=x, edge_index=edge_index, edge_attr=edge_attr)
            h_np = h.detach().cpu().numpy().astype(np.float32)
            graph_embeddings.append(pagerank_weighted_readout(h_np, item["pagerank_adj"]))
            node_outputs.append(h_np)
    return np.stack(graph_embeddings, axis=0).astype(np.float32), node_outputs


def train_protein_structure_embeddings(
    graph_items,
    hidden_dim,
    num_layers,
    epochs,
    lr,
    weight_decay,
    device,
    verbose=False,
):
    if not graph_items:
        return np.zeros((0, hidden_dim), dtype=np.float32), [], []

    node_mean, node_std = compute_feature_stats([item["node_features"] for item in graph_items])
    processed_items = []
    for item in graph_items:
        processed_items.append(
            {
                "node_features": standardize_feature_matrix(item["node_features"], node_mean, node_std),
                "backbone_adj": item["backbone_adj"].astype(np.float32),
                "spatial_adj": item["spatial_adj"].astype(np.float32),
                "pagerank_adj": item["pagerank_adj"].astype(np.float32),
            }
        )

    input_dim = processed_items[0]["node_features"].shape[1]
    model = ProteinStructureEncoder(
        input_dim=input_dim,
        hidden_dim=int(max(1, hidden_dim)),
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(max(1, epochs))
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for item in processed_items:
            x = torch.from_numpy(item["node_features"]).to(device)
            backbone_adj = torch.from_numpy(item["backbone_adj"]).to(device)
            spatial_adj = torch.from_numpy(item["spatial_adj"]).to(device)
            optimizer.zero_grad()
            _, recon = model(node_features=x, backbone_adj=backbone_adj, spatial_adj=spatial_adj)
            loss = torch.nn.functional.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if verbose and (epoch == 1 or epoch == epochs):
            avg_loss = total_loss / max(len(processed_items), 1)
            print(f"[info] Protein structure AE epoch {epoch}/{epochs} recon_loss={avg_loss:.6f}")

    model.eval()
    graph_embeddings = []
    node_outputs = []
    pagerank_graphs = []
    with torch.no_grad():
        for item in processed_items:
            x = torch.from_numpy(item["node_features"]).to(device)
            backbone_adj = torch.from_numpy(item["backbone_adj"]).to(device)
            spatial_adj = torch.from_numpy(item["spatial_adj"]).to(device)
            h = model.encode(node_features=x, backbone_adj=backbone_adj, spatial_adj=spatial_adj)
            h_np = h.detach().cpu().numpy().astype(np.float32)
            graph_embeddings.append(pagerank_weighted_readout(h_np, item["pagerank_adj"]))
            node_outputs.append(h_np)
            pagerank_graphs.append(item["pagerank_adj"].astype(np.float32))
    return np.stack(graph_embeddings, axis=0).astype(np.float32), node_outputs, pagerank_graphs


def build_residue_graph_embedding(residue_embeddings, topk, steps, graph_mode, contact_map=None, contact_threshold=0.5):
    residue_embeddings = np.asarray(residue_embeddings, dtype=np.float32)
    if residue_embeddings.ndim != 2 or residue_embeddings.shape[0] == 0:
        hidden_dim = residue_embeddings.shape[1] if residue_embeddings.ndim == 2 else 0
        return np.zeros((hidden_dim * (steps + 1),), dtype=np.float32)

    num_residues = residue_embeddings.shape[0]
    adjacency = np.zeros((num_residues, num_residues), dtype=np.float32)

    # Sequence-local edges are a feasible approximation to backbone connectivity.
    if num_residues > 1:
        indices = np.arange(num_residues - 1)
        adjacency[indices, indices + 1] = 1.0
        adjacency[indices + 1, indices] = 1.0

    if graph_mode == "contact":
        if contact_map is None or contact_map.size == 0:
            warnings.warn(
                "Contact mode requested but contact map is unavailable; falling back to embedding kNN edges.",
                RuntimeWarning,
            )
            graph_mode = "approx"
        else:
            contact_map = np.asarray(contact_map, dtype=np.float32)
            contact_map = contact_map[:num_residues, :num_residues]
            mask = contact_map >= float(contact_threshold)
            weighted = np.where(mask, contact_map, 0.0).astype(np.float32)
            adjacency = np.maximum(adjacency, weighted)

    if graph_mode == "approx":
        # Use embedding-space kNN as a practical surrogate for contact-map edges.
        norms = np.linalg.norm(residue_embeddings, axis=1, keepdims=True)
        normalized = residue_embeddings / np.clip(norms, a_min=1e-6, a_max=None)
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)
        effective_topk = min(max(int(topk), 0), max(num_residues - 1, 0))
        if effective_topk > 0:
            for i in range(num_residues):
                neighbors = np.argpartition(-similarity[i], effective_topk)[:effective_topk]
                for j in neighbors:
                    weight = max(float(similarity[i, j]), 0.0)
                    if weight > 0:
                        adjacency[i, j] = max(adjacency[i, j], weight)
                        adjacency[j, i] = max(adjacency[j, i], weight)

    return simple_graph_readout(residue_embeddings, adjacency, steps=steps)


def build_target_graph_embeddings(
    residue_embedding_list,
    topk,
    steps,
    max_graph_length,
    graph_mode,
    contact_map_list=None,
    contact_threshold=0.5,
):
    graph_features = []
    hidden_dim = 0
    for idx, residue_embeddings in enumerate(residue_embedding_list):
        residue_embeddings = np.asarray(residue_embeddings, dtype=np.float32)
        if residue_embeddings.ndim == 2 and residue_embeddings.shape[1] > 0:
            hidden_dim = max(hidden_dim, residue_embeddings.shape[1])
        truncated = residue_embeddings[:max_graph_length] if max_graph_length and max_graph_length > 0 else residue_embeddings
        contact_map = None
        if contact_map_list is not None and idx < len(contact_map_list):
            contact_map = contact_map_list[idx]
            if max_graph_length and max_graph_length > 0 and contact_map.size > 0:
                contact_map = np.asarray(contact_map, dtype=np.float32)[:max_graph_length, :max_graph_length]
        graph_features.append(
            build_residue_graph_embedding(
                truncated,
                topk=topk,
                steps=steps,
                graph_mode=graph_mode,
                contact_map=contact_map,
                contact_threshold=contact_threshold,
            )
        )
    if hidden_dim == 0:
        return np.zeros((len(residue_embedding_list), 0), dtype=np.float32)
    expected_dim = hidden_dim * (steps + 1)
    normalized_features = []
    for feature in graph_features:
        if feature.shape[0] < expected_dim:
            pad = np.zeros((expected_dim - feature.shape[0],), dtype=np.float32)
            feature = np.concatenate([feature, pad], axis=0)
        normalized_features.append(feature.astype(np.float32))
    return np.stack(normalized_features, axis=0).astype(np.float32)


def generate_node_representation_llm(
    drug_path,
    target_path,
    output_path,
    output_dim,
    seed,
    drug_model_name,
    target_model_name,
    drug_batch_size,
    target_batch_size,
    device,
    max_protein_length,
    use_static_drug,
    use_static_target,
    use_drug_graph_branch,
    use_target_graph_branch,
    drug_graph_steps,
    target_graph_steps,
    target_graph_topk,
    target_graph_mode,
    target_contact_threshold,
    max_protein_graph_length,
    use_entity_graph_ae,
    entity_graph_ae_latent_dim,
    entity_graph_ae_epochs,
    entity_graph_ae_lr,
    entity_graph_ae_weight_decay,
    target_ssl_num_layers,
    target_ssl_mask_ratio,
    target_ssl_contact_dropout,
    target_ssl_contrastive_lambda,
    target_ssl_contact_exclusion_delta,
    morgan_bits,
    morgan_radius,
    target_tfidf_dim,
    encode_order,
    llm_cache_path="",
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device)

    if target_graph_mode != "contact":
        warnings.warn(
            "The structure-aware protein encoder now requires ESM contact maps; forcing target_graph_mode='contact'.",
            RuntimeWarning,
        )
        target_graph_mode = "contact"

    if use_target_graph_branch:
        print(f"[info] Target graph edge mode (startup): contact (threshold={target_contact_threshold:.2f})")
    else:
        print("[info] Target graph branch disabled.")

    drug_table = load_drug_table(drug_path)
    target_table = load_target_table(target_path)

    drug_ids = drug_table["drug_id"].to_numpy()
    target_ids = target_table["target_id"].to_numpy()
    drug_smiles = drug_table["SMILES"].tolist()
    target_fastas = target_table["FASTA"].tolist()

    target_residue_embeddings = None
    target_contact_maps = None
    drug_semantic = None
    target_semantic = None
    drug_structure_node_embeddings = None
    target_structure_node_embeddings = None
    target_pagerank_graphs = None

    def _encode_drug():
        print(f"[info] Encoding {len(drug_smiles)} drugs with {drug_model_name} on {device}")
        return encode_drug_smiles(
            smiles_list=drug_smiles,
            encoder_name=drug_model_name,
            batch_size=drug_batch_size,
            device=device,
        )

    def _encode_target():
        print(f"[info] Encoding {len(target_fastas)} targets with {target_model_name} on {device}")
        print(f"[info] Protein sequences longer than {max_protein_length} will be truncated for engineering simplicity.")
        return encode_target_sequences_with_tokens(
            seq_list=target_fastas,
            encoder_name=target_model_name,
            batch_size=target_batch_size,
            device=device,
            max_protein_length=max_protein_length,
            include_contact=True,
        )

    cache_output_path = llm_cache_path or make_non_overwriting_path(
        os.path.join(
            os.path.dirname(output_path) or ".",
            f"{os.path.splitext(os.path.basename(output_path))[0]}_llm_cache.npz",
        )
    )
    if llm_cache_path and os.path.exists(llm_cache_path):
        print(f"[info] Loading precomputed LLM cache from {llm_cache_path}")
        cache = load_precomputed_llm_cache(
            cache_path=llm_cache_path,
            drug_ids=drug_ids,
            target_ids=target_ids,
        )
        drug_semantic = cache["drug_semantic"]
        target_semantic = cache["target_semantic"]
        target_residue_embeddings = cache["target_residue_embeddings"]
        target_contact_maps = cache["target_contact_maps"]
    elif encode_order == "target_first":
        target_semantic, target_residue_embeddings, target_contact_maps = _encode_target()
        drug_semantic = _encode_drug()
    else:
        drug_semantic = _encode_drug()
        target_semantic, target_residue_embeddings, target_contact_maps = _encode_target()

    if drug_semantic is None or target_semantic is None:
        raise ValueError("Semantic embeddings were not generated correctly.")
    if target_residue_embeddings is None or target_contact_maps is None:
        raise ValueError("Target residue embeddings and contact maps are required for the structure-aware pipeline.")

    if (not llm_cache_path) or (llm_cache_path and not os.path.exists(llm_cache_path)):
        print(f"[info] Saving LLM cache to {cache_output_path}")
        save_precomputed_llm_cache(
            cache_path=cache_output_path,
            drug_ids=drug_ids,
            target_ids=target_ids,
            drug_semantic=drug_semantic,
            target_semantic=target_semantic,
            target_residue_embeddings=target_residue_embeddings,
            target_contact_maps=target_contact_maps,
        )

    drug_components = [drug_semantic.astype(np.float32)]
    target_components = [target_semantic.astype(np.float32)]
    if use_drug_graph_branch:
        print("[info] Training structure-aware drug encoder (GINE + reconstruction)")
        drug_graph_items = prepare_drug_structure_graphs(smiles_list=drug_smiles)
        drug_graph, drug_structure_node_embeddings = train_drug_structure_embeddings(
            graph_items=drug_graph_items,
            hidden_dim=entity_graph_ae_latent_dim,
            num_layers=drug_graph_steps,
            epochs=entity_graph_ae_epochs,
            lr=entity_graph_ae_lr,
            weight_decay=entity_graph_ae_weight_decay,
            device=device,
            verbose=True,
        )
        drug_components.append(drug_graph.astype(np.float32))
    if use_target_graph_branch:
        print("[info] Training structure-aware protein encoder (backbone/spatial attention + reconstruction)")
        protein_graph_items = prepare_protein_structure_graphs(
            residue_embedding_list=target_residue_embeddings,
            contact_map_list=target_contact_maps,
            max_graph_length=max_protein_graph_length,
            contact_threshold=target_contact_threshold,
        )
        target_graph, target_structure_node_embeddings, target_pagerank_graphs = train_protein_structure_embeddings(
            graph_items=protein_graph_items,
            hidden_dim=entity_graph_ae_latent_dim,
            num_layers=target_graph_steps,
            epochs=entity_graph_ae_epochs,
            lr=entity_graph_ae_lr,
            weight_decay=entity_graph_ae_weight_decay,
            device=device,
            verbose=True,
        )
        target_components.append(target_graph.astype(np.float32))
    if use_static_drug:
        print("[info] Concatenating static drug features (Morgan + MACCS + RDKit descriptors)")
        drug_static = build_static_drug_features(
            smiles_list=drug_smiles,
            morgan_bits=morgan_bits,
            morgan_radius=morgan_radius,
        )
        drug_components.append(drug_static.astype(np.float32))
    if use_static_target:
        print("[info] Concatenating static target features (AAC + dipeptide + physicochemical + TF-IDF/SVD)")
        target_static = build_static_target_features(
            sequences=target_fastas,
            target_tfidf_dim=target_tfidf_dim,
            seed=seed,
        )
        target_components.append(target_static.astype(np.float32))

    drug_raw = np.concatenate(drug_components, axis=1).astype(np.float32)
    target_raw = np.concatenate(target_components, axis=1).astype(np.float32)

    drug_projected = safe_project(drug_raw, output_dim=output_dim, seed=seed)
    target_projected = safe_project(target_raw, output_dim=output_dim, seed=seed)
    node_features = np.concatenate([drug_projected, target_projected], axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_kwargs = dict(
        drug_features=drug_projected.astype(np.float32),
        target_features=target_projected.astype(np.float32),
        drug_semantic=drug_semantic.astype(np.float32),
        target_semantic=target_semantic.astype(np.float32),
        drug_ids=drug_ids,
        target_ids=target_ids,
        node_features=node_features,
        output_dim=np.asarray([output_dim], dtype=np.int64),
        drug_encoder=np.asarray(["huggingface_smiles"], dtype="<U64"),
        target_encoder=np.asarray(["huggingface_protein"], dtype="<U64"),
        drug_model_name=np.asarray([drug_model_name], dtype="<U256"),
        target_model_name=np.asarray([target_model_name], dtype="<U256"),
        use_drug_graph_branch=np.asarray([int(use_drug_graph_branch)], dtype=np.int64),
        use_target_graph_branch=np.asarray([int(use_target_graph_branch)], dtype=np.int64),
        use_entity_graph_ae=np.asarray([int(use_entity_graph_ae)], dtype=np.int64),
        entity_graph_ae_latent_dim=np.asarray([int(entity_graph_ae_latent_dim)], dtype=np.int64),
        entity_graph_ae_epochs=np.asarray([int(entity_graph_ae_epochs)], dtype=np.int64),
        entity_graph_ae_lr=np.asarray([float(entity_graph_ae_lr)], dtype=np.float32),
        entity_graph_ae_weight_decay=np.asarray([float(entity_graph_ae_weight_decay)], dtype=np.float32),
        target_ssl_num_layers=np.asarray([int(target_ssl_num_layers)], dtype=np.int64),
        target_ssl_mask_ratio=np.asarray([float(target_ssl_mask_ratio)], dtype=np.float32),
        target_ssl_contact_dropout=np.asarray([float(target_ssl_contact_dropout)], dtype=np.float32),
        target_ssl_contrastive_lambda=np.asarray([float(target_ssl_contrastive_lambda)], dtype=np.float32),
        target_ssl_contact_exclusion_delta=np.asarray([int(target_ssl_contact_exclusion_delta)], dtype=np.int64),
        llm_cache_path=np.asarray([str(cache_output_path if cache_output_path else llm_cache_path)], dtype="<U512"),
    )
    if target_contact_maps is not None:
        save_kwargs["target_contact_maps"] = np.asarray(target_contact_maps, dtype=object)
    if target_residue_embeddings is not None:
        residue_lengths = np.asarray([int(arr.shape[0]) for arr in target_residue_embeddings], dtype=np.int64)
        save_kwargs["target_residue_lengths"] = residue_lengths
    if drug_structure_node_embeddings is not None:
        save_kwargs["drug_structure_node_embeddings"] = np.asarray(drug_structure_node_embeddings, dtype=object)
    if target_structure_node_embeddings is not None:
        save_kwargs["target_structure_node_embeddings"] = np.asarray(target_structure_node_embeddings, dtype=object)
    if target_pagerank_graphs is not None:
        save_kwargs["target_pagerank_graphs"] = np.asarray(target_pagerank_graphs, dtype=object)

    np.savez_compressed(
        output_path,
        **save_kwargs,
    )

    print("=" * 68)
    print("LLM Node Representation Generated")
    print(f"Drug file              : {drug_path}")
    print(f"Target file            : {target_path}")
    print(f"Drug encoder model     : {drug_model_name}")
    print(f"Target encoder model   : {target_model_name}")
    print(f"Use static drug feats  : {use_static_drug}")
    print(f"Use static target feats: {use_static_target}")
    print(f"Use drug graph branch  : {use_drug_graph_branch}")
    print(f"Use target graph branch: {use_target_graph_branch}")
    print(f"Use entity graph AE    : {use_entity_graph_ae}")
    print(f"Graph hidden dim       : {entity_graph_ae_latent_dim}")
    print(f"Drug graph layers      : {drug_graph_steps}")
    print(f"Target graph layers    : {target_graph_steps}")
    print(f"Num drugs              : {len(drug_ids)}")
    print(f"Num targets            : {len(target_ids)}")
    print(f"Drug feature shape     : {drug_projected.shape}")
    print(f"Target feature shape   : {target_projected.shape}")
    print(f"Node feature shape     : {node_features.shape}")
    print(f"LLM cache path         : {cache_output_path}")
    print(f"Saved to               : {output_path}")
    print("=" * 68)


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate offline node representations for DTI using pretrained molecular/protein models. "
            "Outputs an npz compatible with the existing graph pipeline."
        )
    )
    parser.add_argument("--dataset", type=str, default="BindingDB")
    parser.add_argument("--data_dir", type=str, default="Data")
    parser.add_argument("--dti_lists_dir", type=str, default="Data/dti_lists")
    parser.add_argument("--drug_path", type=str, default="")
    parser.add_argument("--target_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--drug_model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--target_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--drug_batch_size", type=int, default=64)
    parser.add_argument("--target_batch_size", type=int, default=4)
    parser.add_argument(
        "--max_protein_length",
        type=int,
        default=1024,
        help="Protein sequence length cutoff used before ESM encoding. This is an engineering simplification for stable batching.",
    )
    parser.add_argument("--use_static_drug", action="store_true")
    parser.add_argument("--use_static_target", action="store_true")
    parser.add_argument(
        "--use_drug_graph_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable an offline structural drug graph branch from RDKit molecular graphs.",
    )
    parser.add_argument(
        "--use_target_graph_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable an offline structural target graph branch from residue embeddings.",
    )
    parser.add_argument("--drug_graph_steps", type=int, default=2)
    parser.add_argument("--target_graph_steps", type=int, default=2)
    parser.add_argument("--target_graph_topk", type=int, default=8)
    parser.add_argument(
        "--target_graph_mode",
        type=str,
        default="contact",
        choices=["approx", "contact"],
        help="How to construct target residue-graph edges: embedding-kNN approximation or contact map edges.",
    )
    parser.add_argument("--target_contact_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_protein_graph_length",
        type=int,
        default=512,
        help="Only the first residues up to this limit are used for offline residue-graph construction.",
    )
    parser.add_argument(
        "--use_entity_graph_ae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable offline entity-graph autoencoder branch and concatenate graph AE embeddings.",
    )
    parser.add_argument("--entity_graph_ae_latent_dim", type=int, default=64)
    parser.add_argument("--entity_graph_ae_epochs", type=int, default=50)
    parser.add_argument("--entity_graph_ae_lr", type=float, default=1e-3)
    parser.add_argument("--entity_graph_ae_weight_decay", type=float, default=0.0)
    parser.add_argument("--target_ssl_num_layers", type=int, default=2)
    parser.add_argument("--target_ssl_mask_ratio", type=float, default=0.15)
    parser.add_argument("--target_ssl_contact_dropout", type=float, default=0.15)
    parser.add_argument("--target_ssl_contrastive_lambda", type=float, default=0.1)
    parser.add_argument("--target_ssl_contact_exclusion_delta", type=int, default=2)
    parser.add_argument("--morgan_bits", type=int, default=1024)
    parser.add_argument("--morgan_radius", type=int, default=2)
    parser.add_argument("--target_tfidf_dim", type=int, default=128)
    parser.add_argument(
        "--llm_cache_path",
        type=str,
        default="",
        help="Optional path to precomputed drug/target LLM cache. When provided, skip live HF encoding.",
    )
    parser.add_argument(
        "--encode_order",
        type=str,
        default="target_first",
        choices=["drug_first", "target_first"],
        help="Whether to run drug encoding first or target encoding first.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    drug_path, target_path, dataset_base_dir = resolve_feature_paths(
        drug_path=args.drug_path,
        target_path=args.target_path,
        dataset=args.dataset,
        data_dir=args.data_dir,
        dti_lists_dir=args.dti_lists_dir,
    )
    output_path = args.output_path or build_default_output_path(
        dataset=args.dataset,
        base_dir=dataset_base_dir,
        use_drug_graph_branch=args.use_drug_graph_branch,
        use_target_graph_branch=args.use_target_graph_branch,
        use_entity_graph_ae=args.use_entity_graph_ae,
        use_static_drug=args.use_static_drug,
        use_static_target=args.use_static_target,
    )

    generate_node_representation_llm(
        drug_path=drug_path,
        target_path=target_path,
        output_path=output_path,
        output_dim=args.output_dim,
        seed=args.seed,
        drug_model_name=args.drug_model_name,
        target_model_name=args.target_model_name,
        drug_batch_size=args.drug_batch_size,
        target_batch_size=args.target_batch_size,
        device=args.device,
        max_protein_length=args.max_protein_length,
        use_static_drug=args.use_static_drug,
        use_static_target=args.use_static_target,
        use_drug_graph_branch=args.use_drug_graph_branch,
        use_target_graph_branch=args.use_target_graph_branch,
        drug_graph_steps=args.drug_graph_steps,
        target_graph_steps=args.target_graph_steps,
        target_graph_topk=args.target_graph_topk,
        target_graph_mode=args.target_graph_mode,
        target_contact_threshold=args.target_contact_threshold,
        max_protein_graph_length=args.max_protein_graph_length,
        use_entity_graph_ae=args.use_entity_graph_ae,
        entity_graph_ae_latent_dim=args.entity_graph_ae_latent_dim,
        entity_graph_ae_epochs=args.entity_graph_ae_epochs,
        entity_graph_ae_lr=args.entity_graph_ae_lr,
        entity_graph_ae_weight_decay=args.entity_graph_ae_weight_decay,
        target_ssl_num_layers=args.target_ssl_num_layers,
        target_ssl_mask_ratio=args.target_ssl_mask_ratio,
        target_ssl_contact_dropout=args.target_ssl_contact_dropout,
        target_ssl_contrastive_lambda=args.target_ssl_contrastive_lambda,
        target_ssl_contact_exclusion_delta=args.target_ssl_contact_exclusion_delta,
        morgan_bits=args.morgan_bits,
        morgan_radius=args.morgan_radius,
        target_tfidf_dim=args.target_tfidf_dim,
        encode_order=args.encode_order,
        llm_cache_path=args.llm_cache_path,
    )


if __name__ == "__main__":
    main()
