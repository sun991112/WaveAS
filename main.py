"""
Main training entry for the self-contained paper-aligned pipeline.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from features import generate_node_representation_llm
from trainer import (
    align_precomputed_features_to_dti,
    build_drug_dissimilarity_neighbors_from_features,
    build_similarity_edges_from_features,
    load_positive_edges_flexible,
    load_precomputed_node_feature_bundle,
    make_kfold_edge_splits_gmjrl_style,
    train_one_fold,
)


def run_paper_pipeline(
    dataset,
    file_path,
    output_dir,
    node_feature_path="",
    seed=42,
    device="cpu",
    output_dim=128,
    drug_model_name="seyonec/ChemBERTa-zinc-base-v1",
    target_model_name="facebook/esm2_t33_650M_UR50D",
    drug_batch_size=64,
    target_batch_size=4,
    max_protein_length=1024,
    drug_graph_layers=2,
    target_graph_layers=2,
    target_contact_threshold=0.5,
    graph_hidden_dim=64,
    graph_epochs=50,
    graph_lr=1e-3,
    graph_weight_decay=0.0,
    hidden_dim=128,
    dropout=0.2,
    num_scales=4,
    num_layers=2,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    n_splits=5,
    threshold=0.5,
    drug_sim_topk=10,
    drug_sim_threshold=0.3,
    target_sim_topk=10,
    target_sim_threshold=0.3,
    gmjrl_neg_topk=10,
    spectral_topk=16,
    wavegc_diversity_lambda=1e-2,
    wavegc_wavelet_energy_lambda=1e-2,
    wavegc_wavelet_balance_lambda=1e-1,
    wavegc_wavelet_target_ratio=0.25,
    llm_cache_path="",
):
    os.makedirs(output_dir, exist_ok=True)
    if not node_feature_path:
        node_feature_path = os.path.join(output_dir, f"{dataset}_paper_node_features.npz")
        dataset_dir = os.path.join("Data", dataset)
        if not os.path.isdir(dataset_dir):
            dataset_dir = os.path.join("Data", "dti_lists", dataset)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset}")

        drug_path = ""
        target_path = ""
        for candidate in ["drugs.xlsx", "drugs.xls", "drug.csv"]:
            path = os.path.join(dataset_dir, candidate)
            if os.path.exists(path):
                drug_path = path
                break
        for candidate in ["targets.xlsx", "targets.xls", "target.csv"]:
            path = os.path.join(dataset_dir, candidate)
            if os.path.exists(path):
                target_path = path
                break
        if not drug_path or not target_path:
            raise FileNotFoundError(f"drugs/targets file missing under {dataset_dir}")

        generate_node_representation_llm(
            drug_path=drug_path,
            target_path=target_path,
            output_path=node_feature_path,
            output_dim=output_dim,
            seed=seed,
            drug_model_name=drug_model_name,
            target_model_name=target_model_name,
            drug_batch_size=drug_batch_size,
            target_batch_size=target_batch_size,
            device=device,
            max_protein_length=max_protein_length,
            use_static_drug=False,
            use_static_target=False,
            use_drug_graph_branch=True,
            use_target_graph_branch=True,
            drug_graph_steps=drug_graph_layers,
            target_graph_steps=target_graph_layers,
            target_graph_topk=8,
            target_graph_mode="contact",
            target_contact_threshold=target_contact_threshold,
            max_protein_graph_length=max_protein_length,
            use_entity_graph_ae=False,
            entity_graph_ae_latent_dim=graph_hidden_dim,
            entity_graph_ae_epochs=graph_epochs,
            entity_graph_ae_lr=graph_lr,
            entity_graph_ae_weight_decay=graph_weight_decay,
            target_ssl_num_layers=2,
            target_ssl_mask_ratio=0.15,
            target_ssl_contact_dropout=0.15,
            target_ssl_contrastive_lambda=0.1,
            target_ssl_contact_exclusion_delta=2,
            morgan_bits=1024,
            morgan_radius=2,
            target_tfidf_dim=128,
            encode_order="target_first",
            llm_cache_path=llm_cache_path,
        )

    pos_edges, drug_id_map, target_id_map, num_drug, num_target, _ = load_positive_edges_flexible(file_path)
    feature_bundle = load_precomputed_node_feature_bundle(node_feature_path)
    aligned_drug, aligned_target, node_features = align_precomputed_features_to_dti(
        drug_id_map=drug_id_map,
        target_id_map=target_id_map,
        drug_features=feature_bundle["drug_features"].astype(np.float32),
        target_features=feature_bundle["target_features"].astype(np.float32),
        drug_ids=feature_bundle["drug_ids"],
        target_ids=feature_bundle["target_ids"],
    )

    drug_sim_edges = build_similarity_edges_from_features(
        aligned_drug,
        topk=drug_sim_topk,
        threshold=drug_sim_threshold,
    )
    target_sim_edges = build_similarity_edges_from_features(
        aligned_target,
        topk=target_sim_topk,
        threshold=target_sim_threshold,
    )
    drug_dissimmat = build_drug_dissimilarity_neighbors_from_features(
        aligned_drug,
        topk=gmjrl_neg_topk,
    )
    folds = make_kfold_edge_splits_gmjrl_style(
        pos_edges=pos_edges,
        num_drug=num_drug,
        num_target=num_target,
        drug_dissimmat=drug_dissimmat,
        n_splits=n_splits,
        seed=seed,
    )

    fold_results = []
    for fold_idx, (train_pos_edges, test_pos_edges, train_neg_edges, test_neg_edges) in enumerate(folds, start=1):
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        result = train_one_fold(
            train_pos_edges=train_pos_edges,
            test_pos_edges=test_pos_edges,
            train_neg_edges=train_neg_edges,
            test_neg_edges=test_neg_edges,
            num_drug=num_drug,
            num_target=num_target,
            node_features=node_features,
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
            batch_size=None,
            shuffle=True,
            verbose=True,
            threshold=threshold,
            model_name="wavegc_spectral",
            selection_metric="aupr",
            add_self_loops=True,
            use_spectral_context=True,
            spectral_topk=spectral_topk,
            wavegc_diversity_lambda=wavegc_diversity_lambda,
            wavegc_wavelet_energy_lambda=wavegc_wavelet_energy_lambda,
            wavegc_wavelet_balance_lambda=wavegc_wavelet_balance_lambda,
            wavegc_wavelet_target_ratio=wavegc_wavelet_target_ratio,
            use_eigen_encoding=True,
            use_local_mpnn=False,
            use_trainable_graph_branch=False,
            graph_branch_layers=2,
            graph_branch_dropout=0.1,
            disable_post_filter_mlp=False,
            graph_branch_drug_graphs=None,
            graph_branch_target_graphs=None,
            visualization_output_dir=fold_output_dir,
            visualize_wavegc_spectral=False,
        )
        result["fold"] = fold_idx
        fold_results.append(result)

    summary = {
        "dataset": dataset,
        "file_path": file_path,
        "node_feature_path": node_feature_path,
        "mean_auc": float(np.nanmean([item["auc"] for item in fold_results])),
        "mean_aupr": float(np.nanmean([item["aupr"] for item in fold_results])),
        "mean_f1": float(np.nanmean([item["f1"] for item in fold_results])),
        "fold_results": fold_results,
    }
    pd.DataFrame(fold_results).to_csv(os.path.join(output_dir, "paper_cv_results.csv"), index=False)
    with open(os.path.join(output_dir, "paper_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_argparser():
    parser = argparse.ArgumentParser(description="Run the paper-aligned DTI pipeline.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--node_feature_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--drug_model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--target_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--drug_batch_size", type=int, default=64)
    parser.add_argument("--target_batch_size", type=int, default=4)
    parser.add_argument("--max_protein_length", type=int, default=1024)
    parser.add_argument("--drug_graph_layers", type=int, default=2)
    parser.add_argument("--target_graph_layers", type=int, default=2)
    parser.add_argument("--target_contact_threshold", type=float, default=0.5)
    parser.add_argument("--graph_hidden_dim", type=int, default=64)
    parser.add_argument("--graph_epochs", type=int, default=50)
    parser.add_argument("--graph_lr", type=float, default=1e-3)
    parser.add_argument("--graph_weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_scales", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--drug_sim_topk", type=int, default=10)
    parser.add_argument("--drug_sim_threshold", type=float, default=0.3)
    parser.add_argument("--target_sim_topk", type=int, default=10)
    parser.add_argument("--target_sim_threshold", type=float, default=0.3)
    parser.add_argument("--gmjrl_neg_topk", type=int, default=10)
    parser.add_argument("--spectral_topk", type=int, default=16)
    parser.add_argument("--wavegc_diversity_lambda", type=float, default=1e-2)
    parser.add_argument("--wavegc_wavelet_energy_lambda", type=float, default=1e-2)
    parser.add_argument("--wavegc_wavelet_balance_lambda", type=float, default=1e-1)
    parser.add_argument("--wavegc_wavelet_target_ratio", type=float, default=0.25)
    parser.add_argument("--llm_cache_path", type=str, default="")
    return parser


def main():
    args = build_argparser().parse_args()
    summary = run_paper_pipeline(
        dataset=args.dataset,
        file_path=args.file_path,
        output_dir=args.output_dir,
        node_feature_path=args.node_feature_path,
        seed=args.seed,
        device=args.device,
        output_dim=args.output_dim,
        drug_model_name=args.drug_model_name,
        target_model_name=args.target_model_name,
        drug_batch_size=args.drug_batch_size,
        target_batch_size=args.target_batch_size,
        max_protein_length=args.max_protein_length,
        drug_graph_layers=args.drug_graph_layers,
        target_graph_layers=args.target_graph_layers,
        target_contact_threshold=args.target_contact_threshold,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_epochs=args.graph_epochs,
        graph_lr=args.graph_lr,
        graph_weight_decay=args.graph_weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_scales=args.num_scales,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_splits=args.n_splits,
        threshold=args.threshold,
        drug_sim_topk=args.drug_sim_topk,
        drug_sim_threshold=args.drug_sim_threshold,
        target_sim_topk=args.target_sim_topk,
        target_sim_threshold=args.target_sim_threshold,
        gmjrl_neg_topk=args.gmjrl_neg_topk,
        spectral_topk=args.spectral_topk,
        wavegc_diversity_lambda=args.wavegc_diversity_lambda,
        wavegc_wavelet_energy_lambda=args.wavegc_wavelet_energy_lambda,
        wavegc_wavelet_balance_lambda=args.wavegc_wavelet_balance_lambda,
        wavegc_wavelet_target_ratio=args.wavegc_wavelet_target_ratio,
        llm_cache_path=args.llm_cache_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
