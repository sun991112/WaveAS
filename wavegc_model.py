import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NodeFeatureMixin:
    def _init_node_features(self, num_nodes, node_features, emb_dim, node_feature_mode):
        self.node_feature_mode = node_feature_mode
        if node_feature_mode == "learnable":
            self.node_emb = nn.Embedding(num_nodes, emb_dim)
        elif node_feature_mode == "random":
            if node_features is None:
                raise ValueError("node_features must be provided when node_feature_mode='random'")
            if node_features.size(0) != num_nodes:
                raise ValueError("node_features row count must equal num_nodes")
            if node_features.size(1) != emb_dim:
                raise ValueError("node_features column count must equal emb_dim")
            self.register_buffer("node_features", node_features.detach().clone())
        else:
            raise ValueError(f"Unsupported node_feature_mode: {node_feature_mode}")

    def _get_node_table(self):
        if self.node_feature_mode == "learnable":
            return self.node_emb.weight
        return self.node_features


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.constant = 100.0
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.eig_w.weight)
        nn.init.zeros_(self.eig_w.bias)

    def forward(self, eigvals):
        ee = eigvals * self.constant
        div = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=eigvals.device, dtype=eigvals.dtype)
            * (-torch.log(torch.tensor(10000.0, device=eigvals.device, dtype=eigvals.dtype)) / self.hidden_dim)
        )
        pe = ee.unsqueeze(-1) * div
        encoded = torch.cat([eigvals.unsqueeze(-1), torch.sin(pe), torch.cos(pe)], dim=-1)
        return self.eig_w(encoded)


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        return self.layer2(x)


class EigenEncoding(nn.Module):
    def __init__(self, hidden_dim, nheads=4, dropout=0.1):
        super().__init__()
        self.eig_encoder = SineEncoding(hidden_dim)
        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.eig_encoder.reset_parameters()
        self.ffn.reset_parameters()
        self.mha._reset_parameters()

    def forward(self, eigvals):
        eig = self.eig_encoder(eigvals.unsqueeze(0))
        mha_in = self.mha_norm(eig)
        mha_out, _ = self.mha(mha_in, mha_in, mha_in, need_weights=False)
        eig = eig + self.mha_dropout(mha_out)
        ffn_out = self.ffn(self.ffn_norm(eig))
        eig = eig + self.ffn_dropout(ffn_out)
        return eig.squeeze(0)


class IdentityEigenEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(1, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, eigvals):
        return self.proj(eigvals.unsqueeze(-1))


class WaveGCSpectralGenerator(nn.Module):
    def __init__(self, hidden_dim, num_n=5, num_J=3, pre_s=None, nheads=4, dropout=0.1, tight_frames=False, use_eigen_encoding=True):
        super().__init__()
        self.num_n = num_n
        self.num_J = num_J
        self.tight_frames = tight_frames
        if pre_s is None:
            pre_s = list(torch.linspace(1.0, 3.0, steps=max(1, num_J), dtype=torch.float32).tolist())
        if num_J > 0:
            self.pre_s = nn.Parameter(torch.tensor(pre_s, dtype=torch.float32))
        else:
            self.register_parameter("pre_s", None)
        self.register_buffer("scaling_base_coeffs", self._build_scaling_base(num_n))
        self.register_buffer("wavelet_base_coeffs", self._build_wavelet_base(max(1, num_J), num_n))
        self.decoder_scaling = nn.Linear(hidden_dim, num_n)
        self.decoder_wavelet = nn.Linear(hidden_dim, max(1, num_J * num_n))
        self.decoder_scales = nn.Linear(hidden_dim, max(1, num_J))
        self.ee = EigenEncoding(hidden_dim=hidden_dim, nheads=nheads, dropout=dropout) if use_eigen_encoding else IdentityEigenEncoding(hidden_dim=hidden_dim)
        self.wavelet_residual_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.scale_residual_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.reset_parameters()

    @staticmethod
    def _build_scaling_base(num_n):
        base = torch.linspace(2.0, 0.5, steps=num_n, dtype=torch.float32)
        return torch.softmax(base, dim=0)

    @staticmethod
    def _build_wavelet_base(num_J, num_n):
        positions = torch.arange(num_n, dtype=torch.float32)
        centers = torch.linspace(0, max(0, num_n - 1), steps=num_J)
        rows = []
        for center in centers:
            rows.append(torch.softmax(-((positions - center) ** 2) / 1.5, dim=0))
        return torch.stack(rows, dim=0)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.decoder_scaling.weight)
        nn.init.zeros_(self.decoder_scaling.bias)
        nn.init.xavier_uniform_(self.decoder_wavelet.weight)
        nn.init.zeros_(self.decoder_wavelet.bias)
        nn.init.xavier_uniform_(self.decoder_scales.weight)
        nn.init.zeros_(self.decoder_scales.bias)
        self.ee.reset_parameters()
        with torch.no_grad():
            self.wavelet_residual_scale.fill_(0.25)
            self.scale_residual_scale.fill_(0.35)
            if self.pre_s is not None:
                default_pre_s = torch.linspace(1.0, 3.0, steps=self.num_J, dtype=self.pre_s.dtype, device=self.pre_s.device)
                self.pre_s.copy_(default_pre_s)

    def gen_base(self, y):
        t_even = torch.ones_like(y)
        t_odd = y
        even_terms = [t_even.unsqueeze(0)]
        odd_terms = [t_odd.unsqueeze(0)]
        for _ in range(self.num_n - 1):
            t_even = 2 * y * t_odd - t_even
            t_odd = 2 * y * t_even - t_odd
            even_terms.append(t_even.unsqueeze(0))
            odd_terms.append(t_odd.unsqueeze(0))
        return torch.cat(even_terms, dim=0), torch.cat(odd_terms, dim=0)

    def forward(self, eigvals, return_aux=False):
        eigvals = eigvals.reshape(-1)
        eig_filter = self.ee(eigvals)

        scaling_base = self.scaling_base_coeffs.to(eigvals.device, eigvals.dtype)
        scaling_residual = self.decoder_scaling(eig_filter).mean(dim=0)
        scaling_logits = torch.log(scaling_base + 1e-8) + self.scale_residual_scale * scaling_residual
        coe_scaling = torch.softmax(scaling_logits, dim=-1)

        raw_wavelet = self.decoder_wavelet(eig_filter).mean(dim=0)
        if self.num_J > 0:
            wavelet_base = self.wavelet_base_coeffs[: self.num_J].to(eigvals.device, eigvals.dtype)
            raw_wavelet_logits = raw_wavelet.view(self.num_J, self.num_n)
            wavelet_logits = torch.log(wavelet_base + 1e-8) + self.wavelet_residual_scale * raw_wavelet_logits
            coe_wavelet = torch.softmax(wavelet_logits, dim=-1)
        else:
            wavelet_logits = raw_wavelet.new_zeros((0, self.num_n))
            coe_wavelet = raw_wavelet.new_zeros((0, self.num_n))

        raw_scales = self.decoder_scales(eig_filter).mean(dim=0)
        if self.num_J > 0:
            learned_pre_s = F.softplus(self.pre_s.to(eigvals.device, eigvals.dtype))
            coe_scales = learned_pre_s + self.scale_residual_scale * raw_scales
            coe_scales = torch.clamp(coe_scales, min=0.1, max=4.0)
        else:
            coe_scales = raw_scales.new_zeros((0,))

        _, base_scaling = self.gen_base(eigvals - 1.0)
        base_scaling = 0.5 * (-base_scaling.unsqueeze(-1) + 1.0)

        if self.num_J > 0:
            wavelet_signals = torch.clamp(eigvals.unsqueeze(-1) * coe_scales.unsqueeze(0), min=0.0, max=2.0)
            wavelet_inputs = wavelet_signals - 1.0
            base_wavelet, _ = self.gen_base(wavelet_inputs)
            base_wavelet = 0.5 * (-base_wavelet + 1.0)
        else:
            base_wavelet = eigvals.new_zeros((self.num_n, eigvals.numel(), 0))

        curr_scaling = (coe_scaling.view(self.num_n, 1, 1) * base_scaling).sum(dim=0)
        mixed_scaling = curr_scaling
        pre_gate_scaling = curr_scaling
        if self.num_J > 0:
            curr_wavelet = (coe_wavelet.view(self.num_J, self.num_n, 1) * base_wavelet.permute(2, 0, 1)).sum(dim=1).transpose(0, 1)
        else:
            curr_wavelet = curr_scaling.new_zeros((eigvals.numel(), 0))
        mixed_wavelet = curr_wavelet
        zero_mask = eigvals.abs() <= 1e-8
        if zero_mask.any() and curr_wavelet.numel() > 0:
            curr_wavelet = curr_wavelet.clone()
            curr_wavelet[zero_mask] = 0.0
        pre_gate_wavelet = curr_wavelet
        base_filters = torch.cat([curr_scaling, curr_wavelet], dim=-1)
        filter_signals = base_filters
        if self.tight_frames:
            filter_signals = filter_signals / (filter_signals.norm(dim=-1, keepdim=True) + 1e-8)
        if return_aux:
            return filter_signals, {
                "base_filters": base_filters,
                "scaling_logits": scaling_logits.detach(),
                "wavelet_logits": wavelet_logits.detach(),
                "scaling_coeffs": coe_scaling.detach(),
                "wavelet_coeffs": coe_wavelet.detach(),
                "wavelet_scales": coe_scales.detach(),
                "mixed_scaling": mixed_scaling.detach(),
                "mixed_wavelet": mixed_wavelet.detach(),
                "pre_gate_scaling": pre_gate_scaling.detach(),
                "pre_gate_wavelet": pre_gate_wavelet.detach(),
            }
        return filter_signals


class WaveGCSpectralBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dropout,
        num_scales,
        num_n=5,
        use_eigen_encoding=True,
        use_local_mpnn=False,
        mlp_domain="node",
        share_spectral_mlp=False,
        disable_post_filter_mlp=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_scales = num_scales
        self.use_local_mpnn = use_local_mpnn
        if mlp_domain not in {"node", "spectral"}:
            raise ValueError("mlp_domain must be one of: 'node', 'spectral'")
        self.mlp_domain = mlp_domain
        self.share_spectral_mlp = bool(share_spectral_mlp)
        self.disable_post_filter_mlp = bool(disable_post_filter_mlp)
        self.generator = WaveGCSpectralGenerator(
            hidden_dim=hidden_dim,
            num_n=num_n,
            num_J=max(0, num_scales - 1),
            nheads=4 if hidden_dim % 4 == 0 else 1,
            dropout=dropout,
            tight_frames=False,
            use_eigen_encoding=use_eigen_encoding,
        )
        self.local_model = GCNConv(hidden_dim, hidden_dim) if use_local_mpnn else None
        if self.mlp_domain == "spectral" and self.share_spectral_mlp:
            self.shared_spectral_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.scale_mlps = None
        else:
            self.shared_spectral_mlp = None
            self.scale_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(num_scales)
                ]
            )
        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm1_local = nn.LayerNorm(hidden_dim)
        self.norm1_attn = nn.LayerNorm(hidden_dim)
        self.ff_linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ff_linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self._eigvals = None
        self._eigvecs = None
        self._latest_filters = None
        self._latest_base_filters = None
        self._latest_scaling_logits = None
        self._latest_wavelet_logits = None
        self._latest_scaling_coeffs = None
        self._latest_wavelet_coeffs = None
        self._latest_wavelet_scales = None
        self._latest_mixed_scaling = None
        self._latest_mixed_wavelet = None
        self._latest_pre_gate_scaling = None
        self._latest_pre_gate_wavelet = None
        self._latest_input_spectral_energy = None
        self._latest_output_spectral_energy = None
        self._latest_branch_output_norms = None
        self.reset_parameters()

    def reset_parameters(self):
        self.generator.reset_parameters()
        if self.local_model is not None:
            self.local_model.reset_parameters()
        if self.shared_spectral_mlp is not None:
            for layer in self.shared_spectral_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        if self.scale_mlps is not None:
            for mlp in self.scale_mlps:
                for layer in mlp:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        nn.init.xavier_uniform_(self.ff_linear1.weight)
        nn.init.zeros_(self.ff_linear1.bias)
        nn.init.xavier_uniform_(self.ff_linear2.weight)
        nn.init.zeros_(self.ff_linear2.bias)

    def set_spectral_context(self, eigvals=None, eigvecs=None):
        self._eigvals = eigvals
        self._eigvecs = eigvecs

    def _build_filter_bank(self, device, dtype):
        if self._eigvals is None or self._eigvecs is None:
            raise RuntimeError("WaveGCSpectralBlock requires eigvals/eigvecs before forward.")
        eigvals = self._eigvals.to(device=device, dtype=dtype).reshape(-1)
        eigvecs = self._eigvecs.to(device=device, dtype=dtype)
        if eigvecs.dim() != 2 or eigvecs.size(1) != eigvals.numel():
            raise ValueError("eigvecs must have shape [num_nodes, num_eigs] matching eigvals.")
        filters, aux = self.generator(eigvals, return_aux=True)
        if filters.size(-1) != self.num_scales:
            raise ValueError("Generated spectral filters do not match num_scales.")
        stacked = filters.transpose(0, 1).contiguous()
        self._latest_filters = stacked
        self._latest_base_filters = aux["base_filters"].transpose(0, 1).contiguous()
        self._latest_scaling_logits = aux["scaling_logits"]
        self._latest_wavelet_logits = aux["wavelet_logits"]
        self._latest_scaling_coeffs = aux["scaling_coeffs"]
        self._latest_wavelet_coeffs = aux["wavelet_coeffs"]
        self._latest_wavelet_scales = aux["wavelet_scales"]
        self._latest_mixed_scaling = aux["mixed_scaling"]
        self._latest_mixed_wavelet = aux["mixed_wavelet"]
        self._latest_pre_gate_scaling = aux["pre_gate_scaling"]
        self._latest_pre_gate_wavelet = aux["pre_gate_wavelet"]
        return stacked, eigvecs

    @staticmethod
    def _apply_spectral_filter(x, spectral_filter, eigvecs):
        spectral_coeff = eigvecs.transpose(0, 1) @ x
        spectral_coeff = spectral_filter.unsqueeze(-1) * spectral_coeff
        return eigvecs @ spectral_coeff

    @staticmethod
    def _apply_spectral_filter_to_coeffs(spectral_coeff, spectral_filter):
        return spectral_filter.unsqueeze(-1) * spectral_coeff

    def diversity_regularization(self):
        if self._latest_filters is None or self._latest_filters.size(0) <= 1:
            return self.fusion.weight.new_zeros(())
        filters = self._latest_filters / (self._latest_filters.norm(dim=-1, keepdim=True) + 1e-8)
        penalty = filters.new_zeros(())
        count = 0
        for i in range(filters.size(0)):
            for j in range(i + 1, filters.size(0)):
                penalty = penalty + torch.sum(filters[i] * filters[j])
                count += 1
        return penalty / max(count, 1)

    def wavelet_energy_regularization(self):
        if self._latest_filters is None or self._latest_filters.size(0) <= 1:
            return self.fusion.weight.new_zeros(())
        return -self._latest_filters[1:].abs().mean()

    def wavelet_scaling_balance_regularization(self, target_ratio=0.25):
        if self._latest_filters is None or self._latest_filters.size(0) <= 1:
            return self.fusion.weight.new_zeros(())
        scaling_energy = self._latest_filters[0].abs().mean()
        wavelet_energy = self._latest_filters[1:].abs().mean()
        ratio = wavelet_energy / (scaling_energy + 1e-8)
        target = self.fusion.weight.new_tensor(float(target_ratio))
        return torch.relu(target - ratio)

    def _ff_block(self, x):
        x = self.ff_linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_linear2(x)
        return F.dropout(x, p=self.dropout, training=self.training)

    def forward(self, x, edge_index):
        h_in1 = x
        h_out_list = []
        if self.local_model is not None:
            h_local = self.local_model(x, edge_index)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local
            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        filters, eigvecs = self._build_filter_bank(device=x.device, dtype=x.dtype)
        spectral_coeff_input = eigvecs.transpose(0, 1) @ x
        self._latest_input_spectral_energy = spectral_coeff_input.norm(dim=-1).detach()

        kernel_outputs = []
        mlp_iterable = self.scale_mlps if self.scale_mlps is not None else [self.shared_spectral_mlp] * self.num_scales
        for mlp, spectral_filter in zip(mlp_iterable, filters):
            if self.mlp_domain == "spectral":
                filtered_coeff = self._apply_spectral_filter_to_coeffs(spectral_coeff_input, spectral_filter)
                spectral_h = filtered_coeff if self.disable_post_filter_mlp else mlp(filtered_coeff)
                spectral_h = self._apply_spectral_filter_to_coeffs(spectral_h, spectral_filter)
                h = eigvecs @ spectral_h
            else:
                scale_x = self._apply_spectral_filter(x, spectral_filter, eigvecs)
                h = scale_x if self.disable_post_filter_mlp else mlp(scale_x)
                h = self._apply_spectral_filter(h, spectral_filter, eigvecs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            kernel_outputs.append(h)
        self._latest_branch_output_norms = torch.stack([branch.norm(dim=-1) for branch in kernel_outputs], dim=0).detach()

        h = self.fusion(torch.cat(kernel_outputs, dim=-1))
        h_attn = self.dropout_attn(h)
        h_attn = h_in1 + h_attn
        h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)
        h = sum(h_out_list)
        out = h + self._ff_block(h)
        out = self.norm2(out)
        spectral_coeff_output = eigvecs.transpose(0, 1) @ out
        self._latest_output_spectral_energy = spectral_coeff_output.norm(dim=-1).detach()
        return out


class WaveGCSpectralLinkPredictor(nn.Module, NodeFeatureMixin):
    def __init__(
        self,
        num_nodes,
        node_features=None,
        emb_dim=128,
        hidden_dim=128,
        dropout=0.2,
        num_scales=4,
        num_layers=2,
        use_eigen_encoding=True,
        use_local_mpnn=False,
        mlp_domain="node",
        share_spectral_mlp=False,
        use_trainable_graph_branch=False,
        graph_branch_layers=2,
        graph_branch_dropout=0.1,
        graph_branch_encoder_mode="gcn",
        use_entity_graph_autoencoder=False,
        graph_branch_ae_latent_dim=64,
        node_feature_mode="random",
        disable_post_filter_mlp=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_trainable_graph_branch = bool(use_trainable_graph_branch)
        self.graph_branch_dropout = graph_branch_dropout
        encoder_mode = str(graph_branch_encoder_mode).strip().lower()
        if encoder_mode not in {"gcn", "ae"}:
            raise ValueError("graph_branch_encoder_mode must be one of {'gcn', 'ae'}.")
        self.graph_branch_encoder_mode = encoder_mode
        self.use_entity_graph_autoencoder = bool((use_entity_graph_autoencoder or self.graph_branch_encoder_mode == "ae") and self.use_trainable_graph_branch)
        self.graph_branch_ae_latent_dim = int(max(1, graph_branch_ae_latent_dim))
        self._latest_entity_graph_ae_loss = None
        self._cached_drug_ae_embedding = None
        self._cached_target_ae_embedding = None
        self._init_node_features(num_nodes, node_features, emb_dim, node_feature_mode)
        self._entity_num_drug = None
        self._entity_num_target = None
        self._drug_graph_x = None
        self._drug_graph_edge_index = None
        self._drug_graph_batch = None
        self._target_graph_x = None
        self._target_graph_edge_index = None
        self._target_graph_batch = None

        if self.use_trainable_graph_branch:
            del graph_branch_layers
            self.drug_graph_conv = GCNConv(emb_dim, emb_dim)
            self.target_graph_convs = nn.ModuleList([GCNConv(emb_dim, emb_dim), GCNConv(emb_dim, emb_dim), GCNConv(emb_dim, emb_dim)])
            num_heads = 4 if emb_dim % 4 == 0 else 1
            self.drug_graph_attn = nn.MultiheadAttention(emb_dim, num_heads=num_heads, dropout=graph_branch_dropout, batch_first=True)
            self.drug_graph_attn_norm = nn.LayerNorm(emb_dim)
            self.drug_node_proj = nn.LazyLinear(emb_dim)
            self.target_node_proj = nn.LazyLinear(emb_dim)
            self.drug_graph_fuse = nn.Linear(emb_dim * 2, emb_dim)
            if self.use_entity_graph_autoencoder:
                self.drug_ae_encoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Dropout(graph_branch_dropout), nn.Linear(emb_dim, self.graph_branch_ae_latent_dim))
                self.drug_ae_decoder = nn.Sequential(nn.Linear(self.graph_branch_ae_latent_dim, emb_dim), nn.ReLU(), nn.Dropout(graph_branch_dropout), nn.Linear(emb_dim, emb_dim))
                self.drug_ae_readout = nn.Linear(self.graph_branch_ae_latent_dim, emb_dim)
                self.target_ae_encoder = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Dropout(graph_branch_dropout), nn.Linear(emb_dim, self.graph_branch_ae_latent_dim))
                self.target_ae_decoder = nn.Sequential(nn.Linear(self.graph_branch_ae_latent_dim, emb_dim), nn.ReLU(), nn.Dropout(graph_branch_dropout), nn.Linear(emb_dim, emb_dim))
                self.target_ae_readout = nn.Linear(self.graph_branch_ae_latent_dim, emb_dim)
            else:
                self.drug_ae_encoder = None
                self.drug_ae_decoder = None
                self.drug_ae_readout = None
                self.target_ae_encoder = None
                self.target_ae_decoder = None
                self.target_ae_readout = None
        else:
            self.drug_graph_conv = None
            self.target_graph_convs = None
            self.drug_graph_attn = None
            self.drug_graph_attn_norm = None
            self.drug_node_proj = None
            self.target_node_proj = None
            self.drug_graph_fuse = None
            self.drug_ae_encoder = None
            self.drug_ae_decoder = None
            self.drug_ae_readout = None
            self.target_ae_encoder = None
            self.target_ae_decoder = None
            self.target_ae_readout = None

        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                WaveGCSpectralBlock(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    num_scales=num_scales,
                    use_eigen_encoding=use_eigen_encoding,
                    use_local_mpnn=use_local_mpnn,
                    mlp_domain=mlp_domain,
                    share_spectral_mlp=share_spectral_mlp,
                    disable_post_filter_mlp=disable_post_filter_mlp,
                )
                for _ in range(num_layers)
            ]
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.node_feature_mode == "learnable":
            nn.init.xavier_uniform_(self.node_emb.weight)
        if self.use_trainable_graph_branch:
            self.drug_graph_conv.reset_parameters()
            for conv in self.target_graph_convs:
                conv.reset_parameters()
            if hasattr(self.drug_node_proj, "reset_parameters"):
                self.drug_node_proj.reset_parameters()
            if hasattr(self.target_node_proj, "reset_parameters"):
                self.target_node_proj.reset_parameters()
            nn.init.xavier_uniform_(self.drug_graph_fuse.weight)
            nn.init.zeros_(self.drug_graph_fuse.bias)
            if self.use_entity_graph_autoencoder:
                for module in [
                    self.drug_ae_encoder,
                    self.drug_ae_decoder,
                    self.drug_ae_readout,
                    self.target_ae_encoder,
                    self.target_ae_decoder,
                    self.target_ae_readout,
                ]:
                    for layer in module.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight)
                            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for block in self.blocks:
            block.reset_parameters()
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def set_spectral_context(self, eigvals=None, eigvecs=None):
        for block in self.blocks:
            block.set_spectral_context(eigvals=eigvals, eigvecs=eigvecs)

    @staticmethod
    def _pack_entity_graphs(graphs):
        if graphs is None:
            return None, None, None
        xs = []
        edge_indices = []
        batches = []
        node_offset = 0
        for graph_idx, graph in enumerate(graphs):
            x = graph.get("x", None)
            edge_index = graph.get("edge_index", None)
            if x is None:
                x = torch.zeros((1, 1), dtype=torch.float32)
            x = x.detach().cpu().to(torch.float32)
            if x.ndim != 2 or x.size(0) == 0:
                x = torch.zeros((1, max(1, x.size(-1) if x.ndim == 2 else 1)), dtype=torch.float32)
            if edge_index is None:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_index = edge_index.detach().cpu().to(torch.long)
            if edge_index.ndim != 2 or edge_index.size(0) != 2:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            xs.append(x)
            if edge_index.numel() > 0:
                edge_indices.append(edge_index + node_offset)
            batches.append(torch.full((x.size(0),), graph_idx, dtype=torch.long))
            node_offset += x.size(0)
        if not xs:
            return None, None, None
        packed_x = torch.cat(xs, dim=0)
        packed_edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
        packed_batch = torch.cat(batches, dim=0)
        return packed_x, packed_edge_index, packed_batch

    def set_entity_graphs(self, num_drug, num_target, drug_graphs=None, target_graphs=None):
        self._entity_num_drug = int(num_drug)
        self._entity_num_target = int(num_target)
        self._drug_graph_x, self._drug_graph_edge_index, self._drug_graph_batch = self._pack_entity_graphs(drug_graphs)
        self._target_graph_x, self._target_graph_edge_index, self._target_graph_batch = self._pack_entity_graphs(target_graphs)
        self._cached_drug_ae_embedding = None
        self._cached_target_ae_embedding = None

    @staticmethod
    def _global_mean_pool(h, batch, num_graphs):
        if h is None or batch is None or num_graphs <= 0:
            return None
        out = h.new_zeros((num_graphs, h.size(-1)))
        counts = h.new_zeros((num_graphs, 1))
        out.index_add_(0, batch, h)
        ones = h.new_ones((h.size(0), 1))
        counts.index_add_(0, batch, ones)
        return out / counts.clamp_min(1.0)

    @staticmethod
    def _apply_graph_self_attention(h, batch, attn_layer, norm_layer, num_graphs):
        if h is None or batch is None or attn_layer is None or norm_layer is None:
            return h
        out = h.clone()
        for graph_idx in range(num_graphs):
            node_idx = (batch == graph_idx).nonzero(as_tuple=False).view(-1)
            if node_idx.numel() <= 1:
                continue
            seq = out[node_idx].unsqueeze(0)
            attn_seq, _ = attn_layer(seq, seq, seq, need_weights=False)
            seq = norm_layer(seq + attn_seq)
            out[node_idx] = seq.squeeze(0)
        return out

    def _encode_packed_graphs(self, packed_x, packed_edge_index, packed_batch, node_proj, convs, attn_layer, attn_norm, num_graphs, device):
        if packed_x is None or packed_batch is None or num_graphs <= 0:
            return None
        x = packed_x.to(device)
        edge_index = packed_edge_index.to(device) if packed_edge_index is not None else None
        batch = packed_batch.to(device)
        h = node_proj(x)
        for conv in convs:
            h = conv(h, edge_index) if edge_index is not None else h
            h = F.relu(h)
            h = F.dropout(h, p=self.graph_branch_dropout, training=self.training)
        h = self._apply_graph_self_attention(h, batch, attn_layer=attn_layer, norm_layer=attn_norm, num_graphs=num_graphs)
        return self._global_mean_pool(h, batch, num_graphs)

    def _encode_packed_graphs_autoencoder(self, packed_x, packed_batch, node_proj, encoder, decoder, readout, num_graphs, device, compute_loss=True):
        if packed_x is None or packed_batch is None or num_graphs <= 0 or node_proj is None or encoder is None or decoder is None or readout is None:
            return None, None
        x = packed_x.to(device)
        batch = packed_batch.to(device)
        h = node_proj(x)
        latent = encoder(h)
        recon_loss = None
        if compute_loss:
            recon = decoder(latent)
            recon_loss = F.mse_loss(recon, h)
        graph_latent = self._global_mean_pool(latent, batch, num_graphs)
        if graph_latent is None:
            return None, recon_loss
        return readout(graph_latent), recon_loss

    def _materialize_ae_lazy_layers(self, device):
        with torch.no_grad():
            if self._drug_graph_x is not None and self._drug_graph_batch is not None:
                self._encode_packed_graphs_autoencoder(self._drug_graph_x, self._drug_graph_batch, self.drug_node_proj, self.drug_ae_encoder, self.drug_ae_decoder, self.drug_ae_readout, self._entity_num_drug, device, False)
            if self._target_graph_x is not None and self._target_graph_batch is not None:
                self._encode_packed_graphs_autoencoder(self._target_graph_x, self._target_graph_batch, self.target_node_proj, self.target_ae_encoder, self.target_ae_decoder, self.target_ae_readout, self._entity_num_target, device, False)

    def refresh_entity_graph_autoencoder_cache(self, device):
        if not self.use_trainable_graph_branch or self.graph_branch_encoder_mode != "ae":
            return
        if self._entity_num_drug is None or self._entity_num_target is None:
            raise RuntimeError("Call set_entity_graphs(...) before refreshing AE cache.")
        self.eval()
        with torch.no_grad():
            drug_h, _ = self._encode_packed_graphs_autoencoder(self._drug_graph_x, self._drug_graph_batch, self.drug_node_proj, self.drug_ae_encoder, self.drug_ae_decoder, self.drug_ae_readout, self._entity_num_drug, device, False)
            target_h, _ = self._encode_packed_graphs_autoencoder(self._target_graph_x, self._target_graph_batch, self.target_node_proj, self.target_ae_encoder, self.target_ae_decoder, self.target_ae_readout, self._entity_num_target, device, False)
        self._cached_drug_ae_embedding = None if drug_h is None else drug_h.detach().cpu()
        self._cached_target_ae_embedding = None if target_h is None else target_h.detach().cpu()

    def pretrain_entity_graph_autoencoder(self, device, epochs=50, lr=1e-3, weight_decay=0.0, verbose=False):
        if not self.use_trainable_graph_branch or self.graph_branch_encoder_mode != "ae":
            return
        if self._entity_num_drug is None or self._entity_num_target is None:
            raise RuntimeError("Call set_entity_graphs(...) before AE pretraining.")
        epochs = int(max(1, epochs))
        self._materialize_ae_lazy_layers(device=device)
        params = []
        for module in [self.drug_node_proj, self.target_node_proj, self.drug_ae_encoder, self.drug_ae_decoder, self.drug_ae_readout, self.target_ae_encoder, self.target_ae_decoder, self.target_ae_readout]:
            if module is not None:
                params.extend(list(module.parameters()))
        if not params:
            return
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            total_loss = None
            _, drug_loss = self._encode_packed_graphs_autoencoder(self._drug_graph_x, self._drug_graph_batch, self.drug_node_proj, self.drug_ae_encoder, self.drug_ae_decoder, self.drug_ae_readout, self._entity_num_drug, device, True)
            _, target_loss = self._encode_packed_graphs_autoencoder(self._target_graph_x, self._target_graph_batch, self.target_node_proj, self.target_ae_encoder, self.target_ae_decoder, self.target_ae_readout, self._entity_num_target, device, True)
            if drug_loss is not None:
                total_loss = drug_loss if total_loss is None else total_loss + drug_loss
            if target_loss is not None:
                total_loss = target_loss if total_loss is None else total_loss + target_loss
            if total_loss is None:
                break
            total_loss.backward()
            optimizer.step()
            if verbose and (epoch == 1 or epoch == epochs):
                print(f"[Entity AE pretrain] Epoch [{epoch:03d}/{epochs}] ReconLoss={total_loss.item():.6f}")
        self.refresh_entity_graph_autoencoder_cache(device=device)

    def _encode_entity_graph_branch(self, x):
        if not self.use_trainable_graph_branch:
            return x
        if self._entity_num_drug is None or self._entity_num_target is None:
            raise RuntimeError("Entity graph branch enabled but entity sizes were not set. Call set_entity_graphs(...).")
        drug_x = x[: self._entity_num_drug]
        target_x = x[self._entity_num_drug : self._entity_num_drug + self._entity_num_target]
        device = x.device
        self._latest_entity_graph_ae_loss = x.new_zeros(())
        drug_h = None
        if self.graph_branch_encoder_mode == "gcn":
            drug_h = self._encode_packed_graphs(self._drug_graph_x, self._drug_graph_edge_index, self._drug_graph_batch, self.drug_node_proj, [self.drug_graph_conv], self.drug_graph_attn, self.drug_graph_attn_norm, self._entity_num_drug, device)
        elif self.graph_branch_encoder_mode == "ae" and self.use_entity_graph_autoencoder:
            if self._cached_drug_ae_embedding is None:
                raise RuntimeError("AE mode requires precomputed entity AE cache. Call pretrain_entity_graph_autoencoder(...).")
            drug_h = self._cached_drug_ae_embedding.to(device)
        if drug_h is not None and drug_h.shape[0] == drug_x.shape[0]:
            drug_gate = torch.sigmoid(self.drug_graph_fuse(torch.cat([drug_x, drug_h], dim=-1)))
            drug_x = drug_gate * drug_h + (1.0 - drug_gate) * drug_x

        target_h = None
        if self.graph_branch_encoder_mode == "gcn":
            target_h = self._encode_packed_graphs(self._target_graph_x, self._target_graph_edge_index, self._target_graph_batch, self.target_node_proj, self.target_graph_convs, None, None, self._entity_num_target, device)
        elif self.graph_branch_encoder_mode == "ae" and self.use_entity_graph_autoencoder:
            if self._cached_target_ae_embedding is None:
                raise RuntimeError("AE mode requires precomputed entity AE cache. Call pretrain_entity_graph_autoencoder(...).")
            target_h = self._cached_target_ae_embedding.to(device)
        if target_h is not None and target_h.shape[0] == target_x.shape[0]:
            target_x = target_h
        return torch.cat([drug_x, target_x], dim=0)

    def entity_graph_autoencoder_regularization(self):
        return self.input_proj.weight.new_zeros(())

    def diversity_regularization(self):
        penalties = [block.diversity_regularization() for block in self.blocks]
        return torch.stack(penalties).mean() if penalties else self.input_proj.weight.new_zeros(())

    def wavelet_energy_regularization(self):
        penalties = [block.wavelet_energy_regularization() for block in self.blocks]
        return torch.stack(penalties).mean() if penalties else self.input_proj.weight.new_zeros(())

    def wavelet_scaling_balance_regularization(self, target_ratio=0.25):
        penalties = [block.wavelet_scaling_balance_regularization(target_ratio=target_ratio) for block in self.blocks]
        return torch.stack(penalties).mean() if penalties else self.input_proj.weight.new_zeros(())

    def get_visualization_state(self, block_idx=0):
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise IndexError(f"block_idx out of range: {block_idx}")
        block = self.blocks[block_idx]
        if block._eigvals is None or block._latest_filters is None:
            raise RuntimeError("Visualization state is unavailable before a forward pass with spectral context.")
        eigvals = block._eigvals.detach().cpu()
        max_eig = torch.clamp(eigvals.max(), min=eigvals.new_tensor(1e-6))
        lambda_hat = eigvals / max_eig
        continuous_lambda = torch.linspace(0.0, 2.0, steps=400, device=block._eigvals.device, dtype=block._eigvals.dtype)
        continuous_filters, _ = block.generator(continuous_lambda, return_aux=True)
        return {
            "eigvals": eigvals.numpy(),
            "lambda_hat": lambda_hat.numpy(),
            "filters": block._latest_filters.detach().cpu().numpy(),
            "continuous_lambda": continuous_lambda.detach().cpu().numpy(),
            "continuous_filters": continuous_filters.transpose(0, 1).detach().cpu().numpy(),
            "base_filters": block._latest_base_filters.detach().cpu().numpy() if block._latest_base_filters is not None else None,
            "scaling_logits": block._latest_scaling_logits.detach().cpu().numpy() if block._latest_scaling_logits is not None else None,
            "wavelet_logits": block._latest_wavelet_logits.detach().cpu().numpy() if block._latest_wavelet_logits is not None else None,
            "scaling_coeffs": block._latest_scaling_coeffs.detach().cpu().numpy() if block._latest_scaling_coeffs is not None else None,
            "wavelet_coeffs": block._latest_wavelet_coeffs.detach().cpu().numpy() if block._latest_wavelet_coeffs is not None else None,
            "wavelet_scales": block._latest_wavelet_scales.detach().cpu().numpy() if block._latest_wavelet_scales is not None else None,
            "learned_pre_s": F.softplus(block.generator.pre_s.detach().cpu()).numpy() if block.generator.pre_s is not None else None,
            "mixed_scaling": block._latest_mixed_scaling.detach().cpu().numpy().reshape(-1) if block._latest_mixed_scaling is not None else None,
            "mixed_wavelet": block._latest_mixed_wavelet.detach().cpu().numpy().T if block._latest_mixed_wavelet is not None else None,
            "pre_gate_scaling": block._latest_pre_gate_scaling.detach().cpu().numpy().reshape(-1) if block._latest_pre_gate_scaling is not None else None,
            "pre_gate_wavelet": block._latest_pre_gate_wavelet.detach().cpu().numpy().T if block._latest_pre_gate_wavelet is not None else None,
            "input_spectral_energy": block._latest_input_spectral_energy.detach().cpu().numpy() if block._latest_input_spectral_energy is not None else None,
            "output_spectral_energy": block._latest_output_spectral_energy.detach().cpu().numpy() if block._latest_output_spectral_energy is not None else None,
            "branch_output_norms": block._latest_branch_output_norms.detach().cpu().numpy() if block._latest_branch_output_norms is not None else None,
            "wavelet_centers": None,
            "wavelet_widths": None,
        }

    def encode(self, edge_index):
        x = self._get_node_table()
        x = self._encode_entity_graph_branch(x)
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for block in self.blocks:
            x = block(x, edge_index)
        return x

    def decode(self, z, edge_pairs, num_drug, device):
        d_idx = torch.tensor([edge[0] for edge in edge_pairs], dtype=torch.long, device=device)
        t_idx = torch.tensor([edge[1] + num_drug for edge in edge_pairs], dtype=torch.long, device=device)
        zd = z[d_idx]
        zt = z[t_idx]
        return self.edge_mlp(torch.cat([zd, zt], dim=1)).squeeze(1)

    def forward(self, edge_index, edge_pairs, num_drug, device):
        z = self.encode(edge_index)
        return self.decode(z, edge_pairs, num_drug, device)


MODEL_REGISTRY = {
    "wavegc_spectral": WaveGCSpectralLinkPredictor,
}


def build_model(
    model_name,
    num_nodes,
    node_features,
    emb_dim,
    hidden_dim,
    dropout,
    node_feature_mode,
    num_scales=4,
    num_layers=2,
    use_eigen_encoding=True,
    use_local_mpnn=False,
    use_trainable_graph_branch=False,
    graph_branch_layers=2,
    graph_branch_dropout=0.1,
    graph_branch_encoder_mode="gcn",
    use_entity_graph_autoencoder=False,
    graph_branch_ae_latent_dim=64,
    disable_post_filter_mlp=False,
):
    if model_name != "wavegc_spectral":
        raise ValueError("Only 'wavegc_spectral' is kept in the final paper-aligned version.")
    return WaveGCSpectralLinkPredictor(
        num_nodes=num_nodes,
        node_features=node_features,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_scales=num_scales,
        num_layers=num_layers,
        use_eigen_encoding=use_eigen_encoding,
        use_local_mpnn=use_local_mpnn,
        use_trainable_graph_branch=use_trainable_graph_branch,
        graph_branch_layers=graph_branch_layers,
        graph_branch_dropout=graph_branch_dropout,
        graph_branch_encoder_mode=graph_branch_encoder_mode,
        use_entity_graph_autoencoder=use_entity_graph_autoencoder,
        graph_branch_ae_latent_dim=graph_branch_ae_latent_dim,
        node_feature_mode=node_feature_mode,
        disable_post_filter_mlp=disable_post_filter_mlp,
    )
