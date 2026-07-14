# WaveAS: Attribute Structure-Aware Multiscale Spectral Wavelet Propagation for Drug–Target Interaction Prediction


This directory keeps 4 core files:

- `main.py`: main entry
- `features.py`: drug/protein attribute feature generation
- `waveas_model.py`: model implementation
- `trainer.py`: training and evaluation

## Run
We use the Davis dataset as an example.

Feature generation only:

```bash
python.py ./features.py --dataset davis --output_path outputs/davis_paper_features.npz
```

Full training:

```bash
python.py ./main.py --dataset davis --file_path Data/dti_lists/davis/dti.csv --node_feature_path outputs/davis_paper_features.npz --output_dir outputs/davis_paper_run
```

If `--node_feature_path` is not provided, `main.py` will generate features automatically first.

Contact：

For questions or discussions, please feel free to open an issue or contact the authors.
