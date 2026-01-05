# Project drona-GraphMamba



## Architecture

- **Graph Encoder**: GCN layers to encode spatial graph structure
- **Mamba SSM**: Efficient sequence modeling for route prediction
- **Vehicle Embedding**: Multi-agent learning with vehicle-specific context

## Free Stack ($0 Budget)

1. **Training**: Kaggle (Free T4 GPU) - Use `mamba-ssm` (CUDA)
2. **Storage**: Kaggle Datasets (Upload `.pt` files)
3. **Inference**: Your Laptop or Render.com (Free Tier) - Use `mamba_simple` (CPU)

## Quick Start

### 1. Training on Kaggle

1. Create a new Kaggle Notebook
2. Enable GPU (Settings → Accelerator → GPU T4 x2)
3. Upload the following files to your Kaggle notebook:
   - `newton_graphmamba.py`
   - `graph_utils.py`
   - `kaggle_training.ipynb` (or copy cells)
4. Run all cells sequentially
5. Download `newton_mamba_v1.pt` from Output section

**Important**: The first cell contains the "magic install block" that forces Mamba to install correctly on T4 GPUs. Run it FIRST!

### 2. CPU Inference API

On your laptop or Render.com:

```bash
# Install CPU-only dependencies
pip install -r requirements_cpu.txt

# Run inference API
python inference_api.py --model_path newton_mamba_v1.pt --graph_path city_graph.pt
```

The API will start on `http://localhost:5000`

### 3. API Usage

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Predict Next Node
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "route": [1234, 5678, 9012],
    "vehicle_id": 42,
    "top_k": 5
  }'
```

#### Predict Route Continuation
```bash
curl -X POST http://localhost:5000/predict_sequence \
  -H "Content-Type: application/json" \
  -d '{
    "route": [1234, 5678, 9012],
    "vehicle_id": 42,
    "length": 10
  }'
```

## Project Structure

```
drona/
├── kaggle_training.ipynb    # Training notebook for Kaggle
├── newton_graphmamba.py     # Main model implementation
├── graph_utils.py            # NetworkX → PyG conversion utilities
├── mamba_simple.py           # Pure PyTorch Mamba (CPU inference)
├── inference_api.py          # Flask API for inference
├── requirements.txt          # Full requirements (GPU training)
├── requirements_cpu.txt      # CPU-only requirements (inference)
└── README.md                 # This file
```

## Why This Setup Works

### Training on Kaggle
- **30 hours/week** of free GPU time
- Dual T4 GPUs (2x 15GB VRAM)
- Persistent storage
- The "magic install block" forces Mamba to compile correctly on T4 GPUs

### Inference on CPU
- Model weights (`.pt` file) are identical between GPU and CPU
- Only the math operation changes: CUDA kernel → Pure PyTorch
- `mamba_simple.py` provides CPU-compatible Mamba implementation
- Slower but works everywhere

## Troubleshooting

### Mamba Installation Fails on Kaggle
If the magic install block fails, try:
1. Use a different PyTorch version (check Kaggle's default)
2. Install `causal-conv1d` separately first
3. Use `--no-build-isolation` flag (already in the block)

### Model Too Large for CPU
- Reduce `d_model` or `n_layers` in training
- Use model quantization: `torch.quantization.quantize_dynamic()`

### API Memory Issues
- Load graph data separately (don't keep in memory)
- Use batch inference for multiple requests
- Consider using Render.com's free tier (512MB RAM)

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Kaggle Free GPU Guide](https://www.kaggle.com/docs/notebooks)
- [OSMnx Documentation](https://osmnx.readthedocs.io/)

## License

MIT License - Feel free to use for your projects!

