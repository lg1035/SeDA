# Seda

A PyTorch implementation of Few-shot Knowledge Graph Completion (FKGC) that combines BERT-based text representations with Graph Attention Networks (GAT) for improved few-shot learning on knowledge graphs.

## Features

- **BERT Integration**: Leverages pre-trained BERT models for entity and relation text representations
- **Graph Attention Networks**: Uses GAT layers to capture structural information in knowledge graphs
- **LoRA Support**: Implements Low-Rank Adaptation (LoRA) for efficient fine-tuning
- **Flexible Architecture**: Configurable GAT layers and attention mechanisms

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch Scatter (Optional but Recommended)

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Quick Start

### Basic Usage

```bash
python main_fkgc.py --data fb15k237-one --batch_size 32 --epoch 50 
```

### With Custom Parameters

```bash
python main_fkgc.py \
    --data fb15k-237 \
    --batch_size 16 \
    --epoch 100 \
    --sp_num 1 \
    --gat_heads 4 \
    --gat_layers 1 \
```

## Model Architecture

The FKGC model consists of:

1. **BERT Encoder**: Processes entity and relation text descriptions
2. **GAT Layers**: Captures structural relationships in the knowledge graph
3. **Relation Prototypes**: Learns relation-specific representations
4. **Scoring Functions**: Computes similarity scores between queries and candidates

### Debug Mode

```bash
python main_fkgc.py --overfit_debug --overfit_relations 1 --overfit_queries 5
```

## Project Structure

```
├── main_fkgc.py              # Main training script
├── fkgc_model.py             # FKGC model implementation
├── dataloader.py             # Data loading and preprocessing
├── lora_utils.py             # LoRA implementation utilities
└── data/                     # Dataset directory

```

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:
