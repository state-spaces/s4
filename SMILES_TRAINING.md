# SMILES Molecular Sequence Training

This repository has been configured to train S4 models on SMILES molecular sequences from PubChem.

## Dataset Configuration

The SMILES dataset is configured to use the following data files:
- **Train**: `/netscratch/hashmat/Repositories/Test/bert/simple_splits/pubchem_10m_train.txt`
- **Validation**: `/netscratch/hashmat/Repositories/Test/bert/simple_splits/pubchem_10m_val.txt`
- **Test**: `/netscratch/hashmat/Repositories/Test/bert/simple_splits/pubchem_10m_test.txt`

The dataset is configured in `configs/dataset/smiles.yaml`.

## Running Training

To train the S4 model on SMILES data, use:

```bash
python -m train experiment=smiles
```

## Configuration Files

The SMILES training setup consists of:

1. **Dataset Class**: `src/dataloaders/lm.py` - `SMILES` class
   - Inherits from WikiText2 for text-based sequence modeling
   - Character-level tokenization (no BPE)
   - Builds vocabulary from SMILES training data
   - Supports custom file paths

2. **Dataset Config**: `configs/dataset/smiles.yaml`
   - Specifies data file paths
   - Sets BPE to False for character-level encoding
   - Configures sequence length and other parameters

3. **Pipeline Config**: `configs/pipeline/smiles.yaml`
   - Uses language modeling trainer and loader
   - Configures AdamW optimizer and cosine warmup scheduler
   - Sets up language modeling task with perplexity metrics

4. **Experiment Config**: `configs/experiment/smiles.yaml`
   - Combines SMILES pipeline with S4 model
   - Easy entry point for training

## Customization

### Modifying Data Paths

Edit `configs/dataset/smiles.yaml` to change data file paths:

```yaml
train_path: "/path/to/your/train.txt"
val_path: "/path/to/your/val.txt"
test_path: "/path/to/your/test.txt"
```

### Modifying Model Parameters

Override parameters from the command line:

```bash
# Change number of layers
python -m train experiment=smiles model.n_layers=6

# Change batch size
python -m train experiment=smiles loader.batch_size=32

# Change sequence length
python -m train experiment=smiles loader.l_max=1024

# Multiple overrides
python -m train experiment=smiles model.n_layers=6 loader.batch_size=32 loader.l_max=1024
```

### WandB Logging

To disable WandB logging:
```bash
python -m train experiment=smiles wandb.mode=disabled
```

To set WandB project name:
```bash
python -m train experiment=smiles wandb.project=smiles-s4
```

## SMILES Data Format

The SMILES dataset expects text files with one SMILES string per line. The dataset will:
1. Build a character-level vocabulary from the training data
2. Encode sequences as token IDs
3. Cache the processed data for faster subsequent runs

## Model Architecture

The default configuration uses the S4 (Structured State Space) model, which is well-suited for long-range sequence modeling tasks like molecular property prediction and generation.

## Next Steps

After training, you can:
- Evaluate on test set: Add `train.test=True`
- Resume from checkpoint: `train.ckpt=/path/to/checkpoint.ckpt`
- Fine-tune with different learning rates or schedulers
