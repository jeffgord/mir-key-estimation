# MIR - Key Estimation (Group 5)

## Setup

### Requirements

- Python 3.13.1: Download from [python.org](https://www.python.org/downloads/)

### Installation

Create and activate a virtual environment, then install requirements:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

- `demo.ipynb` - Demo notebook
- `analysis.ipynb` - Full analysis notebook for paper
- `utils.py` - Utility functions for demo and analysis
- `baseline_hpc` - Scripts for running the baseline methods on hpc in a parallelized fashion
- `chroma_transformer` - Feature extraction, training pipeline, train/test splits, saved predictions, and model weights for the Chroma Transformer model
- `data` - metadata used for analysis
- `saves` - saved predictions of the various models on the subset of data used in the demo notebook
