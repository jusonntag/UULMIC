# UULMIC — Unilateral Upper Limb Motor Imagery Classification

> **A learning project.** I built this EEG classification pipeline to practice hexagonal (ports & adapters) architecture and to learn building a small deep learning training pipeline end-to-end. The focus was on clean separation of concerns and understanding the patterns — not on production polish.

---

## What This Is

A modular pipeline for classifying unilateral upper limb motor imagery from EEG data, built around the code companion for [this paper](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1617748/full). It covers the full workflow: raw EEG → preprocessing (filtering, ICA, epoching) → model training → experiment tracking.

The architecture follows **hexagonal / ports & adapters** principles: domain models and abstract ports in the center, concrete adapters (MNE-Python, PyTorch, W&B) on the outside, wired together in a single CLI entry point.

📐 **[Architecture Deep Dive →](UULMIC_hexagonal_architecture.md)** — Full Mermaid diagrams, dependency graphs, port-adapter mapping, and a transferable template.

## Limitations

- **Only EEGNet** — FBCNet and NFEEG architectures not yet implemented
- **`train_test_split` leaks into the use case layer** — this is a hexagonal architecture violation (sklearn imported directly in the orchestration layer). In a real project, splitting logic should be abstracted behind a port or moved into the adapter.
- **Combined training mode is a TODO**
- **SklearnModelAdapter is incomplete** — missing `reset()` and `save()` implementations
- **No early stopping**

---

## TODO

- [x] Add TODO comments to empty stub files
- [ ] Fix `Any` import bug in `preprocess.py`
- [ ] Add `reset()`/`save()` to `SklearnModelAdapter`
- [ ] Fix `.gitignore` self-ignore + clean cached build artifacts
- [ ] Clean stale references in architecture doc
- [ ] Export architecture diagram as PNG for README
- [ ] Move `train_test_split` behind a port (or document the violation)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jusonntag/UULMIC.git
   cd UULMIC
   ```

2. **Install `uv` (if not already installed):**
   ```bash
   pip install uv
   ```
   or Homebrew:
   ```bash
   brew install uv
   ```

3. **Install dependencies and sync environment:**
   ```bash
   uv sync
   ```

4. **Weights & Biases (W&B) Tracking:**
   This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking. To enable it:
   - Create a free account at [wandb.ai](https://wandb.ai/)
   - Get your API token from your account settings
   - Create a `.env` file in the project root and add your token:
     ```env
     WANDB_API_KEY=your_api_token_here
     ```

5. **Data Setup:**
   The raw dataset is publicly available [here](https://pub.uni-bielefeld.de/record/3004681#) (please ensure you comply with the usage terms).
   - Download the required dataset.
   - Place the `.fif` and `.set` files for the participant(s) directly into the `data/raw/` directory.

## Usage

Run commands via `uv` to automatically handle dependencies and environments.

**Preprocess Data:**
```bash
uv run uulmic mode=preprocess
```

**Train Models:**
```bash
uv run uulmic
```

Override config parameters on the fly:
```bash
uv run uulmic model.batch_size=8
```

---

## Paper Reference

Based on:

> Sonntag J, Yu L, Wang X and Schack T (2025) *Neurophysiological predictors of deep learning based unilateral upper limb motor imagery classification.* Frontiers in Human Neuroscience, 19:1617748. [doi:10.3389/fnhum.2025.1617748](https://doi.org/10.3389/fnhum.2025.1617748)

```bibtex
@article{sonntag2025neurophysiological,
  title={Neurophysiological predictors of deep learning based unilateral upper limb motor imagery classification},
  author={Sonntag, J and Yu, L and Wang, X and Schack, T},
  journal={Frontiers in Human Neuroscience},
  volume={19},
  pages={1617748},
  year={2025},
  doi={10.3389/fnhum.2025.1617748}
}
```

---

## License

[MIT](LICENSE)
