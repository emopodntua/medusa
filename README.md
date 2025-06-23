<div align="center">
  
# MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for Speech Emotion Recognition in Naturalistic Conditions

[![📄 Paper](https://img.shields.io/badge/Paper-arXiv%3A2506.09556-blue)](https://arxiv.org/abs/2506.09556)

[Georgios Chatzichristodoulou](https://scholar.google.com/citations?hl=en&user=QkMD1BkAAAAJ)\*, [Despoina Kosmopoulou](https://scholar.google.com/citations?user=roxd-tsAAAAJ)\*, [Antonios Kritikos](https://scholar.google.com/citations?user=Ra0Zwb0AAAAJ&hl=en)\*, Anastasia Poulopoulou, [Efthymios Georgiou](https://scholar.google.com/citations?hl=en&user=5Sc6GvEAAAAJ),<br />
[Athanasios Katsamanis](https://scholar.google.com/citations?user=S08iCqYAAAAJ&hl=en), [Vassilis Katsouros](https://scholar.google.com/citations?user=MCna7YcAAAAJ&hl=en), [Alexandros Potamianos](https://scholar.google.com/citations?user=pBQViyUAAAAJ&hl=en)<br />
🏫 [ECE, NTUA](http://ece.ntua.gr/), 🏫 [Archimedes, Athena RC](https://archimedesai.gr/en/), 🏫 [ILSP, Athena RC](https://www.ilsp.gr/en/home-2/)

*Equal contribution

</div>

**Abstract**: SER is a challenging task due to the subjective nature of human emotions and their uneven representation under naturalistic conditions. We propose MEDUSA, a multimodal framework with a four-stage training pipeline, which effectively handles class imbalance and emotion ambiguity. The first two stages train an ensemble of classifiers that utilize DeepSER, a novel extension of a deep cross-modal transformer fusion mechanism from pretrained self-supervised acoustic and linguistic representations. Manifold MixUp is employed for further regularization. The last two stages optimize a trainable meta-classifier that combines the ensemble predictions. Our training approach incorporates human annotation scores as soft targets, coupled with balanced data sampling and multitask learning. MEDUSA ranked 1<sup>st</sup> in Task 1: Categorical Emotion Recognition in the Interspeech 2025: Speech Emotion Recognition in Naturalistic Conditions Challenge.

# ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/emopodntua/medusa.git
cd medusa

# (Recommended) Create and activate a virtual environment
conda create -n virtual_env python=3.10 -y
conda activate virtual_env

# Install requirements
pip install -r requirements.txt
```

# 🗂️ Dataset
This code has been developed for the MSP-Podcast corpus. Access to the dataset requires a special permission. Therefore, we provide sample ```.csv``` files which show the format compatible with our code.

# 🧠 Training
Hyperparameters used in training are configured in a ```.yaml``` file.

## DeepSER
```bash
python3 train_deepser.py ./configs/deepser.yaml
```

## Meta-classifier
```bash
python3 train_metacls.py ./configs/metacls.yaml
```

# 📊 Evaluation
Hyperparameters used in evaluation are configured in a ```.yaml``` file.

## DeepSER
```bash
python3 eval_deepser.py ./configs/deepser.yaml
```

## Meta-classifier
```bash
python3 eval_metacls.py ./configs/metacls.yaml
```


# 📄 Citation

If you find this work useful for your research, please consider citing our paper:
```
@misc{chatzichristodoulou2025medusamultimodaldeepfusion,
      title={MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for Speech Emotion Recognition in Naturalistic Conditions}, 
      author={Georgios Chatzichristodoulou and Despoina Kosmopoulou and Antonios Kritikos and Anastasia Poulopoulou and Efthymios Georgiou and Athanasios Katsamanis and Vassilis Katsouros and Alexandros Potamianos},
      year={2025},
      eprint={2506.09556},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.09556
}
```

This code has been adapted from [here](https://github.com/msplabresearch/MSP-Podcast_Challenge_IS2025).
