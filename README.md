<div align="center">
  
# MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for Speech Emotion Recognition in Naturalistic Conditions


[Georgios Chatzichristodoulou*], [Despoina Kosmopoulou*], [Antonios Kritikos*], [Anastasia Pouloupoulou], [Efthymios Georgiou],<br />
[Athanasios Katsamanis], [Vassilis Katsouros], [Alexandros Potamianos]<br />
🏫 [ECE, NTUA](http://ece.ntua.gr/), 🏫 [Archimedes, Athena RC](https://archimedesai.gr/en/), 🏫 [ILSP, Athena RC](https://www.ilsp.gr/en/home-2/)

*Equal contribution

</div>

**Abstract**: SER is a challenging task due to the subjective nature of human emotions and their uneven representation under naturalistic conditions. We propose MEDUSA, a multimodal framework with a four-stage training pipeline, which effectively handles class imbalance and emotion ambiguity. The first two stages train an ensemble of classifiers that utilize DeepSER, a novel extension of a deep cross-modal transformer fusion mechanism from pretrained self-supervised acoustic and linguistic representations. Manifold MixUp is employed for further regularization. The last two stages optimize a trainable meta-classifier that combines the ensemble predictions. Our training approach incorporates human annotation scores as soft targets, coupled with balanced data sampling and multitask learning. MEDUSA ranked 1<sup>st</sup> in Task 1: Categorical Emotion Recognition in the Interspeech 2025: Speech Emotion Recognition in Naturalistic Conditions Challenge.
