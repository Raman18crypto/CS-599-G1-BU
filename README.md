# Beyond Likelihood: Evaluating Reliability in Deep Generative Models

## 📌 Overview

This project investigates the reliability of deep generative models for **out-of-distribution (OOD) detection**.

Recent research shows that likelihood-based methods (e.g., ELBO, log-likelihood) often fail to distinguish between in-distribution and OOD data. This repository reproduces and analyzes:

* Likelihood failure phenomenon
* Likelihood Regret (LR) for improved OOD detection
* Distribution-based metrics (Density & Coverage)
* Clipped Density and Clipped Coverage (robust evaluation)

---

## 🎯 Objectives

* Demonstrate why likelihood is unreliable for OOD detection
* Implement Likelihood Regret (LR)
* Evaluate generative models using geometric metrics
* Generate research-style plots and comparisons

---

## 🧠 Key Concepts

### Likelihood Failure

Generative models assign higher likelihood to OOD samples due to low-level statistical biases.

### Likelihood Regret (LR)

Measures how much likelihood improves when optimizing latent variables per sample.

### Clipped Metrics

Robust evaluation metrics that:

* Are bounded in [0,1]
* Resist outliers
* Provide interpretable scores

---

## 📊 Generated Figures

| Figure                | Description                         |
| --------------------- | ----------------------------------- |
| `likelihood_plot.png` | Likelihood distribution (ID vs OOD) |
| `lr_plot.png`         | Likelihood Regret separation        |
| `robustness_plot.png` | Clipped metrics robustness          |

## ⚙️ Installation

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

---

## 🚀 Running the Project

```bash
python main.py
```

This will:

* Train a VAE model
* Compute likelihood scores
* Compute Likelihood Regret
* Generate plots

---

## 📂 Output

After running, the following files will be generated:

```
likelihood_plot.png
lr_plot.png
robustness_plot.png
```

---

## 📈 Results Summary

| Metric            | Performance              |
| ----------------- | ------------------------ |
| Likelihood        | Fails (OOD often higher) |
| Likelihood Regret | Improved separation      |
| Clipped Density   | Robust                   |
| Clipped Coverage  | Interpretable            |

---

## 🧪 Dataset

Default setup uses:

* **MNIST** → In-distribution
* **FashionMNIST** → OOD

(Used instead of CIFAR due to download reliability)

---

## 📚 References

* Nalisnick et al., *Do Deep Generative Models Know What They Don’t Know?* (ICLR 2019)
* Xiao et al., *Likelihood Regret* (NeurIPS 2020)
* Salvy et al., *Clipped Density and Coverage* (ICLR 2026)
* Gulrajani et al., *PixelVAE* (ICLR 2017)

---

## ⚠️ Notes

* Likelihood Regret is computationally expensive (per-sample optimization)
* Some plots are illustrative for demonstrating expected behavior
* Results may vary depending on training duration

---

## 🚀 Future Work

* Replace VAE with Flow-based models (Glow)
* Use DINOv2 embeddings for feature extraction
* Reproduce full paper-level experiments
* Extend to real-world anomaly detection tasks

---

## 👤 Author

Raman Deep Shiva Murthy
MS Computer Science, Boston University

---

## ⭐ Acknowledgments

This project is inspired by recent research in generative model evaluation and OOD detection.

---

## 📜 License

MIT License
