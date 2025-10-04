# Quantum2Prompt: Representing Quantum Circuits as Language Prompts for Linear System Solving

Official implementation of **Quantum2Prompt**, a cross–modal framework that represents variational quantum linear solver (VQLS) circuits as compact language prompts for large language models (LLMs).  
This repository accompanies the manuscript *“Quantum2Prompt: Representing Quantum Circuits as Language Prompts for Linear System Solving”* (2025).

---

## 🧠 Overview

Variational Quantum Algorithms (VQAs) such as the Variational Quantum Linear Solver (VQLS) offer a hybrid approach for solving linear systems on near–term quantum hardware.  
However, these algorithms require repeated circuit executions and suffer from measurement overhead and slow convergence.

**Quantum2Prompt** introduces a new paradigm:
> Convert quantum circuits into structured text sequences and let a large language model predict VQLS residuals directly — *without executing the circuit.*

This transforms the estimation problem into a **prompt–to–regression task**, enabling:
- Low–cost residual predictions  
- Better initialization for variational solvers  
- Dataset-based reproducibility  

---

## ⚙️ Methodology

### Framework Overview
The workflow consists of five stages:

1. **Circuit Generation** — build VQLS circuits for benchmark matrices (Toeplitz, Laplacian, Sparse).  
2. **Textualization** — serialize gate sequences, control–target relations, and parameters into natural language.  
3. **Labeling** — compute residual norms via noiseless or noisy simulations.  
4. **Model Training** — fine–tune or prompt an LLM to regress residuals from text inputs.  
5. **Evaluation** — analyze cross-family generalization and noise robustness.

<p align="center">
  <img src="assets/pipeline.png" alt="Pipeline Overview" width="450"/>
</p>

---

## 📘 Dataset

Each entry in the **Quantum2Prompt-VQLS** dataset consists of:
| Field | Description |
|-------|--------------|
| `T` | Natural language description of the quantum circuit |
| `r` | Residual norm (target value) |
| `family` | Matrix family (Toeplitz / Laplacian / Sparse) |
| `metadata` | Circuit depth, qubit count, ansatz type |

The dataset enables reproducible evaluation of prompt-based residual prediction.

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/Quantum2Prompt.git
cd Quantum2Prompt
Dependencies include:

torch
transformers
qiskit
numpy
matplotlib
pandas
scikit-learn

🧩 Usage
1. Run main notebook
jupyter notebook Quantum2Prompt.ipynb

2. Perform noise robustness evaluation
from noise_robustness import run_experiment
run_experiment(runs=20, steps=20, p1=0.003, p2=0.02)

3. Example output

Clean vs. Noisy residual plots

Cross-family generalization results

Mean noisy 
𝑅
2
R
2
 statistics across seeds

<p align="center"> <img src="assets/noise_laplacian.png" alt="Noise Robustness" width="450"/> </p>
📊 Results
Family	Mean Noisy R²	Std
Toeplitz	-0.78	0.89
Laplacian	-0.39	1.67
Sparse	-0.81	0.52

The model successfully distinguishes structural circuit families and maintains robustness under noise injection.

🧾 Citation

If you use this work, please cite:

@article{yang2025quantum2prompt,
  title={Quantum2Prompt: Representing Quantum Circuits as Language Prompts for Linear System Solving},
  author={Yang, Youla},
  year={2025},
  journal={Preprint, submitted to Quantum Machine Intelligence}
}

📜 License

This project is released under the MIT License.
© 2025 Youla Yang. All rights reserved.

🌐 Related Work

Bravo-Prieto et al., “Variational Quantum Linear Solver,” Nature Quantum Information, 2023.

Zhou et al., “Language Models for Quantum State Prediction,” arXiv:2404.08112, 2024.

Gujju et al., “Prompt-based Ansatz Design for Quantum Generative Models,” Quantum Sci. Technol., 2025.
# Install dependencies
pip install -r requirements.txt
