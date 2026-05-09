# Pixels to Predictions — ScienceQA Vision Challenge

**NYU Deep Learning Spring 2026 | Clement Wang & Raunak Ahmed**

We fine-tuned a vision-language model to answer multiple-choice elementary science questions, each paired with an image, a question, and optional text context. The model must return the index of the correct answer choice. Our best submission scored **0.82494** on the public leaderboard — a 17-point improvement over the zero-shot baseline.

---

## What We Did

The competition gave us [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct), a 500M-parameter vision-language model, and capped us at 5 million trainable parameters and the 6,218 provided training examples. Everything had to run on a free Google Colab session.

Our development went through two phases. In the first, we applied QLoRA adapters (rank 16, attention projections only) on top of 4-bit quantized weights and replaced the starter notebook's `model.generate()` approach with log-likelihood scoring — assigning each answer choice a score based on the log-probability of the corresponding letter token, then returning the argmax. This alone pushed accuracy from 65.4% to 74.8%.

In the second phase we moved to native bfloat16 on an A100, expanded LoRA to all seven projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), increased image resolution from 224 to 384 pixels, and trained for 5 epochs. We also added choice-order shuffling during training to prevent the model from exploiting positional shortcuts, and test-time augmentation that scores each question twice (original and reversed choice order) to further reduce positional bias. The final model reached 80.44% on the validation set and 82.49% on the leaderboard.

The single most important lesson: the training objective and the inference objective have to match at the token level. Every time we tried to enrich the supervision signal by adding full answer text or explanation suffixes to the training target, accuracy dropped — because those easy-to-predict tokens diluted the gradient away from the one letter token that actually determines the evaluation outcome.

---

## Results

| Configuration | Leaderboard Score |
|---|---|
| Zero-shot baseline | 0.65392 |
| QLoRA r=16, attn only, 2 epochs | 0.74849 |
| r=24, 3 epochs, no TTA | 0.76056 |
| r=24, choice shuffling, TTA | 0.79275 |
| **bfloat16, r=8, all proj., 384px, 5 epochs** | **0.82494** |

---

## How to Reproduce

You'll need a Google Colab session with an A100 GPU and the dataset files (`train.csv`, `val.csv`, `test.csv`, and the images directory) stored in Google Drive.

1. Mount your Drive in Colab and symlink the images directory to `/content/images`.
2. Open `starter_notebook_Final-DL.ipynb` and run all cells in order. The notebook handles everything from model loading through LoRA setup, training (~80 minutes per epoch), and inference.
3. The final submission CSV is written to `/content/submission_final.csv`.

All key parameters — image resolution, LoRA rank, learning rate, epochs — are collected in a single config cell near the top of the notebook so you don't have to hunt through the code to change things.

The model loads in bfloat16 with `device_map='auto'` and requires no quantization library for the final configuration.

---

## Project Report

The written report is in [`Main/latex/acl_latex.tex`](Main/latex/acl_latex.tex) and covers the full methodology, ablation studies, and results in ACL format. To compile it in Overleaf, set the main document to `latex/acl_latex.tex` and the compiler to pdfLaTeX.

---

*Clement Wang (cyw6947@nyu.edu) · Raunak Ahmed (ra4870@nyu.edu) · New York University*
