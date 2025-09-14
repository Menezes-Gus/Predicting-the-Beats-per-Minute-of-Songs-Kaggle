# Kaggle Competition: Predicting the Beats per Minute of Songs

## 🎯 Objective
The goal of this competition is to predict a song's beats-per-minute.

## 📂 Structure
- `data/` → raw, processed and external data.
- `notebooks/` → notebook flow: EDA → baseline → modeling → submission.
- `src/` → reusable functions for preprocessing, modeling e metrics.
- `results/` → charts, metrics and Kaggle submission.
- `models/` → trained models binaries.
- `requirements.txt` → dependencies.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Menezes-Gus/Predicting-the-Beats-per-Minute-of-Songs-Kaggle.git
   ```
2. If you don't already have the data zip file in data/raw folder, run download_data.py:
    ```bash
   python src/download_data.py
   ```
3. If you don't already extracted the zip file downloaded in the previous step, run extract_zip.py:
    ```bash
   python src/extract_zip.py
   ```



Name: Gustavo Menezes

LinkedIn:  www.linkedin.com/in/gustavomenezes-263b00156 

GitHub: https://github.com/Menezes-Gus

```bibtex
@misc{playground-series-s5e9,
    author = {Walter Reade and Elizabeth Park},
    title = {Predicting the Beats-per-Minute of Songs},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s5e9}},
    note = {Kaggle}
}
```