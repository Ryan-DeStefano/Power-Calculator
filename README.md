# Power Calculator for One- and Two-Sample *t*-Tests

This project is a Streamlit-based web application for calculating statistical power, required sample size, or detectable effect size for one- and two-sample *t*-tests. All calculations are implemented from scratch in Python — no built-in power analysis libraries are used — to provide a transparent and educational approach to hypothesis testing design.

## 🔍 Features

- Calculate:
  - Statistical **power**
  - **Required sample size**
  - **Detectable effect size**
- Support for:
  - **One-sample** *t*-tests
  - **Two-sample** (independent) *t*-tests
- Interactive UI via **Streamlit**
- Fully custom implementation based on core statistical theory

## 📂 Project Structure

```text
├── notebooks/
│   ├── one_sample_t_test.ipynb        # Derivations and scratch work for one-sample case
│   └── two_sample_t_test.ipynb        # Derivations and scratch work for two-sample case
│
├── utils/
│   ├── one_sample.py                  # Functions for one-sample power analysis
│   ├── two_sample.py                  # Functions for two-sample power analysis
│   └── utils.py                       # Shared utilities (e.g., common calculations)
│
├── power_app.py                       # Streamlit app script
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. (Optional) Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run power_app.py
```

### 5. Open in browser

Once started, the app should open automatically at:

```
http://localhost:8501
```

If it doesn’t open, visit that URL manually in your browser.

## 🧠 Motivation

Power analysis is a key part of experimental design, yet built-in functions in many libraries can obscure the underlying logic. This project offers an educational, hands-on approach by manually implementing the algorithms behind power, sample size, and effect size calculations for *t*-tests.

## 📈 Example Use Cases

- Determining the minimum sample size needed to detect a specific effect
- Exploring the relationship between power, effect size, and sample size
- Teaching or learning about the statistical foundations of hypothesis testing

## 🛠️ Technologies

- Python
- Streamlit
- NumPy
- SciPy (used only for validation/comparison)

## 📜 License

MIT License — feel free to use, modify, and share.
