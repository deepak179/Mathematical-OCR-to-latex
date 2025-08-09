# Math-to-LaTeX: Image to LaTeX Converter

![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Datasets-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end deep learning project that fine-tunes a powerful vision-language model to perform Optical Character Recognition (OCR) on mathematical expressions and converts them into their corresponding LaTeX code. The project includes scripts for training, evaluation, and an interactive web application built with Streamlit.

---

## Demo

![Screenshot of the Streamlit App in action](https://i.imgur.com/your-screenshot-url.png)
*(Replace the URL above with a screenshot of your running `app.py`)*

---

## 📜 Table of Contents
- [Features](#-features)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Efficient Fine-Tuning**: Leverages the **Unsloth** library to fine-tune the `unsloth/Qwen2-VL-7B-Instruct` model with LoRA, enabling training on consumer-grade GPUs.
- **Performance Evaluation**: Systematically evaluates the model using Character Error Rate (CER) to measure accuracy before and after fine-tuning.
- **Interactive Web App**: A user-friendly front-end built with **Streamlit** that allows users to upload an image and instantly receive the generated LaTeX code and its rendered version.
- **Modular Codebase**: The project is organized into distinct modules for data loading, model utilities, training, and evaluation, making it easy to maintain and extend.
- **Optimized Inference**: Uses 4-bit quantization for faster and more memory-efficient predictions in the web application.

---

## 📊 Model Performance

The primary goal of fine-tuning is to reduce the Character Error Rate (CER). A lower CER indicates higher accuracy. The evaluation script (`evaluator.py`) compares the base model against our fine-tuned version on 1,000 test samples.

| Model Version      | Character Error Rate (CER) |
| ------------------ | -------------------------- |
| Base Model         | (e.g., 0.4531)             |
| **Fine-Tuned Model** | **(e.g., 0.0892)** |
| **Improvement** | **(e.g., 79.21%)** |

*(Note: The values above are placeholders. Run `main.py` to generate the actual performance metrics for your model.)*

---

## 📁 Project Structure

The project is organized into a modular structure for clarity and scalability.

```
latex_ocr_project/
├── 📂 latex_ocr_model/      # Saved fine-tuned model adapters
├── 📂 outputs/              # Training outputs from SFTTrainer
├── 📜 app.py                # The Streamlit web application
├── 📜 config.py             # Centralized configuration for all parameters
├── 📜 data_loader.py        # Handles dataset loading and preprocessing
├── 📜 evaluator.py          # Generates predictions and calculates CER
├── 📜 main.py                # Main script to run the full training & eval pipeline
├── 📜 model_utils.py         # Handles model and tokenizer loading
├── 📜 requirements.txt       # Project dependencies
├── 📜 trainer.py             # Manages the model training process
└── 📜 README.md              # This file
```

---

## ⚙️ Installation

Follow these steps to set up the project environment.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/latex_ocr_project.git](https://github.com/your-username/latex_ocr_project.git)
cd latex_ocr_project
```

**2. Create a Virtual Environment (Recommended)**
```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**
Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

This project has two main entry points: one for running the complete training and evaluation pipeline and another for launching the interactive web app.

### 1. Training and Evaluating the Model

To run the full pipeline—which includes baseline evaluation, fine-tuning, and final evaluation—execute the `main.py` script. This will train the model and save the fine-tuned adapters to the `latex_ocr_model/` directory.

```bash
python main.py
```

### 2. Running the Web Application

Once the model has been fine-tuned and saved, you can launch the Streamlit web application.

```bash
streamlit run app.py
```
Your web browser will open to `http://localhost:8501`, where you can upload an image and see the model in action.

---

## 🔧 Configuration

All key hyperparameters and settings can be easily modified in the `config.py` file. This includes:
- Model and dataset IDs
- Training parameters (learning rate, batch size, steps)
- LoRA configuration (rank, alpha)
- Directory paths

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new features, please feel free to fork the repository, make your changes, and open a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
