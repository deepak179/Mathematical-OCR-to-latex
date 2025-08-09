# config.py

import torch
from unsloth import is_bf16_supported

# =================================================================================
# Model and Tokenizer Configuration
# =================================================================================
MODEL_ID = "unsloth/Qwen2-VL-7B-Instruct"
MAX_SEQ_LENGTH = 2048

# =================================================================================
# Dataset Configuration
# =================================================================================
DATASET_ID = "unsloth/Latex_OCR"
TRAIN_SAMPLES = 2000
TEST_SAMPLES = 1000
INSTRUCTION_PROMPT = "Write the LaTex representation for this image."

# =================================================================================
# LoRA (Fine-Tuning) Configuration
# =================================================================================
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "random_state": 3407,
    "use_rslora": False,
    "loftq_config": None,
    "finetune_vision_layers": True,
    "finetune_language_layers": True,
    "finetune_attention_modules": True,
    "finetune_mlp_modules": True,
}

# =================================================================================
# SFTTrainer (Training) Configuration
# =================================================================================
TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_steps": 200,
    "learning_rate": 2e-4,
    "fp16": not is_bf16_supported(),
    "bf16": is_bf16_supported(),
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "outputs",
    "report_to": "none",
    "remove_unused_columns": False,
    "dataset_text_field": "",
    "dataset_kwargs": {"skip_prepare_dataset": True},
    "dataset_num_proc": 4,
    "max_seq_length": MAX_SEQ_LENGTH,
}

# =================================================================================
# Evaluation and Saving Configuration
# =================================================================================
EVAL_MAX_NEW_TOKENS = 128
SAVED_MODEL_DIR = "latex_ocr_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"