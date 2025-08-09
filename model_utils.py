# model_utils.py

from unsloth import FastVisionModel
import config

def load_base_model(load_in_4bit=True):
    """Loads the base vision model and tokenizer from Hugging Face."""
    model, tokenizer = FastVisionModel.from_pretrained(
        config.MODEL_ID,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
        max_seq_length=config.MAX_SEQ_LENGTH,
    )
    return model, tokenizer

def prepare_model_for_finetuning(model):
    """Applies PEFT (LoRA) configuration to the model to make it trainable."""
    model = FastVisionModel.get_peft_model(
        model,
        **config.LORA_CONFIG
    )
    return model