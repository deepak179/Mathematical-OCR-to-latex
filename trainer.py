# trainer.py

from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
import config

def run_training(model, tokenizer, train_dataset):
    """
    Configures and runs the SFTTrainer for fine-tuning the model.
    """
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        args=SFTConfig(**config.TRAINING_ARGS),
    )

    print("\n🚀 Starting model fine-tuning...")
    trainer_stats = trainer.train()
    print("✅ Fine-tuning complete.")
    
    return trainer.model, trainer_stats