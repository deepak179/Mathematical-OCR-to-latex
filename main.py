# main.py

import torch
from unsloth import FastVisionModel
import config
import data_loader
import model_utils
import trainer
import evaluator

def main():
    """Main function to run the entire OCR model workflow."""
    
    # 1. Load Dataset
    print("="*80)
    print("1. LOADING DATASET")
    print("="*80)
    train_subset, test_subset = data_loader.load_ocr_dataset()
    if train_subset is None or test_subset is None:
        return

    # 2. Load Base Model
    print("\n" + "="*80)
    print("2. LOADING BASE MODEL")
    print("="*80)
    base_model, tokenizer = model_utils.load_base_model()

    # 3. Baseline Evaluation
    print("\n" + "="*80)
    print("3. BASELINE EVALUATION (BEFORE FINE-TUNING)")
    print("="*80)
    base_predictions, ground_truths = evaluator.generate_predictions(base_model, tokenizer, test_subset)
    print("✅ Baseline evaluation complete.")

    # 4. Fine-Tuning Preparation
    print("\n" + "="*80)
    print("4. PREPARING MODEL FOR FINE-TUNING")
    print("="*80)
    # We use the same 'base_model' instance and apply LoRA adapters to it.
    model_to_finetune = model_utils.prepare_model_for_finetuning(base_model)
    formatted_train_dataset = data_loader.format_dataset_for_training(train_subset)
    print("✅ Model and training dataset prepared.")
    
    # 5. Training
    print("\n" + "="*80)
    print("5. TRAINING THE MODEL")
    print("="*80)
    finetuned_model, _ = trainer.run_training(model_to_finetune, tokenizer, formatted_train_dataset)

    # 6. Save the Fine-Tuned Model
    print("\n" + "="*80)
    print("6. SAVING THE FINE-TUNED MODEL")
    print("="*80)
    finetuned_model.save_pretrained(config.SAVED_MODEL_DIR)
    tokenizer.save_pretrained(config.SAVED_MODEL_DIR)
    print(f"✅ Model saved to '{config.SAVED_MODEL_DIR}' directory.")
    
    # 7. Evaluation after Fine-Tuning
    print("\n" + "="*80)
    print("7. FINAL EVALUATION (AFTER FINE-TUNING)")
    print("="*80)
    finetuned_predictions, _ = evaluator.generate_predictions(finetuned_model, tokenizer, test_subset)
    print("✅ Fine-tuned model evaluation complete.")

    # 8. Compare Results
    print("\n" + "="*80)
    print("8. COMPARING RESULTS")
    print("="*80)
    evaluator.calculate_and_print_cer(base_predictions, finetuned_predictions, ground_truths)

if __name__ == "__main__":
    main()