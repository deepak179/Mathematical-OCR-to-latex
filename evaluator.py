# evaluator.py

import torch
from tqdm import tqdm
import jiwer
import config
from unsloth import FastVisionModel

def generate_predictions(model, tokenizer, dataset):
    """
    Generates predictions for a given dataset using the specified model.
    """
    model.to(config.DEVICE)
    FastVisionModel.for_inference(model)
    
    predictions = []
    ground_truths = []

    for item in tqdm(dataset, desc=f"Evaluating on {len(dataset)} samples"):
        image = item["image"]
        true_latex = item["text"]
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": config.INSTRUCTION_PROMPT},
                {"type": "image"}
            ]}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(image, input_text, return_tensors="pt").to(config.DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                use_cache=True
            )
        
        full_output = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        
        # Extract only the generated part of the response
        gen_prompt_str = "<|im_start|>assistant\n"
        if gen_prompt_str in full_output:
            pred_start_idx = full_output.rfind(gen_prompt_str) + len(gen_prompt_str)
            prediction = full_output[pred_start_idx:].replace("<|im_end|>", "").strip()
        else:
            # Fallback if the template isn't found
            prediction = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()

        predictions.append(prediction)
        ground_truths.append(true_latex)
        
    return predictions, ground_truths

def calculate_and_print_cer(base_preds, finetuned_preds, truths):
    """Calculates and prints the Character Error Rate comparison."""
    base_cer = jiwer.cer(truths, base_preds)
    finetuned_cer = jiwer.cer(truths, finetuned_preds)
    improvement = base_cer - finetuned_cer

    print("\n" + "="*50)
    print("           Model Performance Comparison")
    print("="*50)
    print(f"  Base Model CER (Before Fine-Tuning):     {base_cer:.4f}")
    print(f"  Fine-Tuned Model CER (After Fine-Tuning): {finetuned_cer:.4f}")
    print("-" * 50)
    print(f"  Improvement in CER (Reduction):          {improvement:.4f}")
    print(f"  Relative Improvement:                    {(improvement / base_cer) * 100:.2f}%")
    print("="*50)
    print("\nA lower Character Error Rate (CER) indicates higher accuracy.")