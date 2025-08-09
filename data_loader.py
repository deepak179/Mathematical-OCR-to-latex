# data_loader.py

from datasets import load_dataset
import config

def load_ocr_dataset():
    """Loads and subsets the Latex OCR dataset from Hugging Face."""
    try:
        train_dataset_full = load_dataset(config.DATASET_ID, split="train")
        test_dataset_full = load_dataset(config.DATASET_ID, split="test")
        print("✅ Datasets loaded successfully.")
        
        train_subset = train_dataset_full.select(range(config.TRAIN_SAMPLES))
        test_subset = test_dataset_full.select(range(config.TEST_SAMPLES))

        print(f"Using {len(train_subset)} samples for training.")
        print(f"Using {len(test_subset)} samples for evaluation.")
        return train_subset, test_subset

    except Exception as e:
        print(f"❌ Failed to load dataset. Error: {e}")
        return None, None

def format_dataset_for_training(dataset):
    """
    Formats the raw dataset into a conversational structure required by the model.
    """
    def convert_to_conversation(sample):
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": config.INSTRUCTION_PROMPT},
                    {"type": "image", "image": sample["image"]}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": sample["text"]}
                ]}
            ]
        }
    
    return [convert_to_conversation(sample) for sample in dataset]