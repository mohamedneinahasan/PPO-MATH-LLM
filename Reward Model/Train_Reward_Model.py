import json
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Load feedback data
def load_feedback_data():
    """Load feedback data from JSON file."""
    with open("feedback_data.json", "r") as f:   #dataset path
        return json.load(f)

# Define preprocessing function
def preprocess_data(examples, tokenizer, max_length=52):
    """Ensure consistent sequence lengths with padding and truncation."""
    inputs = tokenizer(
        examples["question"], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
    return {**inputs, "labels": examples["reward"]}  # Use numeric reward label

def train_reward_model():
    """Train a reward model based on user feedback."""
    feedback_data = load_feedback_data()

    # Convert feedback into numerical rewards (Like = 1, Unlike = 0)
    df = pd.DataFrame(feedback_data)
    df["reward"] = df["feedback"].apply(lambda x: 1 if x == "Like" else 0)

    # Load tokenizer and model (DistilBERT-based reward model)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Convert dataset to Hugging Face format
    dataset = Dataset.from_pandas(df)
    
    # Apply preprocessing
    dataset = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./reward_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    # Save trained model
    model.save_pretrained("./reward_model")
    tokenizer.save_pretrained("./reward_model")

    print("✅ Reward model trained and saved successfully!")

# Run the training function
if __name__ == "__main__":
    train_reward_model()
