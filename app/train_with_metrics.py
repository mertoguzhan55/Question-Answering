import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
import os
from sklearn.model_selection import train_test_split


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    dataset = pd.read_csv(r"dataset/dataset.csv")
    print(dataset.shape)

    dataset = Dataset.from_pandas(dataset)
    print(dataset)

    train_df, eval_df = train_test_split(dataset, test_size=0.15, random_state=55)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)


    model_name = "NousResearch/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="balanced",
    )

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        output_dir="./output_with_metrics",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.001,
        logging_steps=15,
        fp16=True,
        bf16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="question", 
        tokenizer=tokenizer,
        eval_dataset = eval_dataset,
        args=training_arguments,
    )

    
    trainer.train()
    

    


if __name__ == "__main__":
    main()
