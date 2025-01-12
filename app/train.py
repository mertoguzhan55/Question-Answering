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
from dataclasses import dataclass


@dataclass
class LlamaTrainer():

    logger: any
    dataset_path: str
    test_size: float
    model_name: str
    trust_remote_code: bool
    padding_side: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    device_map: str
    lora_alpha: int
    lora_rank: int
    lora_dropout: float
    bias: str
    task_type: str
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    learning_rate: float
    logging_steps: int
    fp16: bool
    bf16: bool
    report_to: str
    dataset_text_field: str
    weight_decay: float = 0.001
    gradient_accumulation_steps: int = 1
    random_state: int = 55

    def train(self):
        dataset = pd.read_csv(self.dataset_path)  
        self.logger.info(f"Dataset boyutu: {dataset.shape}")

        # Eğitim ve doğrulama setlerini ayır
        train_df, eval_df = train_test_split(dataset, test_size=self.test_size, random_state=self.random_state)

        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)


        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        special_tokens_list = ["<project>", "<company>"]

        special_tokens_dict = {
            "additional_special_tokens": special_tokens_list
        }
        tokenizer.add_special_tokens(special_tokens_dict)

        

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = self.padding_side

        bnb_config = BitsAndBytesConfig(
            load_in_4bit = self.load_in_4bit,
            bnb_4bit_quant_type = self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = self.bnb_4bit_use_double_quant,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config = bnb_config,
            device_map = self.device_map,
        )
        model.resize_token_embeddings(len(tokenizer))

        

        self.logger.info(f"Llama layers: {model}")

        peft_config = LoraConfig(
            lora_alpha = self.lora_alpha,
            lora_dropout = self.lora_dropout,
            r = self.lora_rank,
            bias = self.bias,
            target_modules= ["q_proj","v_proj","o_proj"],
            task_type = self.task_type,
        )

        training_arguments = TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs = self.num_train_epochs,
            per_device_train_batch_size = self.per_device_train_batch_size,
            gradient_accumulation_steps = self.gradient_accumulation_steps,
            learning_rate = self.learning_rate,
            weight_decay = self.weight_decay,
            logging_steps = self.logging_steps,
            fp16 = self.fp16,
            bf16 = self.bf16,
            report_to = self.report_to,
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
    pass
