from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import torch
from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig
    
@dataclass
class LlamaInference:

    logger:any
    model_name: str
    task: str
    max_length: int
    prompt: str
    
                      
    def __post_init__(self):
        # torch_dtype=torch.float16 reduces GPU memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Using 4-bit quantization instead of 8-bit
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        self.model = AutoPeftModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    
    def generate_text(self, prompt):
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device("cpu")

        pipe = pipeline(task = self.task,
                    model = self.model,
                    tokenizer = self.tokenizer, max_length = self.max_length)
    
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        self.logger.info(f"Question: {prompt}\n")

        self.logger.info("Answer: ")
        self.logger.info(result[0]['generated_text'].split("[/INST]")[1])
        
        return result[0]['generated_text'].split("[/INST]")[1]
    



if __name__ == "__main__":
    logger_config = {
        "filepath":"../logs/yongatek_gpt.log",
        "rotation":"50MB",
        "level":"DEBUG",
    }
    logger = Logger(**logger_config) 
    config = {
        "is_active" : True,
        "model_type" : "llama",
        "model_name" :  "/home/claude/Documents/cv/mert-playground/yongatek-gpt/output/checkpoint-3468",
        "task" : "text-generation",
        "max_length" :  250,
        "num_return_sequences" :  1, # Bir prompt için kaç farklı çıktı üretileceğini belirler.Burada her seferinde tek bir çıktı üretilecek.
        "no_repeat_ngram_size" :  0, # Belirli uzunluktaki kelime gruplarının (n-gram) tekrarını engeller. Yani herhangi bir tekrar kısıtlaması yok.
        "top_p" :  1.0, # 0-1 arasındadır. 1.0 olarak ayarlandığında, model tüm olası token tahminlerini dikkate alır. Daha düşük bir değer (örn. 0.9),
                    # kümülatif olasılığı bu değere ulaşana kadar en olası tokenleri seçer. 
                    # Bu parametre yaratıcılık ve tutarlılık arasında denge kurmaya yarar.
        "top_k" :  50, # Modelin token tahminlerinden yalnızca en yüksek olasılıklı k tanesini dikkate alır. Diğer tahminler tamamen göz ardı edilir.
        "temperature" :  1.0
    }
    
    infer = LlamaInference(**config,logger= logger)
    prompt = "How many?"
    infer.generate_text(prompt)
