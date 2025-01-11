from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)


def inference(model, tokenizer, prompt):
    
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer, max_length=150)
    
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(f"Question: {prompt}\n")
    print("Answer: ")
    print(result[0]['generated_text'].split("[/INST]")[1])

if __name__ == "__main__":
    
    logging.set_verbosity(logging.CRITICAL)
   
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    
    prompt = "What is yozgat?"
    inference(model, tokenizer, prompt)
    



    