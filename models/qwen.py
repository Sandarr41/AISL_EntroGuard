import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenModel:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=150):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    def generate(self, prompt, max_new_tokens=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_tokens = max_new_tokens or self.max_new_tokens

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )


        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()