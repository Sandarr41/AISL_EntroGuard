from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenModel:
    def __init__(self):
        model_name = "Qwen/Qwen2.5-7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)