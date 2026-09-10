from transformers import AutoModelForCausalLM,AutoModelForVision2Seq
from peft import PeftModel
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

base_model_path = "Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3-4B"
lora_path="Cascaded_Retrieval-Augmented_Fine-Tuning/train/checkpoint/Qwen3-4B-sft/checkpoint-350"
save_path = "Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3-4B-CRAFT-stage1"


model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
)

model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained(save_path, safe_serialization=True)