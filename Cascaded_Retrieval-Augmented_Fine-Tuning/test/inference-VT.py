import json
import os
import csv
import re
import torch
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# ======================= 1. 配置 =======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 模型路径

base_model_path = "Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3.5-4B-CRAFT-stage1"
lora_path="Cascaded_Retrieval-Augmented_Fine-Tuning/train/checkpoint/Qwen3.5-4B-sft/checkpoint-350"

IMAGE_ROOT_DIR = "LR_and_GLLM/image" 

device = "cuda"

# 配置 RAG (保持不变)
Embeddingmodel = SentenceTransformer('Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3-Embedding-0.6B')
chroma_client = chromadb.PersistentClient(path="Cascaded_Retrieval-Augmented_Fine-Tuning/dataset/driver_yield_vector_db")
collection = chroma_client.get_collection(name="driver_yield_knowledge-28")

# 4-bit 量化 (保持不变)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ======================= 2. 加载多模态模型 =======================

tokenizer_path="Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3-4B-VL"

processor = Qwen3VLProcessor.from_pretrained(
    tokenizer_path, 
    trust_remote_code=True
)


base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()



def load_image_by_input(input_text):

    location_id_pattern = re.compile(r"Location ID:\s*(\d+)")
    match = location_id_pattern.search(input_text)
    if not match:
        raise ValueError(f"未找到 Location ID: {input_text[:50]}...")
    loc_id = match.group(1)
    
    image_path = os.path.join(IMAGE_ROOT_DIR, f"Slide{loc_id}.png")
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except FileNotFoundError:
        raise FileNotFoundError(f"图片不存在: {image_path}")





#+"analyze step by step and finally provide the answer along with the reasons."
#\n\nRelevant Domain Knowledge that can help you make better predictions.\nAnalyze step by step and finally provide the answer along with the reasons\n
def build_vl_prompt(instruction, inp, image):
    messages = [
        {
            "role": "system",
            "content": f"{instruction}"
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},  # 图片占位符
                {"type": "text", "text": f"Input: {inp}"}
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

def generate_answer(vl_prompt, image):
    inputs = processor(
        text=vl_prompt,
        # images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=40,
            repetition_penalty=1.0
        )
    
    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)


cot="""\n
-Here are some steps to guide you on how to think:
Step 1: Analyze vehicle attributes.Through relevant knowledge and one's own reasoning.
Step 2: Evaluate road conditions.Through relevant knowledge, pictures and one's own reasoning.
Step 3: Analyze pedestrian-related characteristics.Through relevant knowledge and one's own reasoning.
Step 4: Establish the priority of factors influencing driver yielding behavior. Prioritize the following features: Vehicle speed, Crossing width (major), Opposite direction yield, Presence of parking lots, Presence of Restaurants/Bars, Distance to the nearest park, and Distance to the nearest school. Then combine the other characteristics to get the final result, and the reason for the result.
Step 5: Result and reson.
"""


#Analyze step by step and finally provide the answer along with the reasons\n.
for i in range(1, 19):
    output_csv = f"Cascaded_Retrieval-Augmented_Fine-Tuning/test/result/CRAFT-Qwen3.5B/output{i}.csv"
    test_file = f"Cascaded_Retrieval-Augmented_Fine-Tuning/dataset/test_dataset/location_{i}.json"
    
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        for idx, item in enumerate(test_data, start=1):
            instruction = item["instruction"]
            inp = item["input"]
           
            image = load_image_by_input(inp)
 
            inp_emb = Embeddingmodel.encode(inp).tolist()
            rag_results = collection.query(
                    query_embeddings=[inp_emb],
                    n_results=8
                )
            retrieved_docs = rag_results.get("documents", [[]])[0]
            doc_parts = [f"{doc}" for doc in retrieved_docs]
                
            rag_begin = "\n\n### Relevant Domain Knowledge that can help you make better predictions.Analyze step by step and finally provide the answer along with the reasons\n."    
            rag_context = rag_begin + "\n".join(doc_parts)

            augmented_instr = instruction +rag_context
            augmented_instr += cot

            prompt = build_vl_prompt(augmented_instr, inp, image)

            response = generate_answer(prompt, image)

            writer.writerow([response])
            print(response)
            print(f"第 {idx} 条写入完成")

    print(f"--------{i} Location 完成----------")

print(f"所有模型回答已写入")