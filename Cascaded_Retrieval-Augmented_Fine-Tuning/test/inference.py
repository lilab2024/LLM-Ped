import json
import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import LogitsProcessorList
from sentence_transformers import SentenceTransformer
import chromadb
# ======================= 1. 配置 =======================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


base_model_path = "Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3-4B-CRAFT-stage1"


lora_path="Cascaded_Retrieval-Augmented_Fine-Tuning/train/checkpoint/Qwen3-4B-CRAFT/checkpoint-350"
device = "cuda"



#配置RAG
Embeddingmodel = SentenceTransformer('Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3-Embedding-0.6B')
# 初始化Chroma客户端（只初始化1次）
chroma_client = chromadb.PersistentClient(path="Cascaded_Retrieval-Augmented_Fine-Tuning/dataset/driver_yield_vector_db")  # 数据库文件存在本地
# 加载已创建的collection（你的向量库名称，比如"driver_yielding"）
collection = chroma_client.get_collection(name="driver_yield_knowledge-28")


# 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ======================= 2. 加载模型 =======================
tokenizer_model_path="Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()



def build_prompt(instruction, inp):
    messages = [
        {
            "role": "system",
            "content": f"{instruction}"
        },
        {
            "role": "user",
            "content": f"Input: {inp} "
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5000,
            do_sample=True,  # 平 determinism

       
            temperature=0.7,    # 对应Temperature=0.7
            top_p=0.8,          # 对应TopP=0.8
            top_k=20,           # 对应TopK=20
            min_p=0.0
                    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



y_true = []
y_pred = []





cot="""\n
-Here are some steps to guide you on how to think:
Step 1: Analyze vehicle attributes.Through relevant knowledge and one's own reasoning.
Step 2: Evaluate road conditions.Through relevant knowledge, pictures and one's own reasoning.
Step 3: Analyze pedestrian-related characteristics.Through relevant knowledge and one's own reasoning.
Step 4: Establish the priority of factors influencing driver yielding behavior. Prioritize the following features: Vehicle speed, Crossing width (major), Opposite direction yield, Presence of parking lots, Presence of Restaurants/Bars, Distance to the nearest park, and Distance to the nearest school. Then combine the other characteristics to get the final result, and the reason for the result.
Step 5: Result and reson.
"""


for i in range(1,19):
    output_csv = f"Cascaded_Retrieval-Augmented_Fine-Tuning/test/result/CRAFT-Qwen3-4B/output{i}.csv"
    test_file = f"Cascaded_Retrieval-Augmented_Fine-Tuning/dataset/test_dataset/location_{i}.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx, item in enumerate(test_data, start=1):
            instruction = item["instruction"]
            inp = item["input"]

            inp_emb = Embeddingmodel.encode(inp).tolist()
            rag_results = collection.query(
                query_embeddings=[inp_emb],  # 传入inp的向量
                n_results=10  # 返回top_k个最相似的文档
                )
            # print(rag_results.get("ids"))
            retrieved_docs = rag_results.get("documents", [[]])[0]  # 提取文档列表
            doc_parts = []
            for doc in retrieved_docs:
                doc_parts.append(f"{doc}")

            rag_begin="\n\n### Relevant Domain Knowledge that can help you make better predictions.Analyze step by step and finally provide the answer along with the reasons\n."    
            
            rag_context = rag_begin+ "\n".join(doc_parts)

            augmented_instr = instruction + rag_context

            augmented_instr+=cot

            prompt = build_prompt(augmented_instr, inp)
            response = generate_answer(prompt)

            # 直接写一行
            writer.writerow([response])

            print(f"第 {idx} 条写入完成：{response}")

    print(f"--------{i}Location完成----------")

print(f"所有模型回答已写入 {output_csv}")

