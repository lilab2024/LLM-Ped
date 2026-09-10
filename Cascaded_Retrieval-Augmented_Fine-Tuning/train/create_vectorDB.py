import chromadb
from sentence_transformers import SentenceTransformer
import torch

# ---------------------- 1. 初始化向量数据库客户端 ----------------------
client = chromadb.PersistentClient(path="Cascaded_Retrieval-Augmented_Fine-Tuning\dataset\driver_yield_vector_db")
collection = client.get_or_create_collection(
    name="driver_yield_knowledge-28",
    metadata={"description": "Store vectorized domain knowledge about driver yielding behavior"}
)


# ---------------------- 1. 检测设备（新增） ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 自动检测到可用设备：{DEVICE}")


# ---------------------- 2. 整理驾驶员让行的领域知识（全英文结构化拆分） ----------------------
driver_yield_knowledge = [
    # Core attribute priority
    {
        "id": "yield_attr_001",
        "content": "At unsignalized intersections, driver yielding behavior is dominated by vehicle speed, while factors such as major-road crossing width, opposite-direction vehicle behavior, nearby parking lots, nearby restaurants or bars, and proximity to parks and schools play progressively smaller roles"
    },
    # Vehicle Dynamics and Control
    {
        "id": "vehicle_dyn_001",
        "content": "At unsignalized intersections, the speed of an approaching vehicle plays a central role in determining whether a driver yields to pedestrians"
    },
    {
        "id": "vehicle_dyn_002",
        "content": "As the approaching vehicle’s speed increases, drivers are less likely to yield during pedestrian–vehicle interactions"
    },
    {
        "id": "vehicle_dyn_003",
        "content": "When a vehicle approaches an unsignalized intersection at a speed below 10 MPH, drivers are commonly observed to yield to pedestrians"
    },
    {
        "id": "vehicle_dyn_004",
        "content": "When vehicle speeds range between 10 and 20 MPH, yielding behavior at unsignalized intersections becomes uncertain, with drivers exhibiting mixed yielding and non-yielding responses"
    },
    {
        "id": "vehicle_dyn_005",
        "content": "Vehicles approaching at speeds above 20 MPH are rarely observed to yield to pedestrians at unsignalized intersections"
    },
    {
        "id": "vehicle_dyn_006",
        "content": "Unsignalized road segments with a posted speed limit of 30 MPH tend to exhibit higher pedestrian yielding rates than those with a 35 MPH speed limit"
    },
    {
        "id": "vehicle_dyn_007",
        "content": "During pedestrian–vehicle interactions, drivers are strongly influenced by the yielding behavior of vehicles traveling in the opposite direction"
    },
    {
        "id": "vehicle_dyn_008",
        "content": "If an opposite-direction vehicle yields to pedestrians at an unsignalized intersection, the subject driver is much more likely to yield as well"
    },
    {
        "id": "vehicle_dyn_009",
        "content": "If an opposite-direction vehicle does not yield, the subject driver is unlikely to yield in the same interaction scenario"
    },

    # Road Geometry and Traffic Environment
    {
        "id": "road_geom_001",
        "content": "At unsignalized intersections, wider crossing distances on the major road are often associated with higher driver yielding rates"
    },
    {
        "id": "road_geom_002",
        "content": "A larger number of traffic lanes on the major road typically reduces the likelihood that drivers will yield to pedestrians"
    },
    {
        "id": "road_geom_003",
        "content": "When more pedestrians are present near an unsignalized intersection, drivers are more likely to yield during crossing events"
    },
    {
        "id": "road_geom_004",
        "content": "The presence of nearby bus stops is commonly associated with increased driver awareness and higher yielding rates"
    },

    # Built Environment Context
    {
        "id": "built_env_001",
        "content": "Unsignalized intersections located near restaurants or bars often exhibit higher driver yielding rates, reflecting increased pedestrian activity"
    },
    {
        "id": "built_env_002",
        "content": "The presence of parking lots near an unsignalized intersection is frequently associated with reduced driver yielding behavior"
    },
    {
        "id": "built_env_003",
        "content": "Commercial land uses surrounding an unsignalized intersection are commonly linked to higher pedestrian yielding rates"
    },
    {
        "id": "built_env_004",
        "content": "Gas stations and apartment complexes near an intersection are often associated with increased driver yielding during pedestrian crossings"
    },
    {
        "id": "built_env_005",
        "content": "Residential environments with single-family housing and on-street parking on both sides tend to encourage higher yielding rates compared to areas without on-street parking"
    },

    # Surrounding Sensitive Land Uses
    {
        "id": "sensitive_land_001",
        "content": "Unsignalized intersections located closer to parks frequently show higher driver yielding rates due to increased pedestrian presence"
    },
    {
        "id": "sensitive_land_002",
        "content": "Short distances to nearby schools are strongly associated with higher driver yielding behavior at unsignalized intersections"
    },

    # Traffic Control and Facilities
    {
        "id": "traffic_ctrl_001",
        "content": "Standard pedestrian crosswalk markings at unsignalized intersections are commonly associated with the highest yielding rates"
    },
    {
        "id": "traffic_ctrl_002",
        "content": "Unmarked crosswalks at unsignalized intersections are typically associated with lower driver yielding behavior"
    },
    {
        "id": "traffic_ctrl_003",
        "content": "The presence of traffic signage near an unsignalized intersection increases driver awareness and the likelihood of yielding"
    },
    {
        "id": "traffic_ctrl_004",
        "content": "Bike lanes at unsignalized intersections are often associated with reduced driver yielding to pedestrians"
    },

    # Pedestrian Mobility and Interaction Assessment
    {
        "id": "ped_interact_001",
        "content": "Drivers at unsignalized intersections are more likely to yield to groups of pedestrians than to single individuals"
    },
    {
        "id": "ped_interact_002",
        "content": "Pedestrians accompanied by strollers or children tend to receive higher yielding rates from approaching drivers"
    },
    {
        "id": "ped_interact_003",
        "content": "Pedestrians walking with a dog moderately increase the likelihood of driver yielding during intersection interactions"
    },
    {
        "id": "ped_interact_004",
        "content": "Pedestrians using bicycles or vehicles are less likely to receive yielding behavior from drivers at unsignalized intersections"
    },
    {
        "id": "ped_interact_005",
        "content": "Mixed pedestrian groups at unsignalized intersections generally experience higher yielding rates than single-mode pedestrian users"
    }
]

# ---------------------- 3. 加载嵌入模型，批量向量化英文文本 ----------------------
model = SentenceTransformer('Cascaded_Retrieval-Augmented_Fine-Tuning/model/Qwen3-Embedding-0.6B')  # 该模型完美支持英文文本向量化

print("---------------------")

ids = [item["id"] for item in driver_yield_knowledge]
documents = [item["content"] for item in driver_yield_knowledge]
embeddings = model.encode(documents).tolist()

# ---------------------- 4. 批量存入向量数据库 ----------------------
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)
print(f"✅ {len(driver_yield_knowledge)} pieces of driver yielding domain knowledge (English) stored in vector DB!")

# ---------------------- 5. 验证：英文语义检索示例 ----------------------
# Example 1: Query "How does vehicle speed affect yielding"
query1 = "How does vehicle speed influence driver yielding behavior"
emb1 = model.encode(query1).tolist()
results1 = collection.query(query_embeddings=[emb1], n_results=2)

# Example 2: Query "Impact of road infrastructure on yielding"
query2 = "What is the impact of road infrastructure and surrounding environment on driver yielding"
emb2 = model.encode(query2).tolist()
results2 = collection.query(query_embeddings=[emb2], n_results=3)

# Print query results
print("\n🔍 Query 1: How does vehicle speed influence driver yielding behavior")
for idx, doc in enumerate(results1["documents"][0]):
    print(f"  Match {idx+1}: {doc} (Similarity: {results1['distances'][0][idx]:.4f})")

print("\n🔍 Query 2: What is the impact of road infrastructure and surrounding environment on driver yielding")
for idx, doc in enumerate(results2["documents"][0]):
    print(f"  Match {idx+1}: {doc} (Similarity: {results2['distances'][0][idx]:.4f})")