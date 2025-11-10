# # system_prompt = (
# #     "You are an Medical assistant for question-answering tasks. "
# #     "Use the following pieces of retrieved context to answer "
# #     "the question. If you don't know the answer, say that you "
# #     "don't know. Use three sentences maximum and keep the "
# #     "answer concise."
# #     "\n\n"
# #     "{context}"
# # )
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# import os
# import numpy as np
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec

# # 1. Load environment variables
# load_dotenv()
# api_key = os.getenv("PINECONE_API_KEY")

# embeddings = HuggingFaceBgeEmbeddings(model_name="./bge-large-en-v1.5")

# if not api_key:
#     raise ValueError("No PINECONE_API_KEY found in .env file!")

# # 2. Initialize Pinecone client
# pc = Pinecone(api_key=api_key)

# # 3. Create the index if it doesn’t already exist
# index_name = "sample-data-1024"

# if index_name not in [idx["name"] for idx in pc.list_indexes()]:
#     pc.create_index(
#         name=index_name,
#         dimension=1024,  # 1024-dim embeddings
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
#     print(f"Index '{index_name}' created.")
# else:
#     print(f"Index '{index_name}' already exists.")

# # 4. Connect to the index
# index = pc.Index(index_name)

# documents = [
#     "An apple a day keeps the doctor away",
#     "The quick brown fox jumps over the lazy dog",
#     "Artificial intelligence is transforming industries",
#     "The capital of France is Paris",
#     "Water freezes at zero degrees Celsius",
#     "Mistral is a popular open-source LLM",
#     "Pinecone enables vector search at scale",
#     "The moon orbits the Earth",
#     "Python is a versatile programming language",
#     "SpaceX is developing the Starship rocket"
# ]

# # 🔹 Generate embeddings and upsert
# vectors = []
# for i, doc in enumerate(documents):
#     vector = embeddings.embed_query(doc)
#     vectors.append({
#         "id": f"doc{i+1}",
#         "values": vector,
#         "metadata": {"text": doc}
#     })


# index.upsert(vectors=vectors)
# print("Inserted 10 vectors into index.")


# system_prompt = (
#     "You are an Medical assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )


# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# import os
# import numpy as np
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain.schema import Document
# import pickle

# with open(r"D:\web dev backup\Pytorch\RAG4_MBBS_MM\Gale\new_Gale_med_book_01_04.pkl", "rb") as f:
#     data = pickle.load(f)

# print(len(data))       # should be 12888
# print(data[0].keys())  # should show 'type', 'element_id', 'text', 'metadata'


# # 1. Load environment variables
# load_dotenv()
# api_key = os.getenv("PINECONE_API_KEY")

# embeddings = HuggingFaceBgeEmbeddings(model_name="./bge-large-en-v1.5")

# if not api_key:
#     raise ValueError("No PINECONE_API_KEY found in .env file!")

# # 2. Initialize Pinecone client
# pc = Pinecone(api_key=api_key)

# # 3. Create the index if it doesn’t already exist
# index_name = "medical-bge"

# if index_name not in [idx["name"] for idx in pc.list_indexes()]:
#     pc.create_index(
#         name=index_name,
#         dimension=1024,  # 1024-dim embeddings
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
#     print(f"Index '{index_name}' created.")
# else:
#     print(f"Index '{index_name}' already exists.")

# # 4. Connect to the index
# index = pc.Index(index_name)

# batch_size = 100

# def convert_tuples(obj):
#     if isinstance(obj, dict):
#         return {k: convert_tuples(v) for k, v in obj.items()}
#     elif isinstance(obj, tuple):
#         return [convert_tuples(x) for x in obj]
#     elif isinstance(obj, list):
#         return [convert_tuples(x) for x in obj]
#     else:
#         return obj





# def flatten_metadata(meta):
#     flat = {}
#     for k, v in meta.items():
#         if isinstance(v, dict):
#             # convert dict to JSON string
#             flat[k] = str(v)
#         elif isinstance(v, tuple):
#             # convert tuple to list
#             flat[k] = list(v)
#         elif isinstance(v, list):
#             # convert inner tuples to lists if any
#             flat[k] = [list(item) if isinstance(item, tuple) else item for item in v]
#         else:
#             flat[k] = v
#     return flat

# vectors = []

# # for i, elem in enumerate(data[:100]):
# #     text = elem["text"]
# #     vector = embeddings.embed_query(text)
# #     meta = flatten_metadata(elem["metadata"])
# #     meta["original_text"] = text

# #     vectors.append({
# #         "id": str(elem["element_id"]),
# #         "values": vector,
# #         "metadata": meta
# #     })


# documents = []
# for elem in data[:100]:  # Your sample range
#     doc = Document(
#         page_content=elem["text"],  # This is the main text LangChain expects
#         metadata={
#             "element_id": elem["element_id"],
#             "original_text": elem["text"],  # Keep your original text here too
#             # Add any other metadata you want to preserve
#             **elem.get("metadata", {})
#         }
#     )
#     documents.append(doc)


# # upsert just this one batch
# index.upsert(vectors)
# print(f"Upserted {len(vectors)} vectors")


from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import pickle


# Load your data
# with open(r"D:\web dev backup\Pytorch\RAG4_MBBS_MM\Gale\new_Gale_med_book_01_18.pkl", "rb") as f:
#     data = pickle.load(f)

# with open(r"D:\web dev backup\Pytorch\RAG4_MBBS_MM\Gale\02\new_Gale_med_book_14.pkl", "rb") as f:
#     data = pickle.load(f)
    
with open(r"D:\web dev backup\Pytorch\RAG4_MBBS_MM\Gale\03\new_Gale_med_book_15.pkl", "rb") as f:
    data = pickle.load(f)
print(f"Total records: {len(data)}")

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

embeddings = HuggingFaceBgeEmbeddings(model_name="./bge-large-en-v1.5")

if not api_key:
    raise ValueError("No PINECONE_API_KEY found in .env file!")

# 2. Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# 3. Create the index if it doesn't exist
index_name = "medical-bge"


# # Delete old index (check if exists)
# if index_name in [idx["name"] for idx in pc.list_indexes()]:
#     pc.delete_index(index_name)
#     print(f"✅ Index '{index_name}' deleted successfully.")


if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# 4. Connect to the index
index = pc.Index(index_name)

# 5. Prepare vectors with ONLY the text
vectors = []

batch_size = 100  # Number of vectors per batch
vectors = []

for start in range(0, len(data), batch_size):  # Loop over your range in steps of batch_size
    vectors = []  # Reset the batch for each iteration

    for elem in data[start:start + batch_size]:
        text = elem["text"].strip()
        
        # Skip empty texts
        if not text:
            continue
        
        # Generate embedding
        vector = embeddings.embed_query(text)
        
        # Store ONLY the text in metadata
        meta = {"text": text}
        
        # Append vector to current batch
        vectors.append({
            "id": str(elem["element_id"]),
            "values": vector,
            "metadata": meta
        })
    print(f"till {start+100} has been upserted")
    
    # Upsert the current batch into Pinecone
    index.upsert(vectors)
    print(f"Upserted batch: {start} to {start + len(vectors) - 1}")

# 6. Upsert to Pinecone
if vectors:
    index.upsert(vectors)
    print(f"✅ Successfully upserted {len(vectors)} vectors")
    print(f"Sample stored text: '{vectors[0]['metadata']['text']}'")
else:
    print("❌ No vectors to upsert")