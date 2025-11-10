from RAG14_prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
from dotenv import load_dotenv
import uuid
from pinecone import Pinecone

# Load API keys
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embeddings
embeddings = HuggingFaceBgeEmbeddings(model_name="./bge-large-en-v1.5")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Connect Pinecone
index_name = "sample-data-1024"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Build RAG chain using Ollama API via a custom function
import requests
import json

def query_ollama(messages, model="llava:7b-v1.6-mistral-q4_0"):
    """
    Send messages to local Ollama LLava model and return the combined response.
    Handles streaming/multi-line JSON from Ollama API.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages
    }

    response = requests.post(url, json=payload, stream=True)
    if response.status_code == 200:
        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:  # ignore empty lines
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        full_text += json_data["message"]["content"]
                except json.JSONDecodeError:
                    # Some lines might not be valid JSON yet, skip
                    continue
        return full_text
    else:
        return f"Request failed: {response.status_code} - {response.text}"

# Terminal loop
print("Type 'exit' or 'quit' to stop the bot.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    # Retrieve relevant docs with scores
    docs_and_scores = retriever.vectorstore.similarity_search_with_score(query, k=3)

    # Filter by score > 0.75
    filtered_docs = [doc for doc, score in docs_and_scores if score > 0.75]

    if not filtered_docs:
        print("No relevant match found in Pinecone.")
    else:
        for doc, score in docs_and_scores:
            print(f"Doc: {doc.page_content} | Score: {score}")

    # Combine the content from filtered docs
    context_text = "\n".join([doc.page_content for doc in filtered_docs])

    # Prepare messages for Ollama
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context_text}\n\nUser question: {query}"}
    ]

    # Query Ollama
    answer = query_ollama(messages)
    print("retriever:", retriever, "\n")
    print("Bot:", answer, "\n")
