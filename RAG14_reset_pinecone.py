# Install Pinecone client if not already installed
# pip install pinecone-client

import pinecone
import uuid

# Initialize Pinecone
pinecone.init(
    api_key="YOUR_API_KEY",  # Replace with your Pinecone API key
    environment="us-east-1-aws"  # Change if your environment differs
)

# Name of the index
index_name = "sample-movies"

# Connect to the index
index = pinecone.Index(index_name)

# Generate 10 sample objects (vectors)
sample_objects = []
for i in range(10):
    obj_id = str(uuid.uuid4())  # Unique ID for each object
    vector = [float(i + j*0.1) for j in range(384)]  # Example 384-dim vector
    metadata = {
        "title": f"Movie {i+1}",
        "genre": "Action" if i % 2 == 0 else "Comedy",
        "year": 2000 + i
    }
    sample_objects.append((obj_id, vector, metadata))

# Upsert data to Pinecone
index.upsert(vectors=sample_objects)

print("Inserted 10 sample objects into Pinecone index:", index_name)

