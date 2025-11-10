

# --------------------------
# 7️⃣ Delete vectors with short text (<15 chars)
# --------------------------

# Fetch all IDs that were upserted
all_ids = [str(elem["element_id"]) for elem in data]  # use same IDs as upserted
short_ids = []

# Check each vector's metadata
for vec_id in all_ids:
    result = index.fetch(ids=[vec_id])
    
    # Make sure vector exists (sometimes fetch returns empty if deleted)
    if vec_id in result['vectors']:
        text = result['vectors'][vec_id]['metadata']['text']
        if len(text.strip()) < 15:
            short_ids.append(vec_id)

# Delete vectors with short text
if short_ids:
    index.delete(ids=short_ids)
    print(f"✅ Deleted {len(short_ids)} vectors with text length < 15")
else:
    print("ℹ️ No vectors found with text length < 15")
