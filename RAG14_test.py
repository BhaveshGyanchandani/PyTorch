import requests
import json

url = "http://localhost:11434/api/chat"

payload = {
    "model": "llava:7b-v1.6-mistral-q4_0",
    "messages": [
        {"role": "user", 
        "content": "whats python language"}
    ]
}

response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    print("Streaming response from Ollama:\n")
    for line in response.iter_lines(decode_unicode=True):
        if line:  # ignore empty lines
            try:
                # parse each line as JSON
                json_data = json.loads(line)

                # extract and print assistant's message
                if "message" in json_data and "content" in json_data["message"]:
                    print(json_data["message"]["content"], end="")

            except json.JSONDecodeError:
                print(f"\nFailed to parse line: {line}")

    print("\n")  # print final newline
else:
    print(f"Request failed: {response.status_code} - {response.text}")
    
    
#ollama run llava:7b-v1.6-mistral-q4_0  on terminal 


# OR
# import ollama

# # Initialize the Ollama client
# client = ollama.Client()

# # Define the model and the input prompt
# model = "llava:7b-v1.6-mistral-q4_0"  # Replace with your model name
# prompt = "What is Python?"

# # Send the query to the model
# response = client.generate(model=model, prompt=prompt)

# # Print the response from the model
# print("Response from Ollama:")
# print(response.response)


#check the playlist for that 1 video for TEMPERATURE 1 or 0 too
