# simple_stim_chatbot.py

conversation_history = []  # List to store conversation segments and "stimmed" data

def segment_conversation(user_input, agent_response):
    segment = {
        "user_input": user_input,
        "agent_response": agent_response,
        "turn_number": len(conversation_history) + 1,
        "stimmed_data": {} # To store stimmed representations
    }
    return segment

def stim_process_segment(segment):
    text = segment["user_input"] + " " + segment["agent_response"]

    # Simplified "repetitive encoding" - just repeat the string a few times
    repeated_encodings = [text] * 3

    # Very basic summarization - just get first few words (for demo only!)
    summary = text[:20] + "..."

    stimmed_data = {
        "repeated_encodings": repeated_encodings,
        "summary": summary
    }
    return stimmed_data

def store_memory(segment, stimmed_data):
    segment["stimmed_data"] = stimmed_data
    conversation_history.append(segment)

def retrieve_memory(query):
    retrieved_segments = []
    query_keywords = query.lower().split() # Simple keyword extraction

    for segment in conversation_history:
        segment_text_lower = (segment["user_input"] + " " + segment["agent_response"]).lower()
        if any(keyword in segment_text_lower for keyword in query_keywords):
            retrieved_segments.append(segment)
    return retrieved_segments

# --- Main conversation loop (very basic example) ---
print("Simple STIM Chatbot Demo")
while True:
    user_query = input("User: ")
    if user_query.lower() == "exit":
        break

    # --- Agent response (replace with actual LLM integration for real chatbot) ---
    agent_response = "Agent response placeholder." # Replace this!

    segment = segment_conversation(user_query, agent_response)
    stimmed_data = stim_process_segment(segment)
    store_memory(segment, stimmed_data)

    retrieved_memory_segments = retrieve_memory(user_query) # Retrieve based on user query

    print("Agent:", agent_response)
    if retrieved_memory_segments:
        print("\nRetrieved Memory Segments (STIM Demo):")
        for retrieved_segment in retrieved_memory_segments:
            print(f"Turn {retrieved_segment['turn_number']}:")
            print("  User:", retrieved_segment['user_input'])
            print("  Agent:", retrieved_segment['agent_response'])
            print("  Stimmed Summary:", retrieved_segment['stimmed_data']['summary'])
            print("---")
    else:
        print("\nNo relevant memories retrieved (STIM Demo).")
