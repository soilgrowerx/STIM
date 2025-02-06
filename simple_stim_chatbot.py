# simple_stim_chatbot.py

conversation_history = []  # List to store conversation segments and "stimmed" data

import nltk
nltk.download('punkt')
nltk.download('stopwords') # pip install nltk


def segment_conversation(user_input, agent_response):
    segment = {
        "user_input": user_input,
        "agent_response": agent_response,
        "turn_number": len(conversation_history) + 1,
        "stimmed_data": {} # To store stimmed representations
    }
    return segment

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

def stim_process_segment(segment):
    text = segment["user_input"] + " " + segment["agent_response"]

    # Simplified "repetitive encoding" - just repeat the string a few times
    repeated_encodings = [text] * 3

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words
    words = [word_tokenize(sentence) for sentence in sentences]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [[word for word in sentence if word.lower() not in stop_words] for sentence in words]

    # Calculate word frequencies
    word_frequencies = {}
    for sentence in words:
        for word in sentence:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence):
            if word in word_frequencies:
                if i not in sentence_scores:
                    sentence_scores[i] = 0
                sentence_scores[i] += word_frequencies[word]

    # Get the top 3 scoring sentences
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]

    # Create the summary
    summary = ' '.join([sentences[i] for i in sorted(top_sentences)])

    stimmed_data = {
        "repeated_encodings": repeated_encodings,
        "summary": summary
    }
    return stimmed_data # generate more meaningful summaries using NLTK's summarization capabilities

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
