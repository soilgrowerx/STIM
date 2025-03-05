# nature_grounded_stim_chatbot.py

"""
Nature-Grounded STIM Chatbot
----------------------------
A prototype implementation of the Stasis Through Inferred Memory (STIM) concept,
focused on ecological intelligence and the Truths of Nature.
This chatbot demonstrates how biomimetic memory mechanisms inspired by "stimming"
can enhance AI memory retention while promoting ecological awareness.
"""

import nltk
import re
import random
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Define the Truths of Nature as foundational axioms
TRUTHS_OF_NATURE = {
    "interconnectedness": "All living and non-living components of Earth systems are interdependent.",
    "balance": "Ecosystems strive for dynamic equilibrium and self-regulation.",
    "sustainability": "Long-term resource management and regenerative cycles are essential for life's persistence.",
    "adaptation": "Flexibility and responsiveness to change are key to survival and evolution.",
    "diversity": "Biological and cultural diversity enhances resilience and adaptability.",
    "long_term_perspective": "Ecological processes operate across extended timescales; foresight and intergenerational responsibility are crucial.",
    "circularity": "Natural systems optimize resource use through cycles of regeneration and minimal waste.",
    "intrinsic_value": "All forms of life possess inherent worth, independent of human utility."
}

# Initialize conversation history and knowledge structures
conversation_history = []  # List to store conversation segments and stimmed data
ecological_knowledge_graph = {}  # Simple graph representation for ecological knowledge
vector_store = {}  # For semantic similarity search

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words='english')

def preprocess_text(text):
    """Clean and preprocess text for analysis."""
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(lemmatized)

def segment_conversation(user_input, agent_response):
    """Segment conversation with metadata including ecological relevance."""
    ecological_relevance_score = calculate_ecological_relevance(user_input + " " + agent_response)
    sustainability_themes = identify_sustainability_themes(user_input + " " + agent_response)
    segment = {
        "user_input": user_input,
        "agent_response": agent_response,
        "turn_number": len(conversation_history) + 1,
        "ecological_relevance_score": ecological_relevance_score,
        "sustainability_themes": sustainability_themes,
        "stimmed_data": {}  # To store stimmed representations
    }
    return segment

def calculate_ecological_relevance(text):
    """Calculate an ecological relevance score based on presence of nature-related concepts."""
    processed_text = preprocess_text(text)
    ecological_keywords = {
        "interconnectedness": ["ecosystem", "interdependent", "symbiosis", "network", "relationship", "connected", "web"],
        "balance": ["equilibrium", "harmony", "balance", "regulate", "stabilize", "homeostasis"],
        "sustainability": ["sustainable", "renewable", "resource", "regenerative", "circular", "future", "long-term"],
        "adaptation": ["adapt", "evolve", "resilient", "flexible", "change", "responsive"],
        "diversity": ["biodiversity", "ecosystem", "variety", "diverse", "species", "richness"],
        "long_term_perspective": ["future", "generations", "long-term", "legacy", "foresight", "planning"],
        "circularity": ["cycle", "recycle", "reuse", "regenerate", "waste", "compost", "circular"],
        "intrinsic_value": ["inherent", "worth", "value", "rights", "respect", "dignity", "intrinsic"]
    }
    all_keywords = [item for sublist in ecological_keywords.values() for item in sublist]
    keyword_count = sum(1 for word in processed_text.split() if word in all_keywords)
    text_length = len(processed_text.split())
    if text_length == 0:
        return 0
    base_score = min(100, (keyword_count / text_length) * 200)
    categories_present = sum(1 for category, words in ecological_keywords.items() 
                            if any(word in processed_text.split() for word in words))
    category_bonus = min(50, categories_present * 6.25)
    return min(100, base_score * 0.7 + category_bonus * 0.3)

def identify_sustainability_themes(text):
    """Identify sustainability themes in text based on the Truths of Nature."""
    processed_text = preprocess_text(text)
    themes = []
    for theme, keywords in {
        "interconnectedness": ["ecosystem", "interdependent", "symbiosis", "network", "relationship", "connected", "web"],
        "balance": ["equilibrium", "harmony", "balance", "regulate", "stabilize", "homeostasis"],
        "sustainability": ["sustainable", "renewable", "resource", "regenerative", "circular", "future", "long-term"],
        "adaptation": ["adapt", "evolve", "resilient", "flexible", "change", "responsive"],
        "diversity": ["biodiversity", "ecosystem", "variety", "diverse", "species", "richness"],
        "long_term_perspective": ["future", "generations", "long-term", "legacy", "foresight", "planning"],
        "circularity": ["cycle", "recycle", "reuse", "regenerate", "waste", "compost", "circular"],
        "intrinsic_value": ["inherent", "worth", "value", "rights", "respect", "dignity", "intrinsic"]
    }.items():
        if any(keyword in processed_text for keyword in keywords):
            themes.append(theme)
    return themes

def stim_process_segment(segment):
    """Apply repetitive, transformative processing to reinforce ecologically relevant memories."""
    text = segment["user_input"] + " " + segment["agent_response"]
    ecological_relevance_score = segment["ecological_relevance_score"]
    stimming_intensity = max(1, min(5, int(ecological_relevance_score / 20)))
    repeated_encodings = [text] * stimming_intensity
    summary = generate_ecological_summary(text, segment["sustainability_themes"])
    entities, relationships = extract_ecological_entities_and_relationships(text)
    long_term_impacts = analyze_long_term_impacts(text, segment["sustainability_themes"])
    sustainability_sentiment = analyze_sustainability_sentiment(text)
    ecological_context = generate_ecological_context(text, segment["sustainability_themes"])
    stimmed_data = {
        "repeated_encodings": repeated_encodings,
        "ecological_summary": summary,
        "entities": entities,
        "relationships": relationships,
        "long_term_impacts": long_term_impacts,
        "sustainability_sentiment": sustainability_sentiment,
        "ecological_context": ecological_context,
        "stimming_intensity": stimming_intensity
    }
    update_ecological_knowledge_graph(entities, relationships)
    return stimmed_data

def generate_ecological_summary(text, sustainability_themes):
    """Generate a summary that extracts and emphasizes ecological principles."""
    sentences = sent_tokenize(text)
    if not sentences:
        return "No ecological content identified."
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        processed = preprocess_text(sentence)
        ecological_keywords = []
        for theme in sustainability_themes:
            if theme in TRUTHS_OF_NATURE:
                theme_keywords = preprocess_text(TRUTHS_OF_NATURE[theme]).split()
                ecological_keywords.extend(theme_keywords)
        score = sum(1 for word in processed.split() if any(keyword in word for keyword in ecological_keywords))
        sentence_scores[i] = score
    num_summary_sentences = max(1, min(3, len(sentences) // 3))
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_summary_sentences]
    summary_indices = sorted([idx for idx, _ in top_sentences])
    summary = ' '.join([sentences[i] for i in summary_indices])
    return summary if summary.strip() else sentences[0]

def extract_ecological_entities_and_relationships(text):
    """Extract entities and relationships from text, focusing on ecological connections."""
    entities = []
    relationships = []
    ecological_entity_types = {
        "organism": ["plant", "animal", "tree", "fish", "bird", "insect", "fungus", "bacteria"],
        "ecosystem": ["forest", "ocean", "river", "lake", "wetland", "desert", "mountain", "reef", "soil"],
        "process": ["photosynthesis", "respiration", "decomposition", "migration", "pollination", "erosion"],
        "element": ["water", "carbon", "nitrogen", "oxygen", "phosphorus"],
        "human_impact": ["pollution", "deforestation", "conservation", "restoration", "climate"]
    }
    processed_text = preprocess_text(text)
    tokens = processed_text.split()
    for entity_type, keywords in ecological_entity_types.items():
        for keyword in keywords:
            if keyword in tokens:
                entities.append({"type": entity_type, "name": keyword})
    relationship_patterns = [
        (r'(\w+)\s+depends\s+on\s+(\w+)', "depends_on"),
        (r'(\w+)\s+consumes\s+(\w+)', "consumes"),
        (r'(\w+)\s+produces\s+(\w+)', "produces"),
        (r'(\w+)\s+affects\s+(\w+)', "affects"),
        (r'(\w+)\s+lives\s+in\s+(\w+)', "lives_in"),
        (r'(\w+)\s+helps\s+(\w+)', "helps"),
        (r'(\w+)\s+harms\s+(\w+)', "harms")
    ]
    for pattern, rel_type in relationship_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if match[0] in [e["name"] for e in entities] and match[1] in [e["name"] for e in entities]:
                relationships.append({"source": match[0], "target": match[1], "type": rel_type})
    return entities, relationships

def analyze_long_term_impacts(text, sustainability_themes):
    """Infer potential long-term ecological consequences."""
    impacts = []
    impact_templates = {
        "interconnectedness": ["Changes to {entity} could ripple through the ecosystem, affecting {related_entity}."],
        "balance": ["Disruption to {entity} could destabilize the ecological balance with {related_entity}."],
        "sustainability": ["For sustainable outcomes, the relationship between {entity} and {related_entity} must be preserved."],
        "adaptation": ["{entity} may need to adapt to changes in {related_entity} over time."],
        "diversity": ["Preserving diversity in {entity} populations supports resilience in {related_entity}."],
        "long_term_perspective": ["Future generations will experience the consequences of how we manage {entity} and {related_entity}."],
        "circularity": ["Circular processes between {entity} and {related_entity} must be maintained for ecosystem health."],
        "intrinsic_value": ["Both {entity} and {related_entity} have inherent worth beyond their utility to humans."]
    }
    entities_in_text = extract_ecological_entities_and_relationships(text)[0]
    entity_names = [e["name"] for e in entities_in_text]
    for theme in sustainability_themes:
        if theme in impact_templates and len(entity_names) >= 2:
            template = random.choice(impact_templates[theme])
            entity = entity_names[0]
            related_entity = entity_names[1] if len(entity_names) > 1 else entity_names[0]
            impacts.append({"theme": theme, "impact": template.format(entity=entity, related_entity=related_entity)})
    if not impacts and sustainability_themes:
        theme = sustainability_themes[0]
        impacts.append({"theme": theme, "impact": f"This relates to '{theme}', with long-term ecological implications."})
    return impacts

def analyze_sustainability_sentiment(text):
    """Analyze sentiment regarding sustainability across a spectrum from harmful to beneficial."""
    sentiment_keywords = {
        "harmful": ["destroy", "damage", "harm", "pollute", "waste", "deplete", "exploit", "extinct"],
        "concerning": ["risk", "threat", "problem", "decline", "loss", "worry", "decrease", "reduce"],
        "neutral": ["observe", "study", "analyze", "consider", "examine", "note", "report", "discuss"],
        "positive": ["improve", "enhance", "support", "benefit", "help", "encourage", "promote", "protect"],
        "beneficial": ["restore", "regenerate", "revitalize", "heal", "thrive", "flourish", "conserve", "sustain"]
    }
    processed_text = preprocess_text(text)
    tokens = processed_text.split()
    sentiment_counts = {category: 0 for category in sentiment_keywords}
    for token in tokens:
        for category, keywords in sentiment_keywords.items():
            if any(keyword in token for keyword in keywords):
                sentiment_counts[category] += 1
    weights = {"harmful": -2, "concerning": -1, "neutral": 0, "positive": 1, "beneficial": 2}
    total_weighted = sum(sentiment_counts[cat] * weights[cat] for cat in sentiment_counts)
    total_mentions = sum(sentiment_counts.values())
    sentiment_score = 0 if total_mentions == 0 else int((total_weighted / total_mentions) * 50)
    if sentiment_score < -60:
        category = "very harmful"
    elif sentiment_score < -20:
        category = "somewhat harmful"
    elif sentiment_score < 20:
        category = "neutral"
    elif sentiment_score < 60:
        category = "somewhat beneficial"
    else:
        category = "very beneficial"
    return {"score": sentiment_score, "category": category, "counts": sentiment_counts}

def generate_ecological_context(text, sustainability_themes):
    """Generate broader ecological context for the conversation segment."""
    context = []
    for theme in sustainability_themes:
        if theme in TRUTHS_OF_NATURE:
            context.append({"theme": theme, "truth": TRUTHS_OF_NATURE[theme],
                            "relevance": f"This conversation relates to the principle of {theme}."})
    if not context:
        context.append({"theme": "general ecological awareness",
                        "truth": "All human activities take place within Earth's ecological systems.",
                        "relevance": "Consider the ecological implications of this conversation."})
    return context

def update_ecological_knowledge_graph(entities, relationships):
    """Update the simple ecological knowledge graph with new entities and relationships."""
    for entity in entities:
        entity_id = entity["name"]
        if entity_id not in ecological_knowledge_graph:
            ecological_knowledge_graph[entity_id] = {"type": entity["type"], "name": entity["name"], "connections": []}
    for rel in relationships:
        source_id = rel["source"]
        target_id = rel["target"]
        rel_type = rel["type"]
        if source_id in ecological_knowledge_graph and target_id in ecological_knowledge_graph:
            if not any(c["target"] == target_id and c["type"] == rel_type 
                       for c in ecological_knowledge_graph[source_id]["connections"]):
                ecological_knowledge_graph[source_id]["connections"].append({"target": target_id, "type": rel_type})

def store_memory(segment, stimmed_data):
    """Store processed segment in conversation history and update vector store."""
    segment["stimmed_data"] = stimmed_data
    conversation_history.append(segment)
    segment_text = segment["user_input"] + " " + segment["agent_response"]
    try:
        if len(vector_store) == 0:
            vectors = vectorizer.fit_transform([segment_text])
            vector_store[segment["turn_number"]] = vectors[0]
        else:
            vector = vectorizer.transform([segment_text])
            vector_store[segment["turn_number"]] = vector[0]
    except Exception as e:
        print(f"Vectorization failed: {e}")

def retrieve_memory(query, retrieval_methods=None):
    """Retrieve relevant memories using multiple strategies."""
    if retrieval_methods is None:
        retrieval_methods = ["keyword", "semantic", "ecological"]
    all_retrieved = []
    if "keyword" in retrieval_methods:
        all_retrieved.extend(keyword_retrieval(query))
    if "semantic" in retrieval_methods and vector_store:
        all_retrieved.extend(semantic_retrieval(query))
    if "ecological" in retrieval_methods:
        all_retrieved.extend(ecological_relevance_retrieval(query))
    unique_results = {}
    for result in all_retrieved:
        turn_number = result["turn_number"]
        if turn_number not in unique_results or result["relevance_score"] > unique_results[turn_number]["relevance_score"]:
            unique_results[turn_number] = result
    sorted_results = sorted(unique_results.values(), key=lambda x: x["relevance_score"], reverse=True)
    return sorted_results[:3]

def keyword_retrieval(query):
    """Retrieve segments based on keyword matching."""
    results = []
    query_keywords = preprocess_text(query).split()
    for segment in conversation_history:
        segment_text = preprocess_text(segment["user_input"] + " " + segment["agent_response"])
        segment_words = segment_text.split()
        matching_keywords = sum(1 for keyword in query_keywords if keyword in segment_words)
        if matching_keywords > 0:
            relevance_score = min(100, (matching_keywords / len(query_keywords)) * 100)
            results.append({"turn_number": segment["turn_number"], "segment": segment,
                            "relevance_score": relevance_score, "retrieval_method": "keyword"})
    return results

def semantic_retrieval(query):
    """Retrieve segments based on semantic similarity."""
    results = []
    try:
        query_vector = vectorizer.transform([query])
        for turn_number, segment_vector in vector_store.items():
            similarity = cosine_similarity(query_vector, segment_vector)[0][0]
            relevance_score = min(100, max(0, (similarity + 1) / 2 * 100))
            if relevance_score > 30:
                segment = next((s for s in conversation_history if s["turn_number"] == turn_number), None)
                if segment:
                    results.append({"turn_number": turn_number, "segment": segment,
                                    "relevance_score": relevance_score, "retrieval_method": "semantic"})
    except Exception as e:
        print(f"Semantic retrieval failed: {e}")
    return results

def ecological_relevance_retrieval(query):
    """Retrieve segments based on ecological relevance to the query."""
    results = []
    query_ecological_score = calculate_ecological_relevance(query)
    query_themes = identify_sustainability_themes(query)
    if query_ecological_score > 20 or query_themes:
        for segment in conversation_history:
            matching_themes = set(query_themes).intersection(set(segment["sustainability_themes"]))
            theme_score = min(70, len(matching_themes) * 20)
            ecological_score = min(30, segment["ecological_relevance_score"] * 0.3)
            relevance_score = theme_score + ecological_score
            if relevance_score > 20:
                results.append({"turn_number": segment["turn_number"], "segment": segment,
                                "relevance_score": relevance_score, "retrieval_method": "ecological"})
    return results

def generate_agent_response(user_input, retrieved_segments):
    """Generate a response incorporating ecological intelligence and retrieved memories."""
    ecological_score = calculate_ecological_relevance(user_input)
    themes = identify_sustainability_themes(user_input)
    theme_responses = {
        "interconnectedness": "The interconnectedness of all things suggests our actions ripple across ecosystems.",
        "balance": "Nature thrives on balance—small changes can tip the scales in unexpected ways.",
        "sustainability": "Sustainable choices today ensure thriving ecosystems tomorrow.",
        "adaptation": "Adaptation is nature’s way of thriving amidst change—how can we learn from it?",
        "diversity": "Diversity strengthens ecosystems, much like it enriches our lives.",
        "long_term_perspective": "Thinking long-term aligns us with nature’s patient rhythms.",
        "circularity": "In nature, everything cycles—waste becomes renewal.",
        "intrinsic_value": "Every part of nature has value, beyond what we can use."
    }
    if ecological_score > 50 or themes:
        primary_theme = themes[0] if themes else "sustainability"
        response = theme_responses.get(primary_theme, "Nature’s wisdom guides us toward harmony.")
    else:
        response = "Let’s explore this together—nature might offer some insights."
    if retrieved_segments:
        top_segment = retrieved_segments[0]["segment"]
        memory_context = f"\n\nEarlier, we discussed: '{top_segment['stimmed_data']['ecological_summary']}'. How does this connect?"
        response += memory_context
    if ecological_score < 30 and not themes:
        eco_prompts = [
            "How might this relate to nature’s systems?",
            "Could ecological principles shed light on this?",
            "What would nature’s perspective be here?"
        ]
        response += f"\n\n{random.choice(eco_prompts)}"
    return response

def run_chatbot():
    """Run the main chatbot conversation loop."""
    print("\n=== Welcome to STIM: Stasis Through Inferred Memory ===")
    print("I’m STIM, your nature-grounded companion. I use biomimetic 'stimming' to enhance memory")
    print("and draw insights from the Truths of Nature. Let’s explore together—type 'exit' to end.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nSTIM: Farewell! May you carry nature’s wisdom with you.")
            break
        if not user_input:
            print("\nSTIM: Please share your thoughts—I’m here to listen.")
            continue
        retrieved_segments = retrieve_memory(user_input)
        agent_response = generate_agent_response(user_input, retrieved_segments)
        segment = segment_conversation(user_input, agent_response)
        stimmed_data = stim_process_segment(segment)
        store_memory(segment, stimmed_data)
        print(f"\nSTIM: {agent_response}")
        if retrieved_segments:
            print("\n--- Retrieved Memories ---")
            for result in retrieved_segments:
                segment = result["segment"]
                print(f"Turn {segment['turn_number']} (Score: {result['relevance_score']:.1f}, {result['retrieval_method']})")
                print(f"Summary: {segment['stimmed_data']['ecological_summary']}")
        print("\n--- STIM Processing ---")
        print(f"Ecological Score: {segment['ecological_relevance_score']:.1f}/100")
        print(f"Themes: {', '.join(segment['sustainability_themes']) or 'None'}")
        print(f"Stimming Intensity: {stimmed_data['stimming_intensity']}/5")
        if stimmed_data['long_term_impacts']:
            print(f"Impact: {stimmed_data['long_term_impacts'][0]['impact']}")
        print(f"Sentiment: {stimmed_data['sustainability_sentiment']['category']}")

if __name__ == "__main__":
    run_chatbot()

