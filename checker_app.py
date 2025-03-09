import streamlit as st
import openai
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import yaml

# Load OpenAI API
# Load API key from secrets.yaml
with open("secrets.yaml", "r") as file:
    secrets = yaml.safe_load(file)

openai.api_key = secrets["OPENAI_API_KEY"]

# Initialize translator
translator = Translator()

# Function to get ChatGPT response
def get_chatgpt_response(prompt, lang="en"):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Get embeddings for similarity calculation
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Streamlit UI

############################
# Title and description
############################

st.title("🌍 Cross-lingual AI hallucination checker")
st.markdown(
    """
This application helps detect potential AI hallucinations specifically for healthcare questions asked in Ewe. 

**How it works:**
- Enter your healthcare-related question in Ewe.
- The app translates your question into English.
- It generates AI responses in both Ewe and English.
- Semantic similarity between both responses is evaluated.
- A low similarity indicates potential AI hallucination, prompting caution.

""")

# User input in Ewe (or another low-resource language)
user_input_ewe = st.text_area("Enter your healthcare question (Ewe):")

if st.button("Check Response"):
    with st.spinner("Processing..."):
        # Translate user input to English
        translation = translator.translate(user_input_ewe, dest="en")
        user_input_en = translation.text

        # Get responses from ChatGPT
        response_ewe = get_chatgpt_response(user_input_ewe)
        response_en = get_chatgpt_response(user_input_en)

        # Evaluate similarity between responses
        embed_ewe = np.array(get_embedding(response_ewe)).reshape(1, -1)
        embed_en = np.array(get_embedding(response_en)).reshape(1, -1)

        similarity = cosine_similarity(embed_ewe, embed_en)[0][0]

        # Define similarity threshold
        threshold = 0.85

        # Display results based on similarity
        if similarity >= threshold:
            st.success("✅ Confident Response (High Similarity)")
            st.write(f"**AI Response (Ewe):** {response_ewe}")
        else:
            st.warning("⚠️ Uncertain Response Detected")
            st.write("I'm not certain, but here's my closest response:")
            st.write(f"**Ewe:** {response_ewe}")
            st.write(f"**English:** {response_en}")

        st.info(f"Similarity Score: {similarity:.2f}")
