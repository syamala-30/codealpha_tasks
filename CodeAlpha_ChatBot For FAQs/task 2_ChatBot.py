import nltk
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# ---------------------------
# FAQ DATA
# ---------------------------
faqs = [
    {"question": "What is Python?", "answer": "Python is a high-level programming language used for AI, web development, and data science."},
    {"question": "What is AI?", "answer": "AI stands for Artificial Intelligence, which enables machines to simulate human intelligence."},
    {"question": "What is machine learning?", "answer": "Machine learning is a branch of AI where systems learn from data automatically."},
    {"question": "What is deep learning?", "answer": "Deep learning is a subset of machine learning based on neural networks."},
    {"question": "What is an internship?", "answer": "An internship is a short-term work experience offered by companies to students."},
    {"question": "What is data science?", "answer": "Data science is the field of extracting insights from data."},
]

# ---------------------------
# TEXT CLEANING
# ---------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ---------------------------
# CHATBOT LOGIC
# ---------------------------
class FAQBot:
    def __init__(self, faqs):
        self.questions = [preprocess(f["question"]) for f in faqs]
        self.answers = [f["answer"] for f in faqs]

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_input):
        user_input = preprocess(user_input)
        user_vec = self.vectorizer.transform([user_input])

        similarity = cosine_similarity(user_vec, self.matrix)

        index = similarity.argmax()
        score = similarity[0][index]

        if score < 0.2:
            return "Sorry, I couldn't find a relevant answer. Try rephrasing your question."

        return self.answers[index]

# ---------------------------
# STREAMLIT UI SETUP
# ---------------------------
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")

st.title("🤖 FAQ Chatbot")
st.write("Ask any question and get instant answers!")

# Initialize bot
bot = FAQBot(faqs)

# Store chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------------------
# USER INPUT
# ---------------------------
user_input = st.text_input("Type your question here:")

if user_input:
    response = bot.get_response(user_input)

    # Save conversation
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", response))

# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------
for sender, message in st.session_state.chat:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")