import streamlit as st
import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

download_nltk_resources()

faq_data = [
    {
        "question": "What language should I learn first?",
        "answer": "Python is often recommended for beginners because it is easy to learn and widely used."
    },
    {
        "question": "How long does it take to learn programming?",
        "answer": "It depends on your practice and goals. Basic programming can be learned within a few months."
    },
    {
        "question": "Do I need a computer science degree to get a job?",
        "answer": "No. Many developers get jobs through self-learning, certifications, and strong portfolios."
    },
    {
        "question": "What is the difference between front-end and back-end development?",
        "answer": "Front-end focuses on user interfaces while back-end handles servers, databases, and business logic."
    },
    {
        "question": "Do I need to be good at math to code?",
        "answer": "Most programming jobs require logical thinking more than advanced mathematics."
    },
    {
        "question": "How do I build a programming portfolio?",
        "answer": "Create projects, upload them to GitHub, and showcase them through a portfolio website."
    },
    {
        "question": "How can I prepare for a technical coding interview?",
        "answer": "Practice data structures, algorithms, coding problems, and mock interviews."
    },
    {
        "question": "Will AI replace programmers?",
        "answer": "AI helps programmers automate tasks, but human problem-solving skills remain important."
    },
    {
        "question": "What is Git and why do I need to use it?",
        "answer": "Git is a version control system used to track code changes and collaborate with others."
    },
    {
        "question": "What is the best IDE or text editor for coding?",
        "answer": "Popular choices include VS Code, PyCharm, IntelliJ IDEA, and Sublime Text."
    },
    {
        "question": "How do I debug my code effectively?",
        "answer": "Use debugging tools, read error messages carefully, and test code step by step."
    },
    {
        "question": "What is an API and how does it work?",
        "answer": "An API allows different software applications to communicate through requests and responses."
    },
    {
        "question": "What is Object-Oriented Programming (OOP)?",
        "answer": "OOP is a programming paradigm based on objects, classes, inheritance, and encapsulation."
    },
    {
        "question": "What is the difference between a framework and a library?",
        "answer": "A library is called by your code, while a framework calls your code."
    },
    {
        "question": "Why are data structures and algorithms important?",
        "answer": "They help solve problems efficiently and improve software performance."
    },
    {
        "question": "What is the difference between compiled and interpreted languages?",
        "answer": "Compiled languages are translated before execution, while interpreted languages run line by line."
    },
    {
        "question": "What does open-source mean?",
        "answer": "Open-source software makes its source code publicly available for modification and distribution."
    },
    {
        "question": "What is a null pointer exception and how do I fix it?",
        "answer": "It occurs when code tries to access an object that is null. Check for null values before use."
    },
    {
        "question": "What is the difference between a syntax error and a runtime error?",
        "answer": "Syntax errors occur before execution, while runtime errors happen during execution."
    },
    {
        "question": "How do I resolve a merge conflict in Git?",
        "answer": "Review conflicting changes, edit the file manually, and commit the resolved version."
    },
    {
        "question": "How does asynchronous programming work?",
        "answer": "It allows multiple tasks to run without blocking the main program flow."
    },
    {
        "question": "What is the difference between SQL and NoSQL databases?",
        "answer": "SQL databases use tables and schemas, while NoSQL databases offer flexible data structures."
    },
    {
        "question": "What is CI/CD and why is it used?",
        "answer": "CI/CD automates software testing and deployment processes."
    },
    {
        "question": "What is the difference between Python and JavaScript?",
        "answer": "Python is popular for backend development and AI, while JavaScript is mainly used for web development."
    },
    {
        "question": "What is TypeScript and why should I use it instead of JavaScript?",
        "answer": "TypeScript adds static typing to JavaScript, making large applications easier to maintain."
    }
]

df = pd.DataFrame(faq_data)

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

processed_questions = df["question"].apply(preprocess)

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    processed_questions
)
def get_response(user_query):

    processed_query = preprocess(user_query)

    query_vector = vectorizer.transform(
        [processed_query]
    )

    similarity_scores = cosine_similarity(
        query_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.20:
        return "Sorry, I couldn't find a matching FAQ."

    return df.iloc[best_match_index]["answer"]

st.set_page_config(
    page_title="Programming FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 Programming FAQ Chatbot")

st.write(
    "Ask any programming-related FAQ question."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


user_input = st.chat_input(
    "Type your question..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = get_response(user_input)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])
