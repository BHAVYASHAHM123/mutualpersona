import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BartForConditionalGeneration, BartTokenizer
from datetime import datetime

# Load and preprocess the dataset
df = pd.read_csv('mutualfundhai.csv')
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['question'])

# Load the response generation model
response_generation_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
response_generation_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Function to extract answer from the dataset
def extract_answer(input_text):
    input_vector = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vector, question_vectors)
    most_similar_index = similarities.argmax()
    return df.loc[most_similar_index, 'answer'], df.loc[most_similar_index, 'tag']

# Function to generate response based on user input
def generate_response(input_text):
    answer, tag = extract_answer(input_text)

    if answer is not None:
        # Perform text summarization on the answer
        inputs = response_generation_tokenizer.encode(answer, return_tensors='pt')
        summary_ids = response_generation_model.generate(inputs, num_beams=4, max_length=50, min_length=10, early_stopping=True)
        generated_response = response_generation_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    else:
        generated_response = "Ok, I will get back to you on this topic as and when my higher authorities provide more information."

    return generated_response, tag


# Define the main app
def main():
    st.sidebar.title("Navigation")
    pages = ["Overview", "Chatbot", "Chat History"]
    choice = st.sidebar.selectbox("Go to", pages)

    if choice == "Overview":
        show_overview()
    elif choice == "Chatbot":
        chatbot()
    elif choice == "Chat History":
        show_chat_history()


# Function to display the overview page
def show_overview():
    st.title("Chatbot Overview")
    st.image('https://scitechdaily.com/images/Posh-Chatbots.jpg')
    # Add your content and images here


# Function to simulate a chatbot conversation
def chatbot():
    st.title("Chatbot")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat UI
    user_input = st.text_input("User:")
    if st.button("Send"):
        if user_input:
            with st.spinner("Generating response..."):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                chat_history = st.session_state.chat_history
                chat_history.append({"message": user_input, "sender": "user", "timestamp": timestamp})
                generated_response, _ = generate_response(user_input)
                chat_history.append({"message": generated_response, "sender": "bot", "timestamp": timestamp})
                st.session_state.chat_history = chat_history

    for chat in st.session_state.chat_history:
        if chat["sender"] == "user":
            st.write(f'<div style="display:flex; justify-content:flex-start; align-items:flex-start; margin-bottom:10px;">'
                     f'<div style="background-color:#DCF8C6; padding:10px; border-radius:10px;">'
                     f'<p style="font-weight:bold;">User | {chat["timestamp"]}</p>'
                     f'<p>{chat["message"]}</p>'
                     f'</div>'
                     f'</div>',
                     unsafe_allow_html=True)
        else:
            st.write(f'<div style="display:flex; justify-content:flex-end; align-items:flex-start; margin-bottom:10px;">'
                     f'<div style="background-color:#FFFFFF; padding:10px; border-radius:10px;">'
                     f'<p style="font-weight:bold;">Chatbot | {chat["timestamp"]}</p>'
                     f'<p>{chat["message"]}</p>'
                     f'</div>'
                     f'</div>'
                     f'<hr>',
                     unsafe_allow_html=True)

    # Scroll to the latest message
    scroll_latest_message()


def scroll_latest_message():
    # Use JavaScript to scroll to the latest message
    scroll_script = """
        <script>
        var div = document.getElementById("chat-container");
        div.scrollTop = div.scrollHeight;
        </script>
    """
    st.write(scroll_script, unsafe_allow_html=True)


# Function to display the chat history page
def show_chat_history():
    st.title("Chat History")
    # Fetch and display chat history data
    # You can use a database or file to store and retrieve the chat history data


# Run the app
if __name__ == "__main__":
    main()
