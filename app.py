import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer once, cache resource for performance
@st.cache_resource
def load_model():
    model_name = "qwen2.5-0.5B_finetuned_mentalhealth"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

tokenizer, model = load_model()
device = model.device

TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
MAX_NEW_TOKENS = 150
SYSTEM_PROMPT = "You are a compassionate mental health support chatbot. Respond empathetically to user messages."

GREETING_RESPONSES = [
    "Hello! I'm here to listen. How are you feeling today?",
    "Hi there! I'm here if you need someone to talk to.",
    "Hey! I'm glad you're here. How are things going for you?"
]

st.title("🧠 Mental Health Support Chatbot")
st.write("I'm here to support you. Type below to start a conversation.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("How are you feeling today?")

if user_input:
    st.session_state.chat_history.append(("User", user_input))

    if user_input.lower() in ["hi", "hello", "hey"]:
        bot_response = GREETING_RESPONSES[0]
    else:
        full_prompt = SYSTEM_PROMPT + "\n"
        for role, msg in st.session_state.chat_history:
            if role == "User":
                full_prompt += f"User: {msg}\n"
            else:
                full_prompt += f"Bot: {msg}\n"
        full_prompt += "Bot:"

        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        if "Bot:" in decoded:
            bot_response = decoded.split("Bot:")[-1].strip()
        else:
            bot_response = decoded.strip()

    st.session_state.chat_history.append(("Bot", bot_response))

for role, msg in st.session_state.chat_history:
    if role == "User":
        with st.chat_message("user"):
            st.markdown(msg)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)
