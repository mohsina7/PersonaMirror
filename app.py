import streamlit as st
import pandas as pd
from datetime import datetime
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Download VADER sentiment lexicon
nltk.download('vader_lexicon')

# === Load distilgpt2 Model (Small, Free, Offline) ===
@st.cache_resource(show_spinner=False)
def load_reflection_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_reflection_model()

# === Build Prompt for Reflection Generation ===
def build_reflection_prompt(mood, emotion, journal_text):
    prompt = (
        f"You are a kind and insightful AI assistant.\n"
        f"The user wrote this journal entry:\n\"{journal_text}\"\n\n"
        f"The user's mood is: {mood}.\n"
        f"The user's emotion is: {emotion}.\n"
        f"Please provide a short, personalized, motivational or encouraging reflection "
        f"that helps the user gain positive insight.\n\nReflection:"
    )
    return prompt

# === Generate Reflective Insight ===
def generate_reflective_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reflection = generated_text[len(prompt):].strip()
    return reflection

# === Load Emotion Classification Model ===
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_classifier = load_emotion_model()
sia = SentimentIntensityAnalyzer()

# === Data Path and Setup ===
DATA_PATH = "data/journal_entries.csv"
os.makedirs("data", exist_ok=True)

# === Session State Initialization ===
if "journal_text" not in st.session_state:
    st.session_state.journal_text = ""
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None
if "trigger_prefill" not in st.session_state:
    st.session_state.trigger_prefill = False

# === Utility Functions ===
def predict_mood(text):
    if not text.strip():
        return "Neutral"
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def detect_emotion(text):
    if not text.strip():
        return "neutral"
    results = emotion_classifier(text)
    if results and isinstance(results[0], list):
        top_emotion = max(results[0], key=lambda x: x['score'])
        return top_emotion['label'].lower()
    return "neutral"

def load_entries():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        else:
            df["Timestamp"] = pd.NaT
        return df
    else:
        return pd.DataFrame(columns=["Timestamp", "Entry", "Mood", "Emotion"])

def save_entries(df):
    df = df.dropna(subset=["Timestamp"]).copy()
    df["Timestamp"] = df["Timestamp"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f") if pd.notna(x) else ""
    )
    df.to_csv(DATA_PATH, index=False)

def submit_entry():
    text = st.session_state.journal_text.strip()
    if not text:
        st.warning("Cannot submit empty journal entry.")
        return

    mood = predict_mood(text)
    emotion = detect_emotion(text)
    timestamp = datetime.now()

    df = load_entries()

    if st.session_state.edit_index is not None:
        idx = st.session_state.edit_index
        if idx in df.index:
            df.at[idx, "Entry"] = text
            df.at[idx, "Mood"] = mood
            df.at[idx, "Emotion"] = emotion
            df.at[idx, "Timestamp"] = timestamp
        st.session_state.edit_index = None
    else:
        new_row = pd.DataFrame([[timestamp, text, mood, emotion]],
                               columns=["Timestamp", "Entry", "Mood", "Emotion"])
        df = pd.concat([df, new_row], ignore_index=True)

    save_entries(df)
    st.session_state.journal_text = ""
    st.session_state.trigger_prefill = False
    st.rerun()

def delete_entry(index):
    df = load_entries()
    if index in df.index:
        df = df.drop(index).reset_index(drop=True)
        save_entries(df)
        st.success("Entry deleted.")
        st.rerun()

# === Prefill for Edit ===
if st.session_state.trigger_prefill and st.session_state.edit_index is not None:
    df = load_entries()
    idx = st.session_state.edit_index
    if idx in df.index:
        st.session_state.journal_text = df.at[idx, "Entry"]
    st.session_state.trigger_prefill = False

# === UI ===
st.title("📝 PersonaMirror: Journal & Insight Tracker")

# === Journal Entry Form ===
with st.form("journal_form"):
    st.text_area("Write your journal entry:", key="journal_text", height=150)
    st.form_submit_button("Submit", on_click=submit_entry)

# === Display Entries ===
df = load_entries()

if df.empty:
    st.info("No journal entries yet.")
else:
    st.subheader("📜 All Journal Entries")
    df = df.sort_values("Timestamp", ascending=False).copy()

    for i in df.index:
        row = df.loc[i]
        timestamp = row["Timestamp"]
        entry = row["Entry"]
        mood = row["Mood"]
        emotion = row["Emotion"]

        with st.expander(f"🕒 {timestamp} | Mood: {mood} | Emotion: {emotion}"):
            st.markdown(entry)

            if st.button("🤖 Get Reflective Insight", key=f"reflect_{i}"):
                prompt = build_reflection_prompt(mood, emotion, entry)
                with st.spinner("Generating insightful reflection..."):
                    reflection = generate_reflective_text(prompt)
                st.markdown(f"**Reflection:** {reflection}")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✏️ Edit", key=f"edit_{i}"):
                    st.session_state.edit_index = i
                    st.session_state.trigger_prefill = True
                    st.rerun()
            with col2:
                if st.button("❌ Delete", key=f"delete_{i}"):
                    delete_entry(i)

    # === Calendar Heatmap ===
    st.subheader("📆 Calendar View of Entries")
    calendar_df = df.copy()
    calendar_df["Date"] = calendar_df["Timestamp"].dt.date
    daily_counts = calendar_df.groupby("Date").size().reset_index(name="Entries")
    daily_counts["Day"] = 1  # Dummy Y-axis

    fig = px.density_heatmap(
        daily_counts,
        x="Date",
        y="Day",
        z="Entries",
        color_continuous_scale="Viridis",
        labels={'x': 'Date', 'y': '', 'z': 'Entries'},
    )

    fig.update_layout(
        height=200,
        yaxis_visible=False,
        yaxis_showticklabels=False,
        yaxis=dict(showgrid=False),
        xaxis=dict(tickangle=45)
    )

    st.plotly_chart(fig, use_container_width=True)

    # === Mood & Emotion Trends ===
    st.subheader("📊 Mood & Emotion Trends")
    trends_df = df.copy()
    trends_df["Date"] = trends_df["Timestamp"].dt.date
    mood_counts = trends_df.groupby(["Date", "Mood"]).size().unstack(fill_value=0)
    emotion_counts = trends_df.groupby(["Date", "Emotion"]).size().unstack(fill_value=0)

    st.markdown("**Mood Trends**")
    st.line_chart(mood_counts)

    st.markdown("**Emotion Trends**")
    st.line_chart(emotion_counts)
