import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
from langdetect import detect
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Language Translator",
    page_icon="🌍",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align: center;'>🌍 Language Translator</h1>",
    unsafe_allow_html=True
)

st.write("Translate text instantly between multiple languages.")

# ---------------- LANGUAGE OPTIONS ----------------
languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Japanese": "ja",
    "Chinese": "zh-CN",
    "Korean": "ko",
    "Tamil": "ta"
}

# ---------------- INPUT TEXT ----------------
text = st.text_area("Enter Text", height=150)

# ---------------- LANGUAGE SELECTION ----------------
col1, col2 = st.columns(2)

with col1:
    source_lang = st.selectbox(
        "Source Language",
        list(languages.keys())
    )

with col2:
    target_lang = st.selectbox(
        "Target Language",
        list(languages.keys())
    )

# ---------------- TRANSLATE BUTTON ----------------
if st.button("Translate 🚀"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            # Detect language
            detected_lang = detect(text)

            # Translate text
            translated = GoogleTranslator(
                source=languages[source_lang],
                target=languages[target_lang]
            ).translate(text)

            # Display output
            st.success("Translation Completed ✅")

            st.subheader("Translated Text")
            st.write(translated)

            # ---------------- COPY SECTION ----------------
            st.code(translated)

            # ---------------- TEXT TO SPEECH ----------------
            tts = gTTS(
                text=translated,
                lang=languages[target_lang],
                slow=False
            )

            audio_file = "translated_audio.mp3"
            tts.save(audio_file)

            st.audio(audio_file)

            # ---------------- DETECTED LANGUAGE ----------------
            st.info(f"Detected Input Language Code: {detected_lang}")

            # ---------------- TRANSLATION HISTORY ----------------
            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({
                "Original": text,
                "Translated": translated
            })

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- HISTORY ----------------
if "history" in st.session_state:

    st.subheader("📜 Translation History")

    for item in st.session_state.history[::-1]:
        st.write("🔹 Original:", item["Original"])
        st.write("🔸 Translated:", item["Translated"])
        st.markdown("---")