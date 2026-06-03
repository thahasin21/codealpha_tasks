import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import pyperclip
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Language Translation Tool"
)

# ---------------- TITLE ----------------
st.title("🌍 AI Language Translation Tool")
st.markdown("Translate text between multiple languages with voice output.")

# ---------------- LANGUAGES ----------------
languages = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh-cn",
    "Arabic": "ar",
    "Malayalam": "ml",
    "Telugu": "te",
    "Kannada": "kn"
}

# ---------------- SESSION STATE ----------------
if "translated" not in st.session_state:
    st.session_state.translated = ""

# ---------------- LANGUAGE SELECTORS ----------------
col1, col2 = st.columns(2)

with col1:
    source_lang = st.selectbox(
        "Source Language",
        list(languages.keys()),
        index=0
    )

with col2:
    target_lang = st.selectbox(
        "Target Language",
        list(languages.keys()),
        index=1
    )

# ---------------- INPUT TEXT ----------------
text = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Type or paste text here..."
)

# ---------------- TRANSLATE BUTTON ----------------
if st.button("🔄 Translate", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            translated_text = GoogleTranslator(
                source=languages[source_lang],
                target=languages[target_lang]
            ).translate(text)

            st.session_state.translated = translated_text

        except Exception as e:
            st.error(f"Translation Error: {e}")

# ---------------- OUTPUT ----------------
if st.session_state.translated:

    st.subheader("✅ Translated Text")

    st.text_area(
        "Result",
        value=st.session_state.translated,
        height=150
    )

    # ---------------- ACTION BUTTONS ----------------
    col3, col4 = st.columns(2)

    with col3:
        if st.button("📋 Copy Translation"):
            try:
                pyperclip.copy(st.session_state.translated)
                st.success("Copied to clipboard!")
            except Exception as e:
                st.error(f"Copy failed: {e}")

    with col4:
        st.download_button(
            "⬇ Download Translation",
            data=st.session_state.translated,
            file_name="translation.txt",
            mime="text/plain"
        )

    # ---------------- TEXT TO SPEECH ----------------
    st.subheader("🔊 Listen to Translation")

    try:
        audio_file = "translation.mp3"

        tts = gTTS(
            text=st.session_state.translated,
            lang=languages[target_lang]
        )

        tts.save(audio_file)

        with open(audio_file, "rb") as audio:
            st.audio(audio.read(), format="audio/mp3")

    except Exception as e:
        st.warning(f"Audio not available for this language: {e}")

    finally:
        if os.path.exists("translation.mp3"):
            pass
