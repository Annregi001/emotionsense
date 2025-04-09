import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and label map
@st.cache_resource
def load_artifacts():
    model = load_model("emotion_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    return model, tokenizer, label_map

model, tokenizer, label_map = load_artifacts()

# Emoji map (adjust based on your actual label map)
emoji_map = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'love': '❤️',
    'surprise': '😲',
    'neutral': '😐',
    'disgust': '🤢',
    'shame': '😳',
    'guilt': '😔',
    'embarrassment': '🙈',
    'gratitude': '🙏',
    'confusion': '😕',
    'pride': '🦚'
}

# App UI
st.markdown("<h1 style='text-align: center;'>🎭 EmotionSense</h1>", unsafe_allow_html=True)
st.markdown("### 🔍 Enter a sentence to detect its emotion")

user_input = st.text_area("Type your sentence below:")

if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=100)

        # Predict
        prediction = model.predict(padded)[0]
        predicted_class = np.argmax(prediction)
        emotion = label_map.get(predicted_class, str(predicted_class))
        confidence = round(float(prediction[predicted_class]) * 100, 2)

        # Emoji lookup with fallback
        try:
            emoji = emoji_map.get(str(emotion).lower(), '')
        except:
            emoji = ''

        st.success(f"**Emotion:** {str(emotion).capitalize()} {emoji}")
        st.info(f"**Confidence:** {confidence}%")

        # Show all emotion probabilities (bar chart)
        emotion_scores = {
            label_map[i]: float(score) * 100 for i, score in enumerate(prediction)
        }
        df = pd.DataFrame(list(emotion_scores.items()), columns=["Emotion", "Confidence"])
        df = df.sort_values("Confidence", ascending=True)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(df["Emotion"], df["Confidence"], color="#ff4b4b")
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Emotion Prediction Breakdown")
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center')
        st.pyplot(fig)