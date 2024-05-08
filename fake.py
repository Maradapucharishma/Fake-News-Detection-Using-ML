import streamlit as st
import joblib
import numpy as np

# Load the trained model
with open('naive_bayes2.pkl', 'rb') as file:
    model = joblib.load(file)

# Define your Streamlit application
def app():
    st.image('https://thepathologist.com/fileadmin/_processed_/c/a/csm_0522-601-Profession---Timothy-Caulfield-1_Teaser_ecd3b5c8d8.png')
    st.title("Fake News Detection")

    # Collect user input for the features
    text = st.text_input('Enter your text here')
    
    # Add custom CSS to style the text only
    st.markdown("""
    <style>
    .result-text {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Make predictions using the loaded model
    if st.button("Detect"):
        prediction = model.predict([text])[0]  # Get the single prediction
        
        # Map the numerical prediction to sentiment label
        if prediction == 0:
            Result = "Fake News🤥📰"
        else:
            Result = "Real News🔍📰"

        # Display the results with larger and bold text
        st.markdown(f"<div class='result-text'>{Result}</div>", unsafe_allow_html=True)

# Run the Streamlit application
if __name__ == "__main__":
    app()
