import streamlit as st
import joblib
import time 
st.set_page_config(
    page_title="NEWS CATEGORY CLASSIFIER",
    layout="centered" 
)
from streamlit_lottie import st_lottie
import requests

def load_lottie(url):
    r = requests.get(url)
    return r.json()

animation = load_lottie(
    "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
)

st_lottie(animation, height=250)


#model loading 
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# streamlit UI
st.title("***NEWS CATEGORY CLASSIFIER***")

news = st.text_area("Enter news")
if st.button("Predict"):
    
    # used if news box is empty
    if news.strip() == "":
        st.warning("⚠ Please enter a news article.")
        st.stop()
        
    # spinner animation 
    with st.spinner("Analyzing News..."):
        time.sleep(2)
        
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)
        news_category = prediction[0]
    st.success(f"Category: {news_category}")
    st.snow()
    
    if news_category == "Sports":
        st.image("images/sports_news.jpeg",use_container_width=True)
    elif news_category == "Business":
        st.image ("images/business_news.jpeg",use_container_width=True)
    elif news_category == "Science/Technology":       
        st.image("images\science_technology_news.jpeg",use_container_width=True)
                                                    # use for full cover centered page
        
    elif news_category == "World":
        st.image("images\world_news.jpeg",use_container_width=True)
        

    st.toast("Prediction Completed Successfully 🎉")  # top right corner notification