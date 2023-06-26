# Streamlit dependencies
import streamlit as st
import joblib
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import nltk


# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit"""

    # Set the title and subheader of the app
    st.title("TeamExcel Tweeet Classifier")
    st.subheader("Climate change tweet classification")

    # Create sidebar with selection box
    options = ["Information", "Data Analysis", "Prediction", "Team"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Build the "Information" page
    if selection == "Information":
        st.info("General Information")
        st.markdown("Our sentiment analysis machine learning model, specifically designed for climate change-related tweets, utilizes a combination of natural language processing and machine learning techniques. This powerful model enables us to gain deep insights into public sentiments by analyzing the extensive volume of Twitter data. By harnessing the capabilities of this model, we can adopt a data-driven approach to tackle climate change, make informed decisions, and promote effective solutions.")
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page
        st.subheader('Data Statistics')
        st.write(raw.describe())


    # Build the prediction page
    if selection == "Prediction":
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Tweet", "Type Here...")

        if st.button("Sentiment Check"):
            # Transform user input with the vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()

            # Load multiple models and allow the user to choose
            selected_model = open("resources/Linear_SVC.pkl", "rb")
            predictor = joblib.load(selected_model)
            prediction = predictor.predict(vect_text)

            # Mapping of sentiment labels to human-readable descriptions
            sentiment_mapping = {
                -1: "Anti Climate Change",
                0: "Neutral About Climate Change Belief",
                1: "Pro Climate Change",
                2: "News (Informative Broadcast)"
            }
            prediction_description = sentiment_mapping.get(int(prediction[0]))

            # Display the prediction and its description
            st.success("Prediction: {}".format(prediction_description))

            # Additional information based on sentiment
            if prediction[0] == -1:
                st.warning("This tweet indicates a negative sentiment towards climate change.")
            elif prediction[0] == 0:
                st.info("This tweet indicates a neutral sentiment towards climate change.")
            elif prediction[0] == 1:
                st.success("This tweet indicates a positive sentiment towards climate change.")
            elif prediction[0] == 2:
                st.info("This tweet is news or an informative broadcast related to climate change.")
    # Building the EDA page
    if selection == "Data Analysis":
        st.info("Exploratory Data Analysis")
        st.subheader("Sentiment Analysis")
        groups = raw.groupby(by='sentiment').count().message
        anti = groups[-1]
        neu = groups[0]
        pro = groups[1]
        news = groups[2]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Anti', 'Neutral', 'Pro', 'News'],
            y=[anti, neu, pro, news],
            marker_color='indianred',
            width=[0.4, 0.4],
            text=[f'ANTI: {anti}', f'NEU: {neu}', f'PRO: {pro}', f'NEWS: {news}']))
        fig.update_layout(title='Frequency of Sentiments', title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Tweets Analysis")
        nltk.download('punkt')
        words = raw['message'].apply(nltk.word_tokenize)
        all_words = [word for sublist in words for word in sublist]
        frequency_dist = nltk.FreqDist(all_words)
        temp = pd.DataFrame(frequency_dist.most_common(20), columns=['word', 'count'])
        fig = px.bar(temp, x='word', y='count', title='Top words')
        fig.update_layout(xaxis_tickangle=90)
        st.plotly_chart(fig, use_container_width=True)
        df_neutral = raw[raw['sentiment']==0]
        df_pro = raw[raw['sentiment']==1]
        df_news = raw[raw['sentiment']==2]
        tweet_All = " ".join(review for review in raw.message)
        tweet_anti = " ".join(review for review in df_anti.message)
        tweet_neutral = " ".join(review for review in df_neutral.message)
        tweet_pro = " ".join(review for review in df_pro.message)
        tweet_news = " ".join(review for review in df_news.message)
        
        
        fig, ax = plt.subplots(5, 1, figsize  = (800/30, 600/30), dpi=30)
        # Create and generate a word cloud image:
        wordcloud_ALL = WordCloud(width=400, height=300, max_font_size=50, max_words=100,
                                  background_color="white").generate(tweet_All)
        wordcloud_anti = WordCloud(width=400, height=300, max_font_size=50, max_words=100,
                                   background_color="white").generate(tweet_anti)
        wordcloud_neutral = WordCloud(width=400, height=300, max_font_size=50, max_words=100,
                                      background_color="white").generate(tweet_neutral)
        wordcloud_pro = WordCloud(width=400, height=300, max_font_size=50, max_words=100,
                                  background_color="white").generate(tweet_pro)
        wordcloud_news = WordCloud(width=400, height=300, max_font_size=50, max_words=100,
                                   background_color="white").generate(tweet_news)
        # Display the generated image:
        ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
        ax[0].set_title('All Tweets', fontsize=20)
        ax[0].axis('off')
        ax[1].imshow(wordcloud_anti, interpolation='bilinear')
        ax[1].set_title('Tweets under ANTI Class',fontsize=20)
        ax[1].axis('off')
        ax[2].imshow(wordcloud_neutral, interpolation='bilinear')
        ax[2].set_title('Tweets under NEUTRAL Class',fontsize=20)
        ax[2].axis('off')
        ax[3].imshow(wordcloud_pro, interpolation='bilinear')
        ax[3].set_title('Tweets under PRO Class',fontsize=20)
        ax[3].axis('off')
        ax[4].imshow(wordcloud_news, interpolation='bilinear')
        ax[4].set_title('Tweets under NEWS Class',fontsize=20)
        ax[4].axis('off')
        # Show the plot
        st.pyplot(fig)
    if selection == "Team":
        st.subheader('Team Excel')
        st.text("""Introducing Team Excel, an innovative AI company at the forefront of transforming industries globally. With a team of skilled experts, we harness the power of advanced algorithms, machine learning, and natural language processing to pioneer groundbreaking solutions. Whether it's personalized virtual assistants, data analytics, or automation, our goal is to empower businesses to flourish in the digital age. Come and join us on this transformative journey towards success.""")
        st.text("""Ayodele Marcus\t\tTeam Lead\n\nMacDaniel Ogechukwu\tTech Lead\n\nOnyeka Ekese\t\tAdmin Lead\n\nChege Kennedy\t\tAss. Project\n\nGideon Odekina\t\tAss. Tech\n\nTolulope Toluwade\tAss. Admin""")
        

# Required to let Streamlit instantiate our web app.
if __name__ == "__main__":
    main()
