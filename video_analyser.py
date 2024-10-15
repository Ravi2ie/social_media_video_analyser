import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Function to extract transcript and language code
def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    try:
        transcript = transcript_list.find_manually_created_transcript()
        language_code = transcript.language_code
    except:
        try:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            transcript = generated_transcripts[0]
            language_code = transcript.language_code
        except:
            raise Exception("No suitable transcript found.")

    full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    return full_transcript, language_code

# Sentiment Analysis on the transcript
def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # -1 to 1 (negative to positive)
    return sentiment_score

# Keyword Extraction using frequency-based analysis
def extract_keywords(text, num_keywords=10):
    words = text.split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(num_keywords)
    return common_words

# Generate Word Cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Generate Bar Plot for Keywords
def plot_keywords(keywords):
    words, counts = zip(*keywords)
    sns.barplot(x=list(counts), y=list(words))
    plt.xlabel('Frequency')
    plt.ylabel('Keywords')
    plt.title('Top Keywords')
    st.pyplot(plt)

# Summarize the transcript using Sumy LSA Summarizer
def summarize_with_sumy(transcript):
    parser = PlaintextParser.from_string(transcript, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 5)  # 5 sentences in the summary

    return "\n".join([str(sentence) for sentence in summary])

# Calculate the content score based on sentiment and keywords
def calculate_content_score(sentiment_score, keywords):
    keyword_density = sum([count for _, count in keywords]) / len(keywords)
    score = (sentiment_score + 1) * 50 + keyword_density * 10  # Adjust weightings as needed
    
    return max(0, round(score)) # Clamp score between 0 and 100

# Main Streamlit App
def main():
    st.title('📹 YouTube Video Content Analyzer & Summarizer')
    link = st.text_input('Enter the YouTube video URL:')

    if st.button('Analyze Video'):
        if link:
            try:
                progress = st.progress(0)
                status_text = st.empty()

                status_text.text('Fetching the transcript...')
                progress.progress(25)

                # Extract transcript and language code
                transcript, language_code = get_transcript(link)

                # Perform sentiment analysis
                sentiment_score = perform_sentiment_analysis(transcript)
                st.write(f"Sentiment Score: {sentiment_score:.2f} (Range: -1 to 1)")

                # Extract keywords
                keywords = extract_keywords(transcript)
                st.write("*Top Keywords:*")
                st.write(keywords)

                # Visualize Word Cloud and Keyword Bar Plot
                st.write("### Word Cloud")
                generate_wordcloud(transcript)

                st.write("### Keyword Frequency Plot")
                plot_keywords(keywords)

                progress.progress(75)
                status_text.text('Generating summary...')

                # Generate summary using Sumy
                summary = summarize_with_sumy(transcript)
                st.markdown("### Summary:")
                st.markdown(summary)

                # Calculate content score
                content_score = calculate_content_score(sentiment_score, keywords)
                st.write(f"### Content Score: {content_score}/100")
                st.progress(content_score)

                status_text.text('Analysis Complete.')
                progress.progress(100)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning('Please enter a valid YouTube link.')

if __name__ == "__main__":
    main()