import re
import os
import streamlit as st
import requests
import json
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random

# For summary
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Function to securely load API key from environment variables
def load_api_key():
    return os.getenv('YOUR_API_KEY')  # Replace with your actual API key loading method

# Function to extract transcript and language code
def calculate_content_analysis_score(sentiment_score, keywords):
    # Normalize sentiment score to a 0-1 scale
    normalized_sentiment = (sentiment_score + 1) / 2  # Adjusting from [-1, 1] to [0, 1]

    # Calculate keyword score (e.g., the number of unique keywords)
    unique_keywords = len(set(keywords))  # Number of unique keywords extracted
    keyword_score = min(unique_keywords / 20, 1)  # Assume a max of 20 unique keywords for full score

    # Combine the scores with weights (you can adjust the weights as needed)
    content_score = (0.6 * normalized_sentiment) + (0.4 * keyword_score)
    
    return content_score
# proxies = [
#     "http://45.249.50.137:4153",
#     # "http://98.76.54.32:8080",
#     # "http://192.168.1.100:3128",
#  ]

def extract_video_id(url):
    """Extracts video ID from various YouTube URL formats using regex"""
    patterns = [
        r"youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)",  # Standard YouTube URL
        r"youtu\.be\/([a-zA-Z0-9_-]+)"              # Shortened YouTube URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(youtube_url):
    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.warning("⚠️ Invalid YouTube URL. Please enter a correct URL.")
        return None, None  # Ensure proper return type

    try:
        #proxy = {"http": random.choice(proxies), "https": random.choice(proxies)}
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_languages = {t.language: t.language_code for t in transcript_list}

        if not available_languages:
            st.warning("⚠️ No available transcripts for this video.")
            return None, None

        selected_language = list(available_languages.keys())[0]  # Auto-select first available language
        st.session_state["language_code"] = available_languages[selected_language]

        st.success(f"✅ Video is in `{selected_language}`.")

    except TranscriptsDisabled:
        st.error("❌ Transcripts are disabled for this video.")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error fetching transcript list: {str(e)}")
        return None, None

    # Fetch transcript in the selected language
    try:
        lang_code = st.session_state["language_code"]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
        transcript_text = "\n".join([part["text"] for part in transcript])

        return transcript_text, lang_code

    except Exception as e:
        st.error(f"⚠️ Error fetching transcript: {str(e)}")
        return None, None
    # try:
    #     transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, proxies=proxy)
    #     try:
    #         transcript = transcript_list.find_manually_created_transcript(['en'])
    #     except:
    #         generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
    #         if not generated_transcripts:
    #             raise Exception("No suitable transcript found. Generated transcripts are also not available.")
    #         transcript = generated_transcripts[0]
    #     full_transcript = " ".join([part['text'] for part in transcript.fetch()])
    #     language_code = transcript.language_code
    #     return full_transcript, language_code
    # except Exception as e:
    #     raise Exception(f"Transcript not found for the video. Error: {str(e)}")

# Sentiment Analysis on the transcript
def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Keyword Extraction using frequency-based analysis
def extract_keywords(text, num_keywords=10):
    auxiliary_words = set([
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'have', 'has', 
        'is', 'it', 'its', 'in', 'on', 'of', 'the', 'to', 'with', 'that', 'this', 'which', 
        'you', 'your', 'I', 'me', 'my', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 
        'their', 'we', 'us', 'our', 'not', 'no', 'do', 'does', 'did', 'can', 'could', 
        'will', 'would', 'should', 'may', 'might', 'must', 'shall', 'where', 'when', 
        'how', 'why', 'what', 'who'
    ])
    
    # Clean the text and filter out auxiliary words
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in auxiliary_words]
    word_counts = Counter(filtered_words)
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
    summary = summarizer(parser.document, 5)
    return "\n".join([str(sentence) for sentence in summary])

# Use OpenRouter API to get the content summary
def get_content_analysis(content):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",  # Replace with actual API key
        "Content-Type": "application/json"
    }

    data = {
        "model": "google/gemini-2.0-flash-thinking-exp:free",
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the following content and provide a content analysis score based on the following parameters:\n"
                           f"- 🔍🧐Clarity: Score out of 25 for how clear the content is.\n"
                           f"- 🏗️📊Structure: Score out of 25 for how well the content is structured.\n"
                           f"- ✔️🎯Relevance: Score out of 25 for how relevant the content is.\n"
                           f"- 🤝🔥Engagement: Score out of 25 for overall user engagement.\n"
                           f"Provide the total content analysis score (out of 100) by summing the scores for these four parameters.\n\n"
                           f"Content:\n{content}"
            }
        ]
    }

    # Send the request to the API
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        # Return the response content from the API
        return response.json()['choices'][0]['message']['content']
    else:
        # Handle error if the request fails
        st.error(f"Error: {response.status_code}")
        return None

def get_summary_from_api(content):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer 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",  # Replace with your actual API key
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "google/gemini-2.0-flash-thinking-exp:free", #meta-llama/llama-3.2-3b-instruct:free
        "messages": [
            {
                "role": "user",
                "content": f"Include the emojis along with the content and make it awesome.Include medium count of emojis.Summarize the following content and generate the flow of the content with headings and give the top 5 resources link available in the online for better understanding of this topics:\n\n{content} "
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error in generating summary: {response.status_code}, {response.text}"

# Function to fetch YouTube video statistics like view count
def get_video_stats(video_id):
    api_key = os.getenv('YOUTUBE_API_KEY')  # Ensure you have set the YouTube Data API key in your environment
    if not api_key:
        return "YouTube API key not found. Please set it in the environment."
    
    url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics&id={video_id}&key={api_key}"
    response = requests.get(url)
    stats = response.json()
    if 'items' in stats and stats['items']:
        view_count = stats['items'][0]['statistics']['viewCount']
        return view_count
    else:
        return "Stats not found"

def enhance_content(content, level_of_understanding):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",  # Replace with actual API key
        "Content-Type": "application/json"
    }

    # API request data with selected level of understanding
    data = {
        "model": "google/gemini-2.0-flash-thinking-exp:free",
        "messages": [
            {
                "role": "user",
                "content": f"Include the emojis along with the content,Structure it properly and pointed as needed and make it awesome.Make this content easy to understand for a {level_of_understanding} audience:\n\n{content}"
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"Error: {response.status_code}")
        return None

# Comment Section Analysis
def fetch_comments(video_url):
    yt = YouTube(video_url)
    print(yt)
    comments = yt.comments  # Fetch comments
    return comments

def navigate_to_section(section):
    st.session_state.section = section
# Main Streamlit App
def main():
    
    if "section" not in st.session_state:
        st.session_state.section = "analyzer"

    with st.sidebar:
        st.header("Navigate to:")
        if st.button("📹 YouTube Video Content Analyzer & Summarizer"):
            navigate_to_section("analyzer")
        if st.button("Content Enhancement"):
            navigate_to_section("enhancement")
        if st.button("Generate Wordcloud"):
            navigate_to_section("wordcloud")
        

    if st.session_state.section == "analyzer":
        st.title('🎬YouTube Video Content Analyzer & Summarizer')
        link = st.text_input('Enter the YouTube video URL:',placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
        user_query=st.text_input("Enter the query you want to search in the video:(optional)")
        
        value_yt=st.button("Analyze Video")
        if link:
            with st.expander("🎬Video you want to analyze", expanded=True):
                        st.video(link)
        if value_yt:
            if link:
                try:
                    
                    progress = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Fetch Transcript
                    status_text.text('Fetching the transcript...📜🔄')
                    progress.progress(25)

                    transcript, language_code = get_transcript(link)

                    # Step 2: Perform Sentiment Analysis
                    status_text.text('Performing sentiment analysis...🔍📊')
                    progress.progress(40)

                    sentiment_score = perform_sentiment_analysis(transcript)
                    st.write(f"Sentiment Score: {sentiment_score:.2f} (Range: -1 to 1)")

                    # Step 3: Extract Keywords & Calculate Content Analysis Score
                    status_text.text('Extracting keywords and analyzing content...🧠📑')
                    progress.progress(55)

                    keywords = extract_keywords(transcript)
                    content_analysis_score = calculate_content_analysis_score(sentiment_score, [kw[0] for kw in keywords])
                    st.write(f"### Content Analysis Score: {content_analysis_score:.2f} (Range: 0 to 1)")

                    # Step 4: Generate Summary
                    status_text.text('Generating summary...📝✨')
                    progress.progress(75)

                    api_summary = get_summary_from_api("1)give the youtube video title of this video,and theme of the video in 1 paragraph: 'mention as title : ,theme: "+link+"2)must thing(high priority):"+user_query+"3)"+transcript)
                    
                    # Display AI Summary in an Expander (Auto-expanded)
                    with st.expander("### AI Summary", expanded=True):
                        st.markdown(api_summary,unsafe_allow_html=True)

                    # Step 5: Calculate Final Content Score
                    status_text.text('Calculating final content analysis score...📊✅')
                    progress.progress(90)

                    content_anal_score = get_content_analysis(transcript)
                    
                    with st.expander("### Content Analyzed Score", expanded=True):
                        st.markdown(content_anal_score,unsafe_allow_html=True)

                    # Final Step: Mark Completion
                    progress.progress(100)
                    status_text.text('Analysis Complete! 🎉✅')

                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning('Please enter a valid YouTube link.')


    elif st.session_state.section == "enhancement":
        st.title("Content Enhancement")
        link = st.text_input('Enter the YouTube video URL:',placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
        # Dropdown for selecting the level of understanding
        user_query_enhancement=st.text_input("Customize here (optional) : ")
        level = st.selectbox(
            "Select the level of understanding:",
            ("5-year-old children (Pre-school)", "Mid school", "High school", "College (Undergraduate)")
        )
        value=st.button("Enhance it")
        if link:
            with st.expander("🎬Video you want to analyze", expanded=True):
                        st.video(link)
        if value:
            if link:
                
                content, language_code = get_transcript(link)
                if content:
                    with st.status("Enhancing content... Please wait!", expanded=True) as status:
                        enhanced_content = enhance_content("1)give the youtube video title of this video,and theme of the video in 1 paragraph: 'mention as title : ,theme: "+link+"2)must thing(high priority):"+user_query_enhancement+content, level)

                    if enhanced_content:
                        status.update(label="Enhancement completed!", state="complete", expanded=False)
                        
                        # Move expander outside `with status`
                        with st.expander("Enhanced Content", expanded=True):
                            st.write(enhanced_content)
                    else:
                        st.error("Failed to enhance content.")
                else:
                    st.error("Failed to fetch content. Please check the link.")


    elif st.session_state.section == "wordcloud":
        st.title("Generate wordcloud")
        link = st.text_input('Enter the YouTube video URL:',placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
        value_wordcloud=st.button("Get Wordcloud")
        if link:
            with st.expander("🎬Video you want to analyze", expanded=True):
                    st.video(link)
        if value_wordcloud:
            if link:
                transcript, language_code = get_transcript(link)
                st.write("### Word Cloud")
                generate_wordcloud(transcript)

if __name__ == "__main__":
    main()
