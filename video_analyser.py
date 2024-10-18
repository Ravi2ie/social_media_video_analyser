# import os
# import streamlit as st
# from youtube_transcript_api import YouTubeTranscriptApi
# from textblob import TextBlob
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer

# # Function to extract transcript and language code
# def get_transcript(youtube_url):
#     video_id = youtube_url.split("v=")[-1]
    
#     try:
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

#         # First, try to get a manually created transcript
#         try:
#             transcript = transcript_list.find_manually_created_transcript(['en'])
#         except:
#             # If no manually created transcript, try to find a generated one
#             generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
#             if not generated_transcripts:
#                 raise Exception("No suitable transcript found. Generated transcripts are also not available.")
#             transcript = generated_transcripts[0]
        
#         # Extract the full transcript text
#         full_transcript = " ".join([part['text'] for part in transcript.fetch()])
#         language_code = transcript.language_code
#         return full_transcript, language_code
    
#     except Exception as e:
#         raise Exception(f"Transcript not found for the video. Error: {str(e)}")


# # Sentiment Analysis on the transcript
# def perform_sentiment_analysis(text):
#     blob = TextBlob(text)
#     sentiment_score = blob.sentiment.polarity  # -1 to 1 (negative to positive)
#     return sentiment_score

# # Keyword Extraction using frequency-based analysis
# def extract_keywords(text, num_keywords=10):
#     words = text.split()
#     word_counts = Counter(words)
#     common_words = word_counts.most_common(num_keywords)
#     return common_words

# # Generate Word Cloud
# def generate_wordcloud(text):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot(plt)

# # Generate Bar Plot for Keywords
# def plot_keywords(keywords):
#     words, counts = zip(*keywords)
#     sns.barplot(x=list(counts), y=list(words))
#     plt.xlabel('Frequency')
#     plt.ylabel('Keywords')
#     plt.title('Top Keywords')
#     st.pyplot(plt)

# # Summarize the transcript using Sumy LSA Summarizer
# def summarize_with_sumy(transcript):
#     parser = PlaintextParser.from_string(transcript, Tokenizer("english"))
#     summarizer = LsaSummarizer()
#     summary = summarizer(parser.document, 5)  # 5 sentences in the summary

#     return "\n".join([str(sentence) for sentence in summary])

# # Calculate the content score based on sentiment and keywords
# def calculate_content_score(sentiment_score, keywords):
#     keyword_density = sum([count for _, count in keywords]) / len(keywords)
#     score = (sentiment_score + 1) * 50 + keyword_density * 10  # Adjust weightings as needed
    
#     return max(0, round(score)) # Clamp score between 0 and 100

# # Main Streamlit App
# def main():
#     st.title('📹 YouTube Video Content Analyzer & Summarizer')
#     link = st.text_input('Enter the YouTube video URL:')

#     if st.button('Analyze Video'):
#         if link:
#             try:
#                 progress = st.progress(0)
#                 status_text = st.empty()

#                 status_text.text('Fetching the transcript...')
#                 progress.progress(25)

#                 # Extract transcript and language code
#                 transcript, language_code = get_transcript(link)

#                 # Perform sentiment analysis
#                 sentiment_score = perform_sentiment_analysis(transcript)
#                 st.write(f"Sentiment Score: {sentiment_score:.2f} (Range: -1 to 1)")

#                 # Extract keywords
#                 keywords = extract_keywords(transcript)
#                 st.write("*Top Keywords:*")
#                 st.write(keywords)

#                 # Visualize Word Cloud and Keyword Bar Plot
#                 st.write("### Word Cloud")
#                 generate_wordcloud(transcript)

#                 st.write("### Keyword Frequency Plot")
#                 plot_keywords(keywords)

#                 progress.progress(75)
#                 status_text.text('Generating summary...')

#                 # Generate summary using Sumy
#                 summary = summarize_with_sumy(transcript)
#                 st.markdown("### Summary:")
#                 st.markdown(summary)

#                 # Calculate content score
#                 content_score = calculate_content_score(sentiment_score, keywords)
#                 st.write(f"### Content Score: {content_score}/100")
#                 st.progress(content_score)

#                 status_text.text('Analysis Complete.')
#                 progress.progress(100)

#             except Exception as e:
#                 print(f"Error: {str(e)}")
#         else:
#             st.warning('Please enter a valid YouTube link.')

# if __name__ == "__main__":
#     main()

import re
import os
import streamlit as st
import requests
import json
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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

def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[-1]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except:
            generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
            if not generated_transcripts:
                raise Exception("No suitable transcript found. Generated transcripts are also not available.")
            transcript = generated_transcripts[0]
        full_transcript = " ".join([part['text'] for part in transcript.fetch()])
        language_code = transcript.language_code
        return full_transcript, language_code
    except Exception as e:
        raise Exception(f"Transcript not found for the video. Error: {str(e)}")

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
        "model": "meta-llama/llama-3.2-3b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the following content and provide a content analysis score based on the following parameters:\n"
                           f"- Clarity: Score out of 25 for how clear the content is.\n"
                           f"- Structure: Score out of 25 for how well the content is structured.\n"
                           f"- Relevance: Score out of 25 for how relevant the content is.\n"
                           f"- Engagement: Score out of 25 for overall user engagement.\n"
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
        "model": "meta-llama/llama-3.2-3b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": f"Summarize the following content and generate the flow of the content with headings:\n\n{content} "
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
        "model": "meta-llama/llama-3.2-3b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": f"Make this content easy to understand for a {level_of_understanding} audience:\n\n{content}"
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

                    # Calculate content analysis score
                    
                    # Extract keywords
                    keywords = extract_keywords(transcript)
                    # st.write("*Top Keywords:*")
                    # st.write(keywords)

                    content_analysis_score = calculate_content_analysis_score(sentiment_score, [kw[0] for kw in keywords])
                    st.write(f"### Content Analysis Score: {content_analysis_score:.2f} (Range: 0 to 1)")
                    

                    # Visualize Word Cloud and Keyword Bar Plot
                    # st.write("### Word Cloud")
                    # generate_wordcloud(transcript)

                    # st.write("### Keyword Frequency Plot")
                    # plot_keywords(keywords)

                    progress.progress(75)
                    status_text.text('Generating summary...')

                    # Generate summary using OpenRouter API
                    api_summary = get_summary_from_api(transcript)
                    st.markdown("### AI Summary:")
                    st.markdown(api_summary)

                    # Fetch video statistics
                    # video_id = link.split("v=")[-1]
                    # view_count = get_video_stats(video_id)
                    # st.write(f"View Count: {view_count}")

                    # Fetch and display comments
                    # st.write("### Comments:")
                    # comments = fetch_comments(link)
                    # if comments:
                    #     for comment in comments:
                    #         st.write(f"**User:** {comment.author}")
                    #         st.write(f"**Comment:** {comment.text}")
                    #         st.write("---")
                    # else:
                    #     st.write("No comments found.")

                    progress.progress(100)
                    status_text.text('Analysis Complete.')

                    content_anal_score=get_content_analysis(transcript)
                    st.markdown("### Content analysed score:")
                    st.markdown(content_anal_score)
                    status_text.text('Score is generated.')

                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning('Please enter a valid YouTube link.')

    elif st.session_state.section == "enhancement":
        st.title("Content Enhancement")
        link = st.text_input('Enter the YouTube video URL:')
        # Dropdown for selecting the level of understanding
        level = st.selectbox(
            "Select the level of understanding:",
            ("5-year-old children (Pre-school)", "Mid school", "High school", "College (Undergraduate)")
        )
        
        if st.button("Enhance it"):
            if link:
                content, language_code = get_transcript(link)
                if content:
                    # Enhance the content based on the selected level
                    enhanced_content = enhance_content(content, level)
                    if enhanced_content:
                        st.subheader("Enhanced Content:")
                        st.write(enhanced_content)
                    else:
                        st.error("Failed to enhance content.")
                else:
                    st.error("Please enter some content.")

    elif st.session_state.section == "wordcloud":
        st.title("Generate wordcloud")
        link = st.text_input('Enter the YouTube video URL:')
        if st.button('Show Wordcloud'):
            if link:
                transcript, language_code = get_transcript(link)
                st.write("### Word Cloud")
                generate_wordcloud(transcript)

if __name__ == "__main__":
    main()
