# 🎬 Social Media Video Analyser

An **AI-powered YouTube Video Analyzer & Summarizer** built with **Streamlit**, designed to analyze any YouTube video and generate detailed insights.  
Paste a YouTube link, and get **transcripts, sentiment analysis, keyword insights, word clouds, summaries, content scores, and even enhanced explanations** tailored to different levels of understanding.

---

## 📂 Project Structure

    ├── .devcontainer
        └── devcontainer.json
    ├── README.md
    ├── requirements.txt
    ├── video_analyser.ipynb
    └── video_analyser.py


---

## ℹ️ About the Project

This project makes **YouTube video analysis easy, interactive, and intelligent**.  
It leverages AI and NLP techniques to extract meaningful insights from video transcripts, helping content creators, educators, and analysts understand video content quickly.

Key functionalities:

- Auto-fetch video transcripts
- Sentiment analysis of video content
- Keyword extraction & visualization
- WordCloud generation
- AI-powered summarization & enhancement
- Content analysis scoring based on clarity, structure, relevance, and engagement
- Interactive, user-friendly dashboard via Streamlit

---

## 🔧 Detailed Working

1. **Input Video URL** – Enter any valid YouTube link.  
2. **Transcript Extraction** – Automatically fetches subtitles using `youtube-transcript-api`.  
3. **Sentiment Analysis** – Calculates positivity/negativity using `TextBlob`.  
4. **Keyword Analysis** – Frequency-based keyword extraction with bar plot visualization.  
5. **WordCloud Generation** – Interactive word cloud of most frequent words.  
6. **AI Summarization & Enhancement** – Uses **OpenRouter AI API** to generate structured summaries with headings, emojis, and references.  
7. **Content Analysis Score** – Combines sentiment and keyword metrics to provide a score reflecting clarity, structure, relevance, and engagement.  
8. **Interactive Navigation** – Streamlit sidebar allows switching between **Analyzer**, **Content Enhancement**, and **WordCloud** sections.

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Core Libraries**:  
  - `pytube` – Video download & metadata  
  - `youtube-transcript-api` – Fetch video transcripts  
  - `textblob` – Sentiment analysis  
  - `wordcloud`, `matplotlib`, `seaborn` – Visualization  
  - `sumy` – Text summarization (LSA method)  
  - `requests` – API calls to OpenRouter AI  
- **APIs**: OpenRouter AI for summarization and content enhancement

---

## 🚀 How to Clone

    ```bash
        git clone https://github.com/Ravi2ie/social_media_video_analyser.git
        cd social_media_video_analyser


---

## 🛠 How to Run

### 1. Create a virtual environment (optional but recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
2. Install dependencies
    ```bash
    pip install -r requirements.txt
3. Set your API keys
    ```bash
    export YOUTUBE_API_KEY="your_youtube_api_key"
    export YOUR_API_KEY="your_openrouter_api_key"
4. Launch the Streamlit app
    ```bash
    streamlit run video_analyser.py
## 🖥️ Features

- 📹 AI-driven analysis of YouTube video content  
- ✨ Sentiment scoring and keyword extraction  
- ☁️ WordClouds & bar plots for visual insights  
- 📝 Dynamic summaries with headings, emojis, and online resources  
- 🎯 Content enhancement for multiple audience levels  
- 🌟 Interactive and user-friendly dashboard  

---

## 🔮 Future Enhancements

- Support for other social media platforms (Instagram, Twitter)  
- Multi-language summaries  
- Advanced NLP metrics (topic modeling, emotion detection)  
- Export analysis as PDF/CSV reports  
- Improved error handling and logging  

---

## 🤝 Contributing

Contributions are welcome!  

1. **Fork the repository**  

2. **Create a new branch**  

    ```bash
    git checkout -b feature-name
3. **Commit your changes**

      ```bash
    git commit -m "Add new feature"


4. **Push to the branch**

      ```bash
    git push origin feature-name


5. **Create a Pull Request**
