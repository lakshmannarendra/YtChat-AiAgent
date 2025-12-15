# YtChat – YouTube Video AI AGENT

YtChat lets you chat with an AI about any YouTube video. It uses two Gemini AIs: one for analyzing video transcripts, and one for summarizing answers for users. The backend uses PostgreSQL for data storage and BrightData for transcript scraping. The frontend is built with React and Vite.

---

## Tech Stack

- **Frontend:** React, Vite, TypeScript, CSS
- **Backend:** Node.js, Express, PostgreSQL, Gemini AI, BrightData
- **AI Agent (LangChain, agent.js):**
  - Handles user queries about YouTube videos
  - Extracts, analyzes, and summarizes transcript segments using dual Gemini models
  - Supports time-based, topic-based, sentiment, and metadata queries
  - Triggers transcript scraping if needed and provides fallback answers


---

## How It Works

1. **User submits a YouTube video link and question.**
2. **Backend** fetches the transcript using BrightData, stores/retrieves data from PostgreSQL, and uses Gemini AI to analyze and summarize responses.
3. **Frontend** displays the AI's answer and allows further chat.

---

## Getting Started

### 1. Clone the repository

```bash
https://github.com/lakshmannarendra/YtChat-AiAgent.git


```

### 2. Set up environment variables

Create a `.env` file in the `server/` folder. Example:

```
DB_URL=your_postgres_url
BRIGHTDATA_API_KEY=your_brightdata_key
GEMINI_API_KEY_SUMMARIZER=your_gemini_key_1
GEMINI_API_KEY_ANALYZER=your_gemini_key_2
GOOGLE_API_KEY=your_google_api_key
```

> **Never commit your `.env` files or API keys.**

### 3. Install dependencies

```bash
cd server
npm install
cd ../client
npm install
```

### 4. Start the backend

```bash
cd ../server
node index.js
```

### 5. Start the frontend

```bash
cd ../client
npm run dev
```



---

## Notes

- The backend uses **PostgreSQL** and BrightData for transcript scraping.
- The frontend is built with React and Vite.
- For production, deploy the backend to a cloud service (Genezio, Vercel, Render, Railway, AWS, etc.)

---

## Bright Data & Public Access

To use Bright Data for YouTube transcript scraping, your backend server must be accessible from the public internet. Bright Data cannot send data to a backend running only on your local machine.

**Options:**
- Deploy your backend to a cloud service.
- Or, for local development, use a tunneling tool like [ngrok](https://ngrok.com/) to expose your local server to the internet.

Example (using ngrok):

```bash
npm install -g ngrok
ngrok http 5002
```
Then update your Bright Data webhook/API settings to use the public ngrok URL.

---

## AI Agent (LangChain) Tasks

The AI agent in `server/agent.js` uses LangChain and Gemini AI to process user queries about YouTube videos. Here’s how it works:

- **Dual Gemini AI Setup:**
  - One Gemini model ("analyzer") extracts structured facts and key points from transcript segments.
  - Another Gemini model ("summarizer") rewrites the analysis into a user-friendly answer.
- **Transcript Retrieval:**
  - The agent fetches transcript chunks from a vector store (semantic search DB) for the requested YouTube video.
- **Query Understanding:**
  - The agent parses the user’s question for time ranges (e.g., "at 2:30", "last minute"), topics, sentiment, or metadata requests.
- **Segment Analysis:**
  - For time/topic-based queries, it selects relevant transcript segments and analyzes them with the analyzer model.
- **Summarization:**
  - The summarizer model turns the analysis into a concise, engaging answer for the user.
- **Fallbacks:**
  - If no transcript is found, it triggers a BrightData scrape and asks the user to retry.
  - If the query is general, it summarizes the whole transcript.

This approach enables accurate, context-aware answers about any YouTube video, including specific time ranges, topics, or overall sentiment.

---

## Project Structure

```
YtChat-AiAgent-main/
├── client/        # React frontend
├── server/        # Node.js backend
│   └── .env       # Environment variables
├── README.md      # Project documentation
└── ...
```

---

## Author

Alluri Lakshman Narendra – Indian Institute of Technology (IIT) Dharwad – Computer Science and Engineering (CSE)

---
