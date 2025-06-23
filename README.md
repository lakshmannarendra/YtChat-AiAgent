# YtChat – YouTube Video AI Chatbot

This project lets you chat with an AI about any YouTube video.  
It uses two Gemini AIs: one for analyzing video transcripts, and one for summarizing answers for users.

## Features

- Ask questions about any YouTube video
- Get answers about specific time ranges, topics, or general summaries
- Dual Gemini AI setup for accurate and helpful responses

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/YtChat.git
cd YtChat
```

### 2. Set up environment variables

Create a `.env` file in the `server/` folder. Example:

```
DB_URL=your_postgres_url
BRIGHTDATA_API_KEY=your_brightdata_key
GEMINI_API_KEY_SUMMARIZER=your_gemini_key_1
GEMINI_API_KEY_ANALYZER=your_gemini_key_2
```

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

### 6. Open the app

Go to [http://localhost:5173](http://localhost:5173) in your browser.

---

## Notes

- Never commit your `.env` files or API keys.
- The backend uses PostgreSQL and BrightData for transcript scraping.
- The frontend is built with React and Vite.

---
---
## Important: Deploy Backend for Bright Data

To use Bright Data for YouTube transcript scraping, your backend server must be deployed and accessible from the public internet.  
Bright Data cannot send data to a backend running only on your local machine.

**Options:**
- Deploy your backend to a cloud service (e.g., Genezio, Vercel, Render, Railway, AWS, etc.)
- Or, for local development, use a tunneling tool like [ngrok](https://ngrok.com/) to expose your local server to the internet.

Example (using ngrok):

```bash
npm install -g ngrok
ngrok http 3000
```
Then update your Bright Data webhook/API settings to use the public ngrok URL.

---

