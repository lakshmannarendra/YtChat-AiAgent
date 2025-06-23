import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { agent } from './agent.js';
import { addYTVideoToVectorStore } from './embeddings.js';
import { triggerYoutubeVideoScrape } from './brightdata.js';

const port = process.env.PORT || 3000;

const app = express();

app.use(express.json({ limit: '200mb' }));
app.use(cors());

app.get('/', (req, res) => {
  res.send('Hello World!');
});

// curl -X POST http://localhost:3000/generate \
// -H "Content-Type: application/json" \
// -d '{
//   "query": "What will people learn from this video?",
//   "video_id": "Pxn276cWKeI",
//   "thread_id": 1
// }'

app.post('/generate', async (req, res) => {
  const { query, video_id, thread_id } = req.body;
  console.log('Query:', query, 'Video ID:', video_id, 'Thread ID:', thread_id);

  try {
    const response = await agent(query);
    res.send(response);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/webhook', async (req, res) => {
  try {
    const videos = Array.isArray(req.body) ? req.body : [];
    console.log(`📥 Webhook received – ${videos.length} video object(s)`);

    await Promise.all(
      videos.map(async (video, idx) => {
        try {
          console.log(`→ [${idx}] Adding video ${video.video_id || 'unknown'} to vector store…`);
          await addYTVideoToVectorStore(video);
        } catch (err) {
          console.error(`❌ Failed to add video ${video.video_id}:`, err);
        }
      })
    );

    res.json({ status: 'ok' });
  } catch (err) {
    console.error('Webhook processing error:', err);
    res.status(500).json({ error: 'Webhook processing failed' });
  }
});

// Expose an explicit endpoint to trigger BrightData scraping.
// Expected body: { "url": "https://youtu.be/..." }
app.post('/scrape-video', async (req, res) => {
  const { url } = req.body;

  if (!url) {
    return res.status(400).json({ error: 'YouTube URL is required' });
  }

  try {
    console.log('Triggering video scrape for:', url);
    const snapshotId = await triggerYoutubeVideoScrape(url);

    res.json({ message: 'Video scraping initiated', snapshotId, status: 'processing' });
  } catch (error) {
    console.error('Error triggering video scrape:', error);
    res.status(500).json({ error: 'Failed to initiate video scraping' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
