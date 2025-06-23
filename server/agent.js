import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { vectorStore } from './embeddings.js';
import { triggerYoutubeVideoScrape } from './brightdata.js';

// --- Dual Gemini AI setup ---
const summarizerLLM = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
  apiKey: process.env.GEMINI_API_KEY_SUMMARIZER || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY,
  maxOutputTokens: 2048,
  temperature: 0.3,
});
const analyzerLLM = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
  apiKey: process.env.GEMINI_API_KEY_ANALYZER || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY,
  maxOutputTokens: 2048,
  temperature: 0.2,
});

const llm = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
  apiKey: process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY,
  maxOutputTokens: 2048,
  temperature: 0.3,
});

function extractYoutubeUrl(text) {
  const regex = /(https?:\/\/(?:www\.|m\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)[^\s]+)/;
  const match = text.match(regex);
  return match ? match[1] : null;
}

function getVideoIdFromUrl(url) {
  const regex = /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/;
  const match = url.match(regex);
  return match ? match[1] : null;
}

// NEW: Extract minute or timestamp from user query
function extractMinuteOrTimestamp(text) {
  // Match patterns like "5th minute", "at 2:30", "minute 10", "last minute"
  const minuteMatch = text.match(/(\d+)(?:st|nd|rd|th)? minute/);
  if (minuteMatch) {
    return { minute: parseInt(minuteMatch[1], 10) };
  }
  const timeMatch = text.match(/at (\d+):(\d{2})/);
  if (timeMatch) {
    return { seconds: parseInt(timeMatch[1], 10) * 60 + parseInt(timeMatch[2], 10) };
  }
  if (/last minute/.test(text)) {
    return { lastMinute: true };
  }
  return null;
}

// NEW: Find transcript segment(s) for a given minute or timestamp
function getTranscriptSegmentByMinute(docs, minute) {
  // Assume each doc has a chunk of transcript, and metadata may have timing info
  // If not, just split by chunk index
  if (!docs.length) return [];
  const approxChunkPerMinute = Math.ceil(docs.length / 10); // crude guess for 10-min video
  const startIdx = (minute - 1) * approxChunkPerMinute;
  return docs.slice(startIdx, startIdx + approxChunkPerMinute);
}

function getTranscriptSegmentBySeconds(docs, seconds) {
  // If metadata has timing, use it. Otherwise, fallback to chunk index
  if (!docs.length) return [];
  // Try to find a chunk with metadata.start <= seconds < metadata.end
  const found = docs.find(
    (d) => d.metadata && d.metadata.start <= seconds && d.metadata.end > seconds
  );
  if (found) return [found];
  // Fallback: crude split
  const idx = Math.floor((seconds / (10 * 60)) * docs.length); // assume 10-min video
  return [docs[idx]];
}

function getTranscriptSegmentLastMinute(docs) {
  if (!docs.length) return [];
  const approxChunkPerMinute = Math.ceil(docs.length / 10);
  return docs.slice(-approxChunkPerMinute);
}

async function summarizeTranscript(docs) {
  const transcriptText = docs.map((d) => d.pageContent).join('\n');
  const prompt = `You are a helpful assistant.\nHere is a YouTube video transcript:\n"""\n${transcriptText}\n"""\n\nProvide a concise, engaging summary of the video in 3–4 sentences.`;
  const response = await llm.invoke(prompt);
  return response.content ?? response;
}

// NEW: Summarize a specific segment
async function summarizeTranscriptSegment(docs, userQuery) {
  const transcriptText = docs.map((d) => d.pageContent).join('\n');
  const prompt = `You are a helpful assistant.\nThe user asked: "${userQuery}"\nHere is the relevant part of a YouTube video transcript:\n"""\n${transcriptText}\n"""\n\nAnswer the user's question based only on this segment. Be concise.`;
  const response = await llm.invoke(prompt);
  return response.content ?? response;
}

// Analyzer: analyze transcript/time/topic and return structured info
async function analyzeTranscriptSegment(docs, userQuery) {
  const transcriptText = docs.map((d) => d.pageContent).join('\n');
  const prompt = `You are an expert video analyst AI.\nThe user asked: "${userQuery}"\nHere is the relevant part of a YouTube video transcript:\n"""\n${transcriptText}\n"""\n\nAnalyze the transcript and extract the most relevant facts, actions, or technical details that answer the user's question. Respond in structured JSON with keys: {\n  "answer": string,\n  "key_points": string[],\n  "time_range": string (if applicable)\n}`;
  const response = await analyzerLLM.invoke(prompt);
  try {
    return JSON.parse(response.content ?? response);
  } catch {
    // fallback: return as plain text
    return { answer: response.content ?? response, key_points: [], time_range: '' };
  }
}

// Summarizer: take analyzer output and user query, return user-facing answer
async function summarizeAnalyzerResult(analyzerResult, userQuery) {
  const prompt = `You are a helpful assistant.\nThe user asked: "${userQuery}"\nHere is the analysis from another AI:\n${JSON.stringify(analyzerResult, null, 2)}\n\nWrite a clear, concise, and engaging answer for the user, using the analysis above. If key_points are present, include them as a bullet list.`;
  const response = await summarizerLLM.invoke(prompt);
  return response.content ?? response;
}

// Utility: Try to extract a topic or keyword from the user query
function extractTopic(text) {
  // Simple keyword extraction: look for 'about X', 'topic X', or quoted words
  const aboutMatch = text.match(/about ([\w\s]+)/i);
  if (aboutMatch) return aboutMatch[1].trim();
  const topicMatch = text.match(/topic ([\w\s]+)/i);
  if (topicMatch) return topicMatch[1].trim();
  const quoteMatch = text.match(/"([^"]+)"/);
  if (quoteMatch) return quoteMatch[1].trim();
  return null;
}

// Utility: Try to detect sentiment analysis request
function isSentimentRequest(text) {
  return /sentiment|emotion|feeling|tone/i.test(text);
}

// Utility: Try to detect if user wants video/channel metadata
function isMetadataRequest(text) {
  return /channel|views|likes|date|published|duration|length/i.test(text);
}

// Utility: Parse flexible time string (supports '2 hour 30 min', '2:30:00', '90th minute', 'first 10 minutes', 'last 5 minutes', etc.)
function parseTimeString(str, videoDurationSec = 10800) {
  // videoDurationSec: fallback for 'last X minutes' etc. (default 3hr)
  str = str.toLowerCase();
  let h = 0, m = 0, s = 0;
  // '90th minute', '10th minute', etc.
  const nthMinute = str.match(/(\d+)(?:st|nd|rd|th)? minute/);
  if (nthMinute) {
    m = parseInt(nthMinute[1], 10);
    return m * 60;
  }
  // 'first 10 minutes', 'first hour', etc.
  const firstMatch = str.match(/first (\d+) (minute|hour|second)s?/);
  if (firstMatch) {
    const val = parseInt(firstMatch[1], 10);
    if (firstMatch[2].startsWith('hour')) return 0;
    if (firstMatch[2].startsWith('minute')) return 0;
    if (firstMatch[2].startsWith('second')) return 0;
  }
  // 'last 5 minutes', 'last hour', etc.
  const lastMatch = str.match(/last (\d+) (minute|hour|second)s?/);
  if (lastMatch) {
    const val = parseInt(lastMatch[1], 10);
    if (lastMatch[2].startsWith('hour')) return videoDurationSec - val * 3600;
    if (lastMatch[2].startsWith('minute')) return videoDurationSec - val * 60;
    if (lastMatch[2].startsWith('second')) return videoDurationSec - val;
  }
  // 'after 1hr', 'after 90 minutes', etc.
  const afterMatch = str.match(/after (\d+) (hour|minute|second)s?/);
  if (afterMatch) {
    const val = parseInt(afterMatch[1], 10);
    if (afterMatch[2].startsWith('hour')) return val * 3600;
    if (afterMatch[2].startsWith('minute')) return val * 60;
    if (afterMatch[2].startsWith('second')) return val;
  }
  // 'before 2hr', 'before 120 minutes', etc.
  const beforeMatch = str.match(/before (\d+) (hour|minute|second)s?/);
  if (beforeMatch) {
    const val = parseInt(beforeMatch[1], 10);
    if (beforeMatch[2].startsWith('hour')) return val * 3600;
    if (beforeMatch[2].startsWith('minute')) return val * 60;
    if (beforeMatch[2].startsWith('second')) return val;
  }
  // '2 hour 30 min', '2hr 30min', etc.
  const hMatch = str.match(/(\d+)\s*(?:h|hour)/i);
  if (hMatch) h = parseInt(hMatch[1], 10);
  const mMatch = str.match(/(\d+)\s*(?:m|min)/i);
  if (mMatch) m = parseInt(mMatch[1], 10);
  const sMatch = str.match(/(\d+)\s*(?:s|sec)/i);
  if (sMatch) s = parseInt(sMatch[1], 10);
  // '2:30:00' or '2:30'
  const colonMatch = str.match(/(\d+):(\d{2})(?::(\d{2}))?/);
  if (colonMatch) {
    h = parseInt(colonMatch[1], 10);
    m = parseInt(colonMatch[2], 10);
    if (colonMatch[3]) s = parseInt(colonMatch[3], 10);
  }
  // 'half an hour', 'quarter hour', 'midway', etc.
  if (/half an hour/.test(str)) return 30 * 60;
  if (/quarter/.test(str)) return 15 * 60;
  if (/midway|middle/.test(str)) return Math.floor(videoDurationSec / 2);
  return h * 3600 + m * 60 + s;
}

// Improved: Extract time range from user query, e.g. 'from 2hr 30min to 3hr', 'first 10 minutes', 'last 5 minutes', etc.
function extractTimeRange(text, videoDurationSec = 10800) {
  // 'from X to Y', 'between X and Y', 'minutes 10 to 20', etc.
  const rangeMatch = text.match(/from ([^\s]+(?: [^\s]+)*) to ([^\s]+(?: [^\s]+)*)/i) ||
    text.match(/between ([^\s]+(?: [^\s]+)*) and ([^\s]+(?: [^\s]+)*)/i) ||
    text.match(/minutes? (\d+) to (\d+)/i);
  if (rangeMatch) {
    let start, end;
    if (rangeMatch[3] && rangeMatch[4]) {
      start = parseInt(rangeMatch[3], 10) * 60;
      end = parseInt(rangeMatch[4], 10) * 60;
    } else {
      start = parseTimeString(rangeMatch[1], videoDurationSec);
      end = parseTimeString(rangeMatch[2], videoDurationSec);
    }
    if (!isNaN(start) && !isNaN(end)) {
      return { start, end };
    }
  }
  // 'first X minutes/hours/seconds'
  const firstMatch = text.match(/first (\d+) (minute|hour|second)s?/);
  if (firstMatch) {
    let end = 0;
    if (firstMatch[2].startsWith('hour')) end = parseInt(firstMatch[1], 10) * 3600;
    if (firstMatch[2].startsWith('minute')) end = parseInt(firstMatch[1], 10) * 60;
    if (firstMatch[2].startsWith('second')) end = parseInt(firstMatch[1], 10);
    return { start: 0, end };
  }
  // 'last X minutes/hours/seconds'
  const lastMatch = text.match(/last (\d+) (minute|hour|second)s?/);
  if (lastMatch) {
    let start = videoDurationSec;
    if (lastMatch[2].startsWith('hour')) start -= parseInt(lastMatch[1], 10) * 3600;
    if (lastMatch[2].startsWith('minute')) start -= parseInt(lastMatch[1], 10) * 60;
    if (lastMatch[2].startsWith('second')) start -= parseInt(lastMatch[1], 10);
    return { start, end: videoDurationSec };
  }
  // 'after X', 'before Y'
  const afterMatch = text.match(/after ([^\s]+(?: [^\s]+)*)/);
  if (afterMatch) {
    const start = parseTimeString(afterMatch[1], videoDurationSec);
    return { start, end: videoDurationSec };
  }
  const beforeMatch = text.match(/before ([^\s]+(?: [^\s]+)*)/);
  if (beforeMatch) {
    const end = parseTimeString(beforeMatch[1], videoDurationSec);
    return { start: 0, end };
  }
  // 'around X', 'about X', 'midway', etc. (return a small window)
  const aroundMatch = text.match(/around ([^\s]+(?: [^\s]+)*)/i) || text.match(/about ([^\s]+(?: [^\s]+)*)/i) || text.match(/midway|middle/);
  if (aroundMatch) {
    const center = parseTimeString(aroundMatch[1] || 'midway', videoDurationSec);
    return { start: Math.max(0, center - 60), end: center + 60 };
  }
  return null;
}

// Enhanced agent logic
export const agent = async (input) => {
  const youtubeUrl = extractYoutubeUrl(input);

  if (youtubeUrl) {
    const videoId = getVideoIdFromUrl(youtubeUrl);
    let docs = [];
    let metadata = {};
    let videoDurationSec = 10800; // fallback 3hr
    if (videoId) {
      try {
        docs = await vectorStore.similaritySearch('', 40, { video_id: videoId });
        if (docs[0] && docs[0].metadata) {
          metadata = docs[0].metadata;
          if (metadata.duration) {
            // Try to parse duration in seconds if available
            if (typeof metadata.duration === 'number') videoDurationSec = metadata.duration;
            else if (typeof metadata.duration === 'string') videoDurationSec = parseTimeString(metadata.duration);
          }
        }
      } catch (err) {
        console.error('Vector store retrieval error:', err);
      }
    }

    if (docs.length > 0) {
      // 0. Time range query (improved)
      const timeRange = extractTimeRange(input, videoDurationSec);
      if (timeRange) {
        const segment = getTranscriptSegmentsByTimeRange(docs, timeRange.start, timeRange.end);
        if (segment.length) {
          const analysis = await analyzeTranscriptSegment(segment, input);
          return await summarizeAnalyzerResult(analysis, input);
        }
      }
      // 1. Time-based query (improved for 'at', 'after', 'before', 'around', etc.)
      const timeRef = extractMinuteOrTimestamp(input);
      if (timeRef) {
        let segment = [];
        if (timeRef.minute) {
          segment = getTranscriptSegmentByMinute(docs, timeRef.minute);
        } else if (timeRef.seconds) {
          segment = getTranscriptSegmentBySeconds(docs, timeRef.seconds);
        } else if (timeRef.lastMinute) {
          segment = getTranscriptSegmentLastMinute(docs);
        }
        if (segment.length) {
          const analysis = await analyzeTranscriptSegment(segment, input);
          return await summarizeAnalyzerResult(analysis, input);
        }
      }
      // 2. Topic-based query
      const topic = extractTopic(input);
      if (topic) {
        const topicDocs = await vectorStore.similaritySearch(topic, 4, { video_id: videoId });
        if (topicDocs.length) {
          const analysis = await analyzeTranscriptSegment(topicDocs, input);
          return await summarizeAnalyzerResult(analysis, input);
        }
      }
      // 3. Sentiment analysis
      if (isSentimentRequest(input)) {
        const transcriptText = docs.map((d) => d.pageContent).join('\n');
        const analysis = await analyzerLLM.invoke(`Analyze the overall sentiment, emotion, and tone of this YouTube video transcript.\n"""\n${transcriptText}\n"""`);
        return await summarizeAnalyzerResult({ answer: analysis.content ?? analysis }, input);
      }
      // 4. Metadata request
      if (isMetadataRequest(input)) {
        if (metadata && Object.keys(metadata).length) {
          const analysis = await analyzerLLM.invoke(`Here is the video metadata: ${JSON.stringify(metadata, null, 2)}\n\nThe user asked: "${input}"\n\nExtract and explain the relevant metadata for the user's question.`);
          return await summarizeAnalyzerResult({ answer: analysis.content ?? analysis }, input);
        } else {
          return 'Sorry, video metadata is not available.';
        }
      }
      // 5. Otherwise, summarize whole video
      const analysis = await analyzeTranscriptSegment(docs, input);
      return await summarizeAnalyzerResult(analysis, input);
    }

    // No transcript yet -> trigger scrape
    try {
      await triggerYoutubeVideoScrape(youtubeUrl);
      return 'The video is being scraped. Please wait about 10 seconds and ask again.';
    } catch (err) {
      console.error('Failed to trigger scrape:', err);
      return 'Failed to trigger scraping for this video.';
    }
  }

  // Fallback: just echo through summarizer LLM
  const response = await summarizerLLM.invoke(input);
  return response.content ?? response;
};
