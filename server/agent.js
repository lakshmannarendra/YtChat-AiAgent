import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { vectorStore } from './embeddings.js';
import { triggerYoutubeVideoScrape } from './brightdata.js';

/*
  agent.js – minimal, LLM-first implementation (≈150 lines)
  ------------------------------------------------------------------
  A dedicated "extractor" Gemini model converts a raw user question
  into a structured JSON plan (url, video_id, time_range, …). Down-
  stream nodes act purely on that structure; no hand-written regexes
  or heuristics for URL parsing, timestamp detection, sentiment checks
  – everything is delegated to the LLMs.
*/

// 1️⃣  LLM setup ---------------------------------------------------------
const extractorLLM = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
  apiKey: process.env.GEMINI_API_KEY_EXTRACTOR || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY,
  temperature: 0.0,
});

const analyzerLLM = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
  apiKey: process.env.GEMINI_API_KEY_ANALYZER || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY,
  temperature: 0.2,
});

const summarizerLLM = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
  apiKey: process.env.GEMINI_API_KEY_SUMMARIZER || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY,
  temperature: 0.3,
});

// // 2️⃣  Helper – pick transcript chunks by absolute seconds --------------
// function chunksByTime(docs, start, end, totalSec = 3600) {
//   if (!docs.length) return [];
//   const timed = docs.filter((d) => typeof d.metadata?.start === 'number' && typeof d.metadata?.end === 'number');
//   if (timed.length) return timed.filter((d) => d.metadata.start < end && d.metadata.end > start);
//   const from = Math.floor((start / totalSec) * docs.length);
//   const to = Math.max(from + 1, Math.ceil((end / totalSec) * docs.length));
//   return docs.slice(from, to);
// }

// 2️⃣.b Regex fall-back helpers (ensures old reliability if LLM misses something) ----
function extractYoutubeUrl(text) {
  const regex = /(https?:\/\/(?:www\.|m\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)[^\s]+)/;
  const match = text.match(regex);
  return match ? match[1] : null;
}

function getVideoIdFromUrl(url) {
  const regex = /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/;
  const match = url?.match(regex);
  return match ? match[1] : null;
}

// 3️⃣  Sequential agent --------------------------------------------------

export const agent = async (input) => {
  // --- Step 1: Parse query ------------------------------------------------
  const extractionPrompt = `You are an extraction AI.\nReturn ONLY valid JSON with keys: {\n  "youtube_url":string|null,\n  "video_id":string|null,\n  "time_range":{start:number,end:number}|null,\n  "timestamp":number|null,\n  "topic":string|null,\n  "request_type":"sentiment"|"metadata"|"general"\n}.\nUser:"${input}"`;
  let plan = {};
  try {
    const res = await extractorLLM.invoke(extractionPrompt);
    plan = JSON.parse(res.content ?? res);
  } catch {
    // ignore – plan stays empty
  }

  let {
    youtube_url: initialUrl = null,
    video_id: initialVid = null,
    // time_range = null,
    // timestamp = null,
    // topic = null,
    // request_type = 'general',
  } = plan;

  // Allow reassignment for fallbacks
  let youtube_url = initialUrl;
  let video_id = initialVid;

  // --- Regex fallback ---------------------------------------------------
  if (!youtube_url) youtube_url = extractYoutubeUrl(input);
  if (!video_id && youtube_url) video_id = getVideoIdFromUrl(youtube_url);

  // --- Step 2: Fetch transcript (if we know the video) -------------------
  let docs = [];
  let videoDurationSec = 3600;
  if (video_id) {
    try {
      docs = await vectorStore.similaritySearch('video', 400, { video_id });
      videoDurationSec = parseInt(docs[0]?.metadata?.duration) || 3600;
    } catch {
      /* ignore */
    }
  }

  // --- Step 3: If no transcript, maybe trigger scrape -------------------
  if (!docs.length) {
    if (youtube_url) {
      try {
        await triggerYoutubeVideoScrape(youtube_url);
      } catch {/* ignore */}
      return 'Video is being scraped. Please retry in ~10 seconds.';
    }
    // No video: just echo through summarizer
    const resp = await summarizerLLM.invoke(input);
    return resp.content ?? resp;
  }

  // // --- Step 4: Special sentiment / metadata requests --------------------
  // if (request_type === 'sentiment') {
  //   const full = docs.map((d) => d.pageContent).join('\n');
  //   const res = await analyzerLLM.invoke(`Analyse sentiment & tone:\n"""\n${full}\n"""`);
  //   return res.content ?? res;
  // }
  // if (request_type === 'metadata') {
  //   const md = docs[0]?.metadata || {};
  //   if (!Object.keys(md).length) return 'No metadata available.';
  //   const res = await analyzerLLM.invoke(`User:"${input}"\nMetadata:${JSON.stringify(md, null, 2)}\nExplain.`);
  //   return res.content ?? res;
  // }

  // // --- Step 5: Narrow to relevant segment -------------------------------
  let segment = docs;
  // if (time_range) {
  //   segment = chunksByTime(docs, time_range.start, time_range.end, videoDurationSec);
  // } else if (timestamp) {
  //   segment = chunksByTime(docs, Math.max(0, timestamp - 30), timestamp + 30, videoDurationSec);
  // } else if (topic) {
  //   segment = await vectorStore.similaritySearch(topic, 4, { video_id });
  // }

  // --- Step 6: Analyse & summarise --------------------------------------
  const transcriptText = segment.map((d) => d.pageContent).join('\n');
  const analysisPrompt = `You are an expert analyst.\nUser:"${input}"\nTranscript:"""${transcriptText}"""\nReturn JSON with keys {answer:string,key_points:string[],time_range:string}`;
  let analysis;
  try {
    const raw = await analyzerLLM.invoke(analysisPrompt);
    const txt = raw.content ?? raw;
    try {
      analysis = JSON.parse(txt);
    } catch {
      // LLM replied with plain text; treat it as the answer directly
      analysis = { answer: txt, key_points: [], time_range: '' };
    }
  } catch (err) {
    analysis = { answer: err?.message || 'Unable to analyse', key_points: [], time_range: '' };
  }

  const summaryPrompt = `User:"${input}"\nAnalysis:${JSON.stringify(analysis)}\nWrite concise helpful answer.`;
  const finalRes = await summarizerLLM.invoke(summaryPrompt);
  return finalRes.content ?? finalRes;
};
