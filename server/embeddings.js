import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Document } from '@langchain/core/documents';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import pkg from 'pg';
const { Client } = pkg;

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: 'models/embedding-001',
  apiKey: process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY,
});

// Ensure the pgvector extension and the expected table exist in the database.
// async function ensureDatabaseSetup() {
//   const client = new Client({
//     connectionString: process.env.DB_URL,
//     ssl: {
//       // Neon requires SSL; the certificate chain is trusted so we can skip verification
//       rejectUnauthorized: false,
//     },
//   });

//   await client.connect();

//   // Enable the pgvector extension (safe to run every time)
//   await client.query('CREATE EXTENSION IF NOT EXISTS vector');

//   // Determine embedding vector dimension dynamically
//   const sampleVector = await embeddings.embedQuery('dimension_test');
//   const dimension = sampleVector.length;
//   globalThis.__EMBEDDING_DIM__ = dimension; // cache for reuse

//   // Create the transcripts table if it does not exist yet
//   const createTableSQL = `
//     CREATE TABLE IF NOT EXISTS transcripts (
//       id SERIAL PRIMARY KEY,
//       content TEXT,
//       metadata JSONB,
//       vector vector(${dimension})
//     );
//   `;
//   await client.query(createTableSQL);

//   await client.end();
// }

// // Make sure the database is ready before we try to use it in the vector store
// await ensureDatabaseSetup();

export const vectorStore = await PGVectorStore.initialize(embeddings, {
  postgresConnectionOptions: {
    connectionString: process.env.DB_URL,
    ssl: {
      rejectUnauthorized: false,
    },
  },
  tableName: 'transcripts',
  columns: {
    idColumnName: 'id',
    vectorColumnName: 'vector',
    contentColumnName: 'content',
    metadataColumnName: 'metadata',
  },
  dimensions: globalThis.__EMBEDDING_DIM__,
  distanceStrategy: 'cosine',
});

export const addYTVideoToVectorStore = async (videoData) => {
  const { transcript, video_id } = videoData;
  console.log(`Ingesting video ${video_id}. Transcript length: ${transcript?.length ?? 0} chars`);

  const docs = [
    new Document({
      pageContent: transcript,
      metadata: { video_id },
    }),
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const chunks = await splitter.splitDocuments(docs);
  console.log(`Split into ${chunks.length} chunks, embedding & storing…`);
  await vectorStore.addDocuments(chunks);
  console.log(`✅ Stored ${chunks.length} chunks for ${video_id}`);

  // Log total row count
  const client = new Client({ connectionString: process.env.DB_URL, ssl: { rejectUnauthorized: false } });
  await client.connect();
  const { rows } = await client.query('SELECT COUNT(*) FROM transcripts');
  console.log('🔥 Row-count seen by server:', rows[0].count);
  await client.end();
};
