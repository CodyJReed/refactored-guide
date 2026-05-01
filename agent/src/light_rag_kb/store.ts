// This file should handle the following...
// KB (knowledge base)...
// ... store embedding in RAM, letting us insert chunks to later...
// ... run a search against

// Create embeddings client
import { OllamaEmbeddings } from "@langchain/ollama";
import { TaskType } from "@google/generative-ai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";


type Provider = "ollama" | "google";

function getProvider(): Provider {
  const getCurrentProvider = (
    process.env.RAG_MODEL_PROVIDER ?? "gemini"
  ).toLowerCase();

  return getCurrentProvider === "gemini" ? "google" : "ollama";
}

function makeGoogleEmbeddings() {
  const key = process.env.GOOGLE_API_KEY;
  if (!key) {
    throw new Error("Google api key is missing");
  }

  return new GoogleGenerativeAIEmbeddings({
    apiKey: key,
    model: "gemini-embedding-001",
    taskType: TaskType.RETRIEVAL_DOCUMENT,
  });
}

// TODO Make use of MLX and local embedding...
// Ollama cloud does not support embedding; yet...
function makeOllamaEmbeddings() {
  const key = process.env.OLLAMA_API_KEY;
  if (!key) {
    throw new Error("Ollama api key is missing");
  }

  return new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "https://ollama.com",
    headers: {
      Authorization: `Bearer ${key}`,
    },
  });
}

function makeEmbeddings(provider: Provider) {
  return provider === "ollama"
    ? makeOllamaEmbeddings()
    : makeGoogleEmbeddings();
}

// Vector Store
let store: MemoryVectorStore | null = null;
let currentSetProvider: Provider | null = null;

// Handle swapping bewteen available Providers
export function getVectorStore(): MemoryVectorStore {
  const provider = getProvider();
  // Return store...
  //  if current provider remains unchanged
  if (store && currentSetProvider === provider) {
    return store;
  } 

  // Handle initialization or provider change
  store = new MemoryVectorStore(makeEmbeddings(provider));
  currentSetProvider = provider;

  return store;
}

export async function addChunks(docs: Document[]): Promise<number> {
  // Take incoming docs [...{metadata: {source, chunkId}}]
  if (!Array.isArray(docs) || docs.length === 0) return 0;
  // Retrieve store...
  const store = getVectorStore();
  // Add docs (chunks) to store
  await store.addDocuments(docs);
  // Return count
  return docs.length;
}

export function resetStore() {
  store = null;
  currentSetProvider = null;
}
