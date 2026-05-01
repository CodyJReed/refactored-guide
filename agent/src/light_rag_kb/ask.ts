import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { getChatModel } from "../shared/models";
import { getVectorStore } from "./store";

export type KBSource = {
  source: string;
  chunkId: number;
};

export type KBAskResult = {
  answer: string;
  sources: KBSource[];
  confidence: number;
};

function buildConfidence(scores: number[]): number {
  if (!scores.length) return 0;

  const clamped = scores.map((score) => Math.max(0, Math.min(1, score)));
  const average = clamped.reduce((a, b) => a + b, 0);

  return Math.round((average * 100) / 100);
}

// Provide query and k === results to return
export async function askKb(query: string, k = 2): Promise<KBAskResult> {
  // Validate...
  const validatedQuery = (query ?? "").trim();
  // Handle empty string...
  if (!validatedQuery) {
    throw new Error("Query is empty, please try again.");
  }
  // Retrieve current store (with KB embeddings)...
  const store = getVectorStore();

  // Embed validated query
  const embedQuery = await store.embeddings.embedQuery(validatedQuery);
  // Run search against embeddings using embedded query...
  const pairs = await store.similaritySearchVectorWithScore(embedQuery, k);
  // Extract document chunks from pairs...
  const chunks = pairs.map(([doc]) => ({
    text: doc.pageContent || "",
    meta: doc.metadata || {},
  }));
  // Extract scores from pairs
  const scores = pairs.map(([__dirname, score]) => Number(score) || 0);

  //   Create context for model's answer
  const context = buildContext(chunks);

  //   Retrieve Model's answer using query && conext...
  const answer = await buildLLMAnswer(validatedQuery, context);

  //   Compose sources...
  const sources: KBSource[] = chunks.map((c) => ({
    source: String(c.meta?.source ?? "unknown"),
    chunkId: Number(c.meta?.chunkId) ?? 0,
  }));

  //   Compose a scoring average
  const confidence = buildConfidence(scores);

  return {
    answer,
    sources,
    confidence,
  };
}

function buildContext(
  chunks: {
    text: string;
    meta: any;
  }[],
) {
  // Provide the intended format...
  return chunks
    .map(({ text, meta }, i) =>
      [
        `[#${i + 1}] ${String(meta?.source ?? "unknown")} #${String(meta?.chunkId ?? "?")}`,
        text ?? "Empty text",
      ].join("\n"),
    )
    .join("\n\n---\n\n");
}

async function buildLLMAnswer(query: string, context: string) {
  const model = getChatModel({
    temperature: 0.2,
  });

  const res = await model.invoke([
    new SystemMessage(
      [
        "You concisely answer questions using provided context.",
        "If the answer is not found in the current context, say so briefly",
        "Rules:",
        "- Be accurate, netral, and avoid marketing info",
        "- 5-8 sentences max",
        "- Use only the provided context; do not invent new facts",
      ].join("\n"),
    ),
    new HumanMessage(
      [
        `Question: ${query}`,
        "Context: (quoted chunks) ->",
        context || "no relevant context",
      ].join("\n"),
    ),
  ]);

  const finalRes =
    typeof res.content === "string" ? res.content : String(res.content);

  return finalRes.trim().slice(0, 1500);
}
