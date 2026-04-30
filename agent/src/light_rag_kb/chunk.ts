import { Document } from "@langchain/core/documents";

export const CHUNK_SIZE = 1000;
export const CHUNK_OVERLAP = 150;

// text -> markdown | article | policy
// src -> context used to retrieve 'answer'
export function chunkText(text: string, src: string): Document[] {
  const clean = (text ?? "").replace(/\r\n/g, "\n");

  const docs: Document[] = [];

  if (!clean.trim()) return docs;

  const step = Math.max(1, CHUNK_SIZE - CHUNK_OVERLAP);

  let start = 0;
  let chunkId = 0;

  while (start < clean.length) {
    const end = Math.min(clean.length, start + CHUNK_SIZE);

    // Create a chunk...
    const slice = clean.slice(start, end).trim();
    // If chunk contains text...
    if (slice.length > 0) {
      // push to docs array
      docs.push(
        new Document({
          pageContent: slice,
          metadata: {
            source: src,
            chunkId,
          },
        }),
      );
      // Update chunkId for next slice/chunk
      chunkId += 1;
    }
    start += step;
  }

  return docs;
}
