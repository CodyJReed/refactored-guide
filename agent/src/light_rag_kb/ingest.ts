import { chunkText } from "./chunk";
import { addChunks } from "./store";

export type IngestTextInput = {
  text: string;
  source: string;
};

export async function ingestText(input: IngestTextInput) {
  const raw = (input.text ?? "").trim();

  // Handle input missing text
  if (!raw) {
    throw new Error("No file to ingest");
  }
  // Handle missing source...
  // If source is missing assume 'text' was pasted within form input...
  const source = input?.source ?? "pasted-text";

  // Call to transform 'raw' text into chunks
  const docs = chunkText(raw, source);

  // Embed each chunk into vector store...
  //   Return LangChain Document
  const chunkCount = await addChunks(docs);

  return {
    docCount: 1,
    chunkCount,
    source,
  };
}
