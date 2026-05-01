import { Router } from "express";
import * as z from "zod";
import { ingestText } from "../light_rag_kb/ingest";
import { resetStore } from "../light_rag_kb/store";
import { askKb } from "../light_rag_kb/ask";

export const kbRouter = Router();

const ingestBody = z.object({
  text: z.string().min(1, "Provide some text to ingest."),
  source: z.string().optional(),
});

type ingestBodyT = z.infer<typeof ingestBody>;
// Route for ingesting LangChain Documents (text && source)
kbRouter.post("/ingest", async (req, res) => {
  try {
    const body = ingestBody.parse(req.body) as ingestBodyT;
    const { text, source } = body;
    const result = await ingestText({ text, source: source ?? "pasted text" });

    return res.status(200).json({
      ok: true,
      ...result,
    });
  } catch (e) {
    console.log(e);

    res.status(400).json({
      error: "An error occurred during ingestion.",
    });
  }
});
// Route to reset current store && provider
kbRouter.post("/reset", (_req, res) => {
  try {
    resetStore();

    return res.status(200).json({
      ok: true,
    });
  } catch (e) {
    console.log(e);
    res.status(400).json({
      error: "An error occurred during store reset.",
    });
  }
});

const AskBody = z.object({
  query: z.string().min(3, "Please ask a complete query"),
  k: z.number().int().min(1).max(10).optional(),
});

type AskBodyT = z.infer<typeof AskBody>;
// Route for retrieving a LLM response from a given query...
// optionally mark number of results to consider for context
kbRouter.post("/ask", async (req, res) => {
  try {
    const body = AskBody.parse(req.body) as AskBodyT;
    const { query, k } = body;

    const result = await askKb(query, k ?? 2);

    return res.status(200).json({
      answer: result.answer,
      sources: result.sources,
      confidence: result.confidence,
    });
  } catch (e) {
    console.log(e);
    res.status(400).json({
      error: "An error occurred during store reset.",
    });
  }
});
