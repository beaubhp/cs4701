import { readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SYSTEMS = [
  "non_rag",
  "bm25_rag",
  "dense_rag",
  "bm25_rerank_rag",
  "dense_rerank_rag",
  "iterkey_rag",
];

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = join(__dirname, "..");
const repoRoot = join(webRoot, "..");

function jsonl(text) {
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line));
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function readJsonl(path) {
  return jsonl(await readFile(path, "utf8"));
}

function compactQuestion(row) {
  return {
    question_id: row.question_id,
    question: row.question,
    answerable: row.answerable,
    topic: row.topic,
    difficulty: row.difficulty,
    question_type: row.question_type,
    gold_answer_short: row.gold_answer_short,
    gold_chunk_ids: row.gold_chunk_ids,
    gold_doc_ids: row.gold_doc_ids,
    negative_rationale: row.negative_rationale,
  };
}

function compactReview(row) {
  return {
    question_id: row.question_id,
    system: row.system,
    question: row.question,
    answerable: row.answerable,
    gold_answer_short: row.gold_answer_short,
    gold_chunk_ids: row.gold_chunk_ids,
    model_answer: row.model_answer,
    abstained: row.abstained,
    abstention_reason: row.abstention_reason,
    citations: row.citations,
    retrieved_chunk_ids: row.retrieved_chunk_ids,
    manual_correct: row.manual_correct,
    manual_citation_supported: row.manual_citation_supported,
    manual_hallucinated: row.manual_hallucinated,
    review_abstention_correct: row.review_abstention_correct,
    review_hallucination_severity: row.review_hallucination_severity,
    manual_notes: row.manual_notes,
  };
}

async function main() {
  const questions = (await readJsonl(join(repoRoot, "data/benchmark/questions.jsonl"))).map(compactQuestion);
  const metrics = await readJson(join(repoRoot, "data/results/reviewed_generation_metrics.json"));
  const reviews = {};

  for (const system of SYSTEMS) {
    const path = join(repoRoot, `data/results/generations_${system}_review_template.jsonl`);
    reviews[system] = (await readJsonl(path)).map(compactReview);
  }

  const payload = {
    generated_at: "reviewed-results-snapshot",
    questions,
    metrics,
    reviews,
  };

  await writeFile(
    join(webRoot, "src/data/dashboard-data.json"),
    `${JSON.stringify(payload, null, 2)}\n`,
    "utf8",
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
