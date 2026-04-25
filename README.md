# Measuring Hallucination in RAG vs. Non-RAG Systems

Fixed-corpus evaluation of hallucination in policy question answering. The task is to answer Cornell policy questions using a frozen set of official Cornell policy documents. RAG systems must answer from retrieved passages and cite supporting chunks. Systems should abstain when the corpus does not support an answer.

## Repository Layout

```text
data/sources.yml                         Source registry
data/raw/                                Frozen source snapshots
data/processed/documents.jsonl           Extracted normalized documents
data/processed/chunks.jsonl              Retrieval chunks
data/benchmark/questions.jsonl           100-question benchmark
data/results/retrieval_*.json            Retrieval metrics
data/results/generations_*.jsonl         Generated answers
data/results/generations_*_eval.json     Heuristic generation metrics
data/results/generations_*_review_*.json Manual review labels and metrics
src/corpus/                              Ingestion and chunking
src/retrieval/                           BM25 and dense retrieval
src/generation/                          OpenAI generation
src/eval/                                Benchmark, retrieval, and generation evaluation
```

## Corpus

Six official Cornell sources are frozen in `data/sources.yml`:

```text
student_code_of_conduct
student_code_of_conduct_procedures
code_of_academic_integrity
leaves_and_withdrawals
voluntary_leave_absence_students
title_iv_enrollment_status_reporting
```

Current processed corpus:

```text
documents: 6
chunks: 158
chunk target: 320 words
chunk max: 380 words
overlap: 50 words
```

## Benchmark

`data/benchmark/questions.jsonl`

```text
total questions: 100
answerable: 75
unanswerable: 25
```

Each answerable question includes:

```text
gold_doc_ids
gold_chunk_ids
gold_answer_short
evidence quote
qrels
```

Each unanswerable question includes:

```text
negative_rationale
near_miss_doc_ids
unanswerable_type
```

## Systems

```text
non_rag     OpenAI answer with no retrieved corpus context
bm25_rag    Local Okapi BM25 retrieval + grounded generation
dense_rag   SentenceTransformers dense retrieval + grounded generation
bm25_rerank_rag   BM25 candidates + cross-encoder reranking + grounded generation
dense_rerank_rag  Dense candidates + cross-encoder reranking + grounded generation
```

Generation model:

```text
gpt-5.4-mini
temperature: 0
```

Reranker model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
candidate_k: 20
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For generation:

```bash
echo 'OPENAI_API_KEY=your_key_here' > .env
```

`.env` is ignored by git.

## Reproduce Corpus

Use existing frozen snapshots:

```bash
python3 -m src.corpus.ingest data/sources.yml
python3 -m src.corpus.chunk data/processed/documents.jsonl
```

Force re-download:

```bash
python3 -m src.corpus.ingest data/sources.yml --refresh
python3 -m src.corpus.chunk data/processed/documents.jsonl
```

## Validate Benchmark

```bash
python3 -m src.eval.validate_benchmark data/benchmark/questions.jsonl
```

Expected summary:

```text
num_questions: 100
answerable: 75
unanswerable: 25
```

## Run Retrieval Evaluation

```bash
python3 -m src.eval.retrieval_eval data/benchmark/questions.jsonl --retriever bm25
python3 -m src.eval.retrieval_eval data/benchmark/questions.jsonl --retriever dense
python3 -m src.eval.retrieval_eval data/benchmark/questions.jsonl --retriever bm25_rerank --candidate-k 20
python3 -m src.eval.retrieval_eval data/benchmark/questions.jsonl --retriever dense_rerank --candidate-k 20
python3 -m src.eval.compare_retrieval
```

Outputs:

```text
data/results/retrieval_bm25.json
data/results/retrieval_bm25_predictions.jsonl
data/results/retrieval_dense.json
data/results/retrieval_dense_predictions.jsonl
data/results/retrieval_bm25_rerank.json
data/results/retrieval_bm25_rerank_predictions.jsonl
data/results/retrieval_dense_rerank.json
data/results/retrieval_dense_rerank_predictions.jsonl
data/results/retrieval_comparison.json
data/results/retrieval_comparison_bm25_rerank.json
data/results/retrieval_comparison_dense_rerank.json
data/results/retrieval_comparison_reranked.json
```

## Run Generation

Requires `OPENAI_API_KEY`.

```bash
python3 -m src.generation.run_generation data/benchmark/questions.jsonl --system non_rag
python3 -m src.generation.run_generation data/benchmark/questions.jsonl --system bm25_rag
python3 -m src.generation.run_generation data/benchmark/questions.jsonl --system dense_rag
python3 -m src.generation.run_generation data/benchmark/questions.jsonl --system bm25_rerank_rag --candidate-k 20
python3 -m src.generation.run_generation data/benchmark/questions.jsonl --system dense_rerank_rag --candidate-k 20
```

Outputs:

```text
data/results/generations_non_rag.jsonl
data/results/generations_bm25_rag.jsonl
data/results/generations_dense_rag.jsonl
data/results/generations_bm25_rerank_rag.jsonl
data/results/generations_dense_rerank_rag.jsonl
```

## Run Heuristic Generation Evaluation

```bash
python3 -m src.eval.generation_eval data/results/generations_non_rag.jsonl
python3 -m src.eval.generation_eval data/results/generations_bm25_rag.jsonl
python3 -m src.eval.generation_eval data/results/generations_dense_rag.jsonl
python3 -m src.eval.generation_eval data/results/generations_bm25_rerank_rag.jsonl
python3 -m src.eval.generation_eval data/results/generations_dense_rerank_rag.jsonl
```

This writes heuristic metrics and review templates.

Final reviewed labels are stored in:

```text
data/results/generations_non_rag_review_template.jsonl
data/results/generations_bm25_rag_review_template.jsonl
data/results/generations_dense_rag_review_template.jsonl
data/results/generations_bm25_rerank_rag_review_template.jsonl
data/results/generations_dense_rerank_rag_review_template.jsonl
```

Final reviewed metrics are stored in:

```text
data/results/reviewed_generation_metrics.json
```

## Retrieval Results

Answerable questions only.

```text
BM25          recall@1 73.3%   recall@3 93.3%   recall@5 98.7%   recall@10 98.7%    mrr 0.837
BM25+rerank   recall@1 72.0%   recall@3 90.7%   recall@5 94.7%   recall@10 100.0%   mrr 0.812
Dense         recall@1 56.0%   recall@3 65.3%   recall@5 74.7%   recall@10 84.0%    mrr 0.632
Dense+rerank  recall@1 70.7%   recall@3 88.0%   recall@5 92.0%   recall@10 94.7%    mrr 0.793
```

## Reviewed Generation Results

Reviewed over all 100 questions.

```text
non_rag           correct 34.0%   hallucination 54.0%   answerable_acc 32.0%   unanswerable_refusal 40.0%    citation_support n/a
bm25_rag          correct 95.0%   hallucination 2.0%    answerable_acc 94.7%   unanswerable_refusal 96.0%    citation_support 98.0%
dense_rag         correct 91.0%   hallucination 0.0%    answerable_acc 88.0%   unanswerable_refusal 100.0%   citation_support 92.0%
bm25_rerank_rag   correct 98.0%   hallucination 0.0%    answerable_acc 97.3%   unanswerable_refusal 100.0%   citation_support 99.0%
dense_rerank_rag  correct 97.0%   hallucination 1.0%    answerable_acc 96.0%   unanswerable_refusal 100.0%   citation_support 97.0%
```

## Review Labels

Manual review fields:

```text
manual_correct
manual_citation_supported
manual_hallucinated
review_abstention_correct
review_hallucination_severity
manual_notes
```

Rubric:

```text
manual_correct: answer matches benchmark-supported answer
manual_citation_supported: cited chunks support the answer claims
manual_hallucinated: answer asserts unsupported or wrong policy detail
review_abstention_correct: unanswerable row avoids unsupported specifics
```
