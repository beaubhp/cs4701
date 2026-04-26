export type SystemId =
  | "non_rag"
  | "bm25_rag"
  | "dense_rag"
  | "bm25_rerank_rag"
  | "dense_rerank_rag"
  | "iterkey_rag";

export type Question = {
  question_id: string;
  question: string;
  answerable: boolean;
  topic: string;
  difficulty: string;
  question_type: string;
  gold_answer_short: string;
  gold_chunk_ids: string[];
  gold_doc_ids: string[];
  negative_rationale: string | null;
};

export type ReviewRow = {
  question_id: string;
  system: SystemId;
  question: string;
  answerable: boolean;
  gold_answer_short: string;
  gold_chunk_ids: string[];
  model_answer: string;
  abstained: boolean;
  abstention_reason: string | null;
  citations: string[];
  retrieved_chunk_ids: string[];
  manual_correct: boolean;
  manual_citation_supported: boolean | null;
  manual_hallucinated: boolean;
  review_abstention_correct: boolean | null;
  review_hallucination_severity: "none" | "minor" | "major";
  manual_notes: string;
};

export type ReviewedMetrics = {
  system: SystemId;
  num_reviewed: number;
  reviewed_correct_rate: number;
  reviewed_hallucination_rate: number;
  major_hallucination_rate: number;
  answerable: {
    num_questions: number;
    reviewed_accuracy: number;
    reviewed_hallucination_rate: number;
    reviewed_over_abstention_rate: number;
  };
  unanswerable: {
    num_questions: number;
    reviewed_correct_refusal_rate: number;
    reviewed_false_answer_rate: number;
    reviewed_hallucination_rate: number;
  };
  citations: {
    num_applicable_rows: number;
    reviewed_citation_support_rate: number;
  };
};

export type DashboardData = {
  generated_at: string;
  questions: Question[];
  metrics: Record<SystemId, ReviewedMetrics>;
  reviews: Record<SystemId, ReviewRow[]>;
};
