import dashboardData from "../data/dashboard-data.json";
import type { DashboardData, SystemId } from "./types";

export const SYSTEM_ORDER: SystemId[] = [
  "non_rag",
  "bm25_rag",
  "dense_rag",
  "bm25_rerank_rag",
  "dense_rerank_rag",
  "iterkey_rag",
];

export const SYSTEM_LABELS: Record<SystemId, string> = {
  non_rag: "Non-RAG",
  bm25_rag: "BM25 RAG",
  dense_rag: "Dense RAG",
  bm25_rerank_rag: "BM25 + rerank",
  dense_rerank_rag: "Dense + rerank",
  iterkey_rag: "IterKey RAG",
};

export function getDashboardData(): DashboardData {
  return dashboardData as DashboardData;
}
