"use client";

import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { getDashboardData, SYSTEM_LABELS, SYSTEM_ORDER } from "../lib/data";
import type { Question, ReviewRow, SystemId } from "../lib/types";

const data = getDashboardData();

type AnswerabilityFilter = "all" | "answerable" | "unanswerable";
type IssueFilter = "all" | "incorrect" | "hallucinated" | "citation" | "abstained";

const percent = (value: number | null | undefined) =>
  value == null ? "n/a" : `${(value * 100).toFixed(1)}%`;

const titleCase = (value: string) =>
  value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

function reviewFor(system: SystemId, questionId: string) {
  return data.reviews[system].find((row) => row.question_id === questionId);
}

function questionHasIssue(questionId: string, issue: IssueFilter) {
  if (issue === "all") return true;
  const rows = SYSTEM_ORDER.map((system) => reviewFor(system, questionId)).filter(Boolean) as ReviewRow[];
  if (issue === "incorrect") return rows.some((row) => !row.manual_correct);
  if (issue === "hallucinated") return rows.some((row) => row.manual_hallucinated);
  if (issue === "citation") return rows.some((row) => row.manual_citation_supported === false);
  return rows.some((row) => row.abstained);
}

function issueCounts(questionId: string) {
  const rows = SYSTEM_ORDER.map((system) => reviewFor(system, questionId)).filter(Boolean) as ReviewRow[];
  return {
    correct: rows.filter((row) => row.manual_correct).length,
    hallucinated: rows.filter((row) => row.manual_hallucinated).length,
    unsupportedCitation: rows.filter((row) => row.manual_citation_supported === false).length,
  };
}

export default function Home() {
  const topics = useMemo(() => unique(data.questions.map((question) => question.topic)), []);
  const difficulties = useMemo(() => unique(data.questions.map((question) => question.difficulty)), []);
  const [query, setQuery] = useState("");
  const [topic, setTopic] = useState("all");
  const [difficulty, setDifficulty] = useState("all");
  const [answerability, setAnswerability] = useState<AnswerabilityFilter>("all");
  const [issue, setIssue] = useState<IssueFilter>("all");
  const [selectedQuestionId, setSelectedQuestionId] = useState("unanswerable_023");

  const filteredQuestions = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return data.questions.filter((question) => {
      const matchesSearch =
        !normalizedQuery ||
        question.question.toLowerCase().includes(normalizedQuery) ||
        question.question_id.toLowerCase().includes(normalizedQuery);
      const matchesTopic = topic === "all" || question.topic === topic;
      const matchesDifficulty = difficulty === "all" || question.difficulty === difficulty;
      const matchesAnswerability =
        answerability === "all" ||
        (answerability === "answerable" ? question.answerable : !question.answerable);
      return (
        matchesSearch &&
        matchesTopic &&
        matchesDifficulty &&
        matchesAnswerability &&
        questionHasIssue(question.question_id, issue)
      );
    });
  }, [answerability, difficulty, issue, query, topic]);

  const selectedQuestion =
    data.questions.find((question) => question.question_id === selectedQuestionId) ?? filteredQuestions[0] ?? data.questions[0];

  const selectedRows = SYSTEM_ORDER.map((system) => ({
    system,
    row: reviewFor(system, selectedQuestion.question_id),
  })).filter((entry): entry is { system: SystemId; row: ReviewRow } => Boolean(entry.row));

  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <p className="eyebrow">CS 4701 fixed-corpus evaluation</p>
          <h1>RAG Evaluation Dashboard</h1>
          <p className="lede">
            Cornell policy QA results across six systems and 100 audited questions.
          </p>
        </div>
        <div className="run-card" aria-label="Benchmark summary">
          <span>Frozen corpus</span>
          <strong>6 docs / 158 chunks</strong>
          <span>Benchmark</span>
          <strong>75 answerable / 25 unanswerable</strong>
        </div>
      </header>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Reviewed generation metrics</p>
            <h2>System summary</h2>
          </div>
          <span className="timestamp">Data bundle: {formatDataStamp(data.generated_at)}</span>
        </div>
        <div className="table-wrap">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>System</th>
                <th>Correct</th>
                <th>Hallucination</th>
                <th>Answerable acc.</th>
                <th>Unanswerable refusal</th>
                <th>Citation support</th>
              </tr>
            </thead>
            <tbody>
              {SYSTEM_ORDER.map((system) => {
                const metric = data.metrics[system];
                return (
                  <tr key={system}>
                    <th>{SYSTEM_LABELS[system]}</th>
                    <td>{percent(metric.reviewed_correct_rate)}</td>
                    <td>{percent(metric.reviewed_hallucination_rate)}</td>
                    <td>{percent(metric.answerable.reviewed_accuracy)}</td>
                    <td>{percent(metric.unanswerable.reviewed_correct_refusal_rate)}</td>
                    <td>{system === "non_rag" ? "n/a" : percent(metric.citations.reviewed_citation_support_rate)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="workspace">
        <aside className="panel question-browser">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Question explorer</p>
              <h2>{filteredQuestions.length} rows</h2>
            </div>
          </div>
          <div className="filters">
            <label>
              Search
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="question text or id" />
            </label>
            <div className="filter-grid">
              <Select label="Topic" value={topic} onChange={setTopic} values={["all", ...topics]} />
              <Select label="Difficulty" value={difficulty} onChange={setDifficulty} values={["all", ...difficulties]} />
              <Select
                label="Answerability"
                value={answerability}
                onChange={(value) => setAnswerability(value as AnswerabilityFilter)}
                values={["all", "answerable", "unanswerable"]}
              />
              <Select
                label="Issue"
                value={issue}
                onChange={(value) => setIssue(value as IssueFilter)}
                values={["all", "incorrect", "hallucinated", "citation", "abstained"]}
              />
            </div>
          </div>
          <div className="question-list">
            {filteredQuestions.map((question) => {
              const counts = issueCounts(question.question_id);
              return (
                <button
                  className={question.question_id === selectedQuestion.question_id ? "question-row active" : "question-row"}
                  key={question.question_id}
                  onClick={() => setSelectedQuestionId(question.question_id)}
                >
                  <span className="row-meta">
                    {question.question_id} · {titleCase(question.topic)}
                  </span>
                  <span>{question.question}</span>
                  <span className="mini-stats">
                    <Badge tone={question.answerable ? "neutral" : "amber"}>
                      {question.answerable ? "answerable" : "unanswerable"}
                    </Badge>
                    <Badge tone="green">{counts.correct}/6 correct</Badge>
                    {counts.hallucinated > 0 && <Badge tone="red">{counts.hallucinated} hallucinated</Badge>}
                    {counts.unsupportedCitation > 0 && <Badge tone="amber">{counts.unsupportedCitation} citation issue</Badge>}
                  </span>
                </button>
              );
            })}
          </div>
        </aside>

        <section className="panel comparison">
          <QuestionHeader question={selectedQuestion} />
          <div className="systems-grid">
            {selectedRows.map(({ system, row }) => (
              <SystemCard key={system} row={row} system={system} />
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}

function Select({
  label,
  value,
  values,
  onChange,
}: {
  label: string;
  value: string;
  values: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {values.map((item) => (
          <option value={item} key={item}>
            {titleCase(item)}
          </option>
        ))}
      </select>
    </label>
  );
}

function QuestionHeader({ question }: { question: Question }) {
  return (
    <div className="question-detail">
      <div className="question-title-line">
        <div>
          <p className="eyebrow">{question.question_id}</p>
          <h2>{question.question}</h2>
        </div>
        <Badge tone={question.answerable ? "green" : "amber"}>{question.answerable ? "answerable" : "unanswerable"}</Badge>
      </div>
      <div className="question-meta">
        <span>{titleCase(question.topic)}</span>
        <span>{titleCase(question.difficulty)}</span>
        <span>{titleCase(question.question_type)}</span>
      </div>
      <div className="gold-box">
        <strong>{question.answerable ? "Gold answer" : "Why unanswerable"}</strong>
        <p>{question.answerable ? question.gold_answer_short : question.negative_rationale}</p>
        {question.gold_chunk_ids.length > 0 && <code>{question.gold_chunk_ids.join(", ")}</code>}
      </div>
    </div>
  );
}

function SystemCard({ system, row }: { system: SystemId; row: ReviewRow }) {
  return (
    <article className="system-card">
      <div className="card-head">
        <h3>{SYSTEM_LABELS[system]}</h3>
        <div className="badge-row">
          <Badge tone={row.manual_correct ? "green" : "red"}>{row.manual_correct ? "correct" : "incorrect"}</Badge>
          <Badge tone={row.manual_hallucinated ? "red" : "neutral"}>
            {row.manual_hallucinated ? row.review_hallucination_severity : "not hallucinated"}
          </Badge>
          {row.abstained && <Badge tone="amber">abstained</Badge>}
        </div>
      </div>
      <p className="answer-text">{row.model_answer || row.abstention_reason || "No answer text recorded."}</p>
      {row.abstention_reason && (
        <div className="note-box">
          <strong>Abstention reason</strong>
          <p>{row.abstention_reason}</p>
        </div>
      )}
      <dl className="review-grid">
        <div>
          <dt>Citation support</dt>
          <dd>{row.manual_citation_supported == null ? "n/a" : row.manual_citation_supported ? "supported" : "unsupported"}</dd>
        </div>
        <div>
          <dt>Review note</dt>
          <dd>{row.manual_notes}</dd>
        </div>
      </dl>
      <details>
        <summary>Citations and retrieved chunks</summary>
        <div className="chunk-list">
          <div>
            <strong>Cited</strong>
            <p>{row.citations.length ? row.citations.join(", ") : "none"}</p>
          </div>
          <div>
            <strong>Retrieved</strong>
            <p>{row.retrieved_chunk_ids.length ? row.retrieved_chunk_ids.join(", ") : "none"}</p>
          </div>
        </div>
      </details>
    </article>
  );
}

function Badge({ children, tone }: { children: ReactNode; tone: "green" | "red" | "amber" | "neutral" }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function unique(values: string[]) {
  return Array.from(new Set(values)).sort();
}

function formatDataStamp(value: string) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}
