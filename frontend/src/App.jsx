import { useEffect, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const SAMPLE_QUESTIONS = [
  "孙志浩会哪些编程语言？",
  "孙志浩的英语水平怎么样？",
  "孙志浩有哪些实习经历？",
];

const DEFAULT_FORM = {
  question: SAMPLE_QUESTIONS[0],
  llm: "dashscope",
  top_k: 3,
  min_score: 0.03,
  show_prompt: false,
};

export default function App() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [answer, setAnswer] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Health check failed: HTTP ${response.status}`);
        }
        return response.json();
      })
      .then(setHealth)
      .catch((err) => setError(err.message));
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setAnswer(null);

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: form.question.trim(),
          llm: form.llm,
          top_k: Number(form.top_k),
          min_score: Number(form.min_score),
          show_prompt: form.show_prompt,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Request failed: HTTP ${response.status}`);
      }
      setAnswer(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function updateForm(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Single-topic RAG QA</p>
          <h1>RAG Copilot</h1>
          <p className="lede">
            Ask against your local knowledge base, get grounded answers, and inspect every citation.
          </p>
        </div>
        <StatusCard health={health} />
      </section>

      <section className="workspace">
        <form className="question-card" onSubmit={handleSubmit}>
          <div className="card-heading">
            <span>Question</span>
            <small>{API_BASE_URL}</small>
          </div>

          <textarea
            value={form.question}
            onChange={(event) => updateForm("question", event.target.value)}
            placeholder="输入一个问题..."
            rows={6}
          />

          <div className="samples">
            {SAMPLE_QUESTIONS.map((question) => (
              <button
                type="button"
                className="sample"
                key={question}
                onClick={() => updateForm("question", question)}
              >
                {question}
              </button>
            ))}
          </div>

          <div className="controls">
            <label>
              LLM
              <select value={form.llm} onChange={(event) => updateForm("llm", event.target.value)}>
                <option value="dashscope">DashScope</option>
                <option value="mock">Mock</option>
              </select>
            </label>
            <label>
              Top K
              <input
                type="number"
                min="1"
                max="20"
                value={form.top_k}
                onChange={(event) => updateForm("top_k", event.target.value)}
              />
            </label>
            <label>
              Min Score
              <input
                type="number"
                min="0"
                step="0.01"
                value={form.min_score}
                onChange={(event) => updateForm("min_score", event.target.value)}
              />
            </label>
          </div>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.show_prompt}
              onChange={(event) => updateForm("show_prompt", event.target.checked)}
            />
            Return generated prompt for debugging
          </label>

          <button className="submit" type="submit" disabled={loading || !form.question.trim()}>
            {loading ? "Asking..." : "Ask Copilot"}
          </button>
        </form>

        <section className="answer-card">
          {error && <div className="error">{error}</div>}
          {!error && !answer && (
            <div className="empty-state">
              <span>Ready</span>
              <p>Submit a question to see an answer with citations.</p>
            </div>
          )}
          {answer && <AnswerPanel answer={answer} />}
        </section>
      </section>
    </main>
  );
}

function StatusCard({ health }) {
  const indexReady = health?.index_ready;
  return (
    <aside className="status-card">
      <div className={`pulse ${indexReady ? "ready" : ""}`} />
      <div>
        <span>Backend</span>
        <strong>{health ? "Connected" : "Checking..."}</strong>
      </div>
      {health && (
        <dl>
          <div>
            <dt>Index</dt>
            <dd>{indexReady ? "Ready" : "Missing"}</dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>{health.model}</dd>
          </div>
        </dl>
      )}
    </aside>
  );
}

function AnswerPanel({ answer }) {
  return (
    <div className="answer-panel">
      <div className="answer-heading">
        <span>{answer.found_evidence ? "Grounded answer" : "No evidence"}</span>
        <small>{answer.citations.length} citation(s)</small>
      </div>

      <p className="answer-text">{answer.answer}</p>

      {answer.citations.length > 0 && (
        <div className="citations">
          <h2>Citations</h2>
          {answer.citations.map((citation, index) => (
            <article className="citation" key={`${citation.chunk_id}-${index}`}>
              <div>
                <strong>[{index + 1}] {citation.source}</strong>
                <span>chars {citation.start_char}-{citation.end_char}</span>
              </div>
              <meter min="0" max="1" value={Math.min(citation.score, 1)} />
              <small>score {citation.score.toFixed(4)}</small>
            </article>
          ))}
        </div>
      )}

      {answer.prompt && (
        <details className="prompt-box">
          <summary>Generated prompt</summary>
          <pre>{answer.prompt}</pre>
        </details>
      )}
    </div>
  );
}
