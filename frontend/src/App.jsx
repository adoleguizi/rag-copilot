import { useEffect, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const SAMPLE_QUESTIONS = [
  "这份资料主要讲了什么？",
  "资料里有哪些关键事实？",
  "有哪些内容可以作为引用依据？",
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
  const [documents, setDocuments] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    refreshBackendState();
  }, []);

  async function refreshBackendState() {
    try {
      const [healthResponse, documentsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/health`),
        fetch(`${API_BASE_URL}/documents`),
      ]);
      if (!healthResponse.ok) {
        throw new Error(`Health check failed: HTTP ${healthResponse.status}`);
      }
      if (!documentsResponse.ok) {
        throw new Error(`Document list failed: HTTP ${documentsResponse.status}`);
      }
      setHealth(await healthResponse.json());
      setDocuments(await documentsResponse.json());
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleUpload(event) {
    event.preventDefault();
    if (selectedFiles.length === 0) {
      setError("请选择至少一个文档。");
      return;
    }

    setUploading(true);
    setError("");
    setNotice("");

    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append("files", file));

      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Upload failed: HTTP ${response.status}`);
      }

      setSelectedFiles([]);
      setNotice(`已上传 ${payload.files.length} 个文档。请重建索引后再提问。`);
      await refreshBackendState();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleRebuildIndex() {
    setRebuilding(true);
    setError("");
    setNotice("");

    try {
      const response = await fetch(`${API_BASE_URL}/index/rebuild`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chunk_size: 450, chunk_overlap: 80 }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Rebuild failed: HTTP ${response.status}`);
      }

      setNotice(
        `索引已重建：${payload.files} 个源文件，${payload.documents} 个文档单元，${payload.chunks} 个 chunks。PDF 会按页计作文档单元。`,
      );
      await refreshBackendState();
    } catch (err) {
      setError(err.message);
    } finally {
      setRebuilding(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setNotice("");
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

  async function handleDeleteDocument(name) {
    setError("");
    setNotice("");

    try {
      const response = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Delete failed: HTTP ${response.status}`);
      }

      setNotice(`已删除 ${payload.name}。请重建索引后再提问。`);
      await refreshBackendState();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Single-topic RAG QA</p>
          <h1>RAG Copilot</h1>
          <p className="lede">
            上传一组资料，重建本地索引，然后基于引用来源向百炼模型提问。
          </p>
        </div>
        <StatusCard health={health} />
      </section>

      <section className="workspace">
        <div className="left-stack">
          <section className="upload-card">
            <div className="card-heading">
              <span>Documents</span>
              <small>{documents.length} file(s)</small>
            </div>

            <form onSubmit={handleUpload}>
              <label className="file-picker">
                <input
                  type="file"
                  multiple
                  accept=".txt,.md,.markdown,.csv,.pdf"
                  onChange={(event) => setSelectedFiles(Array.from(event.target.files || []))}
                />
                <span>选择 PDF / Markdown / TXT / FAQ CSV</span>
              </label>

              {selectedFiles.length > 0 && (
                <ul className="pending-files">
                  {selectedFiles.map((file) => (
                    <li key={`${file.name}-${file.size}`}>
                      {file.name}
                      <small>{formatBytes(file.size)}</small>
                    </li>
                  ))}
                </ul>
              )}

              <div className="document-actions">
                <button className="secondary" type="submit" disabled={uploading}>
                  {uploading ? "Uploading..." : "Upload"}
                </button>
                <button className="secondary dark" type="button" onClick={handleRebuildIndex} disabled={rebuilding}>
                  {rebuilding ? "Rebuilding..." : "Rebuild Index"}
                </button>
              </div>
            </form>

            <DocumentList documents={documents} onDelete={handleDeleteDocument} />
          </section>

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
        </div>

        <section className="answer-card">
          {notice && <div className="notice">{notice}</div>}
          {error && <div className="error">{error}</div>}
          {!error && !notice && !answer && (
            <div className="empty-state">
              <span>Ready</span>
              <p>上传资料并重建索引后，提交问题查看答案和引用。</p>
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

function DocumentList({ documents, onDelete }) {
  if (documents.length === 0) {
    return <p className="document-empty">还没有上传文档。</p>;
  }

  return (
    <ul className="document-list">
      {documents.map((document) => (
        <li key={document.name}>
          <span>{document.name}</span>
            <div className="document-meta">
              <small>{formatBytes(document.size)}</small>
              <button type="button" onClick={() => onDelete(document.name)}>
              Remove from index
              </button>
            </div>
          </li>
        ))}
    </ul>
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
                <span>
                  chars {citation.start_char}-{citation.end_char}
                </span>
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

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
