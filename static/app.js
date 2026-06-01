const messages = document.querySelector("#messages");
const form = document.querySelector("#questionForm");
const input = document.querySelector("#questionInput");
const askButton = document.querySelector("#askButton");
const clearButton = document.querySelector("#clearButton");
let conversationHistory = [];

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderAnswer(text) {
  const paragraphs = text
    .trim()
    .split(/\n{2,}/)
    .map((part) => `<p>${escapeHtml(part).replaceAll("\n", "<br>")}</p>`);
  return paragraphs.join("");
}

function addMessage(role, content, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <div class="avatar">${role === "user" ? "Y" : "R"}</div>
    <div class="bubble">
      ${renderAnswer(content)}
      ${renderSources(sources)}
    </div>
  `;
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function remember(role, content) {
  conversationHistory.push({ role, content });
  conversationHistory = conversationHistory.slice(-8);
}

function renderSources(sources) {
  if (!sources || sources.length === 0) {
    return "";
  }

  const items = sources
    .slice(0, 5)
    .map((source) => {
      const label = escapeHtml(source.title || source.source || "Source");
      const loader = escapeHtml(source.loader || "document");
      const href = source.url ? escapeHtml(source.url) : "";
      const snippet = source.snippet ? `<p>${escapeHtml(source.snippet)}</p>` : "";
      const method = source.retrieval_type ? `<span>${escapeHtml(source.retrieval_type)}</span>` : "";

      if (href) {
        return `
          <a class="source-card" href="${href}" target="_blank" rel="noreferrer">
            <strong>${label}</strong>
            <small>${loader}${method}</small>
            ${snippet}
          </a>
        `;
      }

      return `
        <div class="source-card">
          <strong>${label}</strong>
          <small>${loader}${method}</small>
          ${snippet}
        </div>
      `;
    })
    .join("");

  return `<div class="sources">${items}</div>`;
}

function setBusy(isBusy) {
  askButton.disabled = isBusy;
  input.disabled = isBusy;
  askButton.textContent = isBusy ? "Thinking" : "Ask";
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    document.querySelector("#modelName").textContent = status.model || "Unknown";
    document.querySelector("#retrieverK").textContent = `${status.retriever_k || "-"} chunks`;
  } catch {
    document.querySelector("#modelName").textContent = "Offline";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();

  if (!question) {
    input.focus();
    return;
  }

  addMessage("user", question);
  remember("user", question);
  input.value = "";
  setBusy(true);
  const thinking = addMessage("assistant thinking", "Searching your indexed sources...");

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, history: conversationHistory.slice(0, -1) }),
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "The assistant could not answer.");
    }

    thinking.remove();
    const answer = result.answer || "I do not have enough information.";
    addMessage("assistant", answer, result.sources || []);
    remember("assistant", answer);
  } catch (error) {
    thinking.remove();
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    form.requestSubmit();
  }
});

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  conversationHistory = [];
  addMessage(
    "assistant",
    "Ask me about your indexed documents. I will answer from retrieved context."
  );
  input.focus();
});

loadStatus();
