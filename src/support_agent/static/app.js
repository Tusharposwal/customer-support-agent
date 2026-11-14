// Chat page logic. Plain JavaScript, no framework.
// All customer and model text is inserted with textContent (never as raw HTML),
// so it is always shown as text and can't run as HTML or script.

const CHAT_URL = "/chat";
const REQUEST_TIMEOUT_MS = 60000;
const DEFAULT_HINT = "Press Enter to send · Shift+Enter for a new line";

const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const messages = document.getElementById("messages");
const hint = document.getElementById("hint");

let busy = false;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(kind, build) {
  // kind: "user" | "assistant" | "handoff" | "error"
  const row = el("div", "message " + (kind === "user" ? "user" : "assistant " + kind));
  const bubble = el("div", "bubble");
  build(bubble);
  row.appendChild(bubble);
  messages.appendChild(row);
  scrollToBottom();
  return row;
}

function isSafeUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:";
  } catch {
    return false;
  }
}

function renderSources(bubble, sources) {
  if (!Array.isArray(sources) || sources.length === 0) return;

  const box = el("div", "sources");
  box.appendChild(el("p", "sources-title", "Sources"));
  const list = el("ul");

  for (const source of sources) {
    const item = el("li");
    const title = String(source.title || source.id || "Support article");
    if (isSafeUrl(source.url)) {
      const link = el("a", null, title);
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      item.appendChild(link);
      item.appendChild(el("span", "source-url", source.url));
    } else {
      item.appendChild(el("span", null, title));
    }
    list.appendChild(item);
  }

  box.appendChild(list);
  bubble.appendChild(box);
}

function renderAnswer(data) {
  if (data.escalate) {
    addMessage("handoff", (bubble) => {
      bubble.appendChild(el("span", "label", "Handed off to a human agent"));
      bubble.appendChild(el("p", null, data.answer));
    });
    return;
  }

  addMessage("answer", (bubble) => {
    bubble.appendChild(el("p", null, data.answer));
    renderSources(bubble, data.sources);
  });
}

function showError(text) {
  addMessage("error", (bubble) => {
    bubble.appendChild(el("span", "label", "Something went wrong"));
    bubble.appendChild(el("p", null, text));
  });
}

function showLoading() {
  return addMessage("loading", (bubble) => {
    const typing = el("span", "typing");
    typing.appendChild(el("span", null, "Searching our help articles"));
    const dots = el("span", "dots");
    dots.setAttribute("aria-hidden", "true");
    dots.append(el("span"), el("span"), el("span"));
    typing.appendChild(dots);
    bubble.appendChild(typing);
  });
}

function setHint(text, warn) {
  hint.textContent = text;
  hint.classList.toggle("warn", Boolean(warn));
}

function setBusy(value) {
  busy = value;
  input.disabled = value;
  updateSendButton();
}

function updateSendButton() {
  sendButton.disabled = busy || input.value.trim() === "";
}

function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

function isValidAnswer(data) {
  return (
    data !== null &&
    typeof data === "object" &&
    typeof data.answer === "string" &&
    typeof data.escalate === "boolean"
  );
}

async function askQuestion(question) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });

    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (response.status === 422) {
      return { error: "Please enter a question of up to 1000 characters." };
    }
    // The server sends a safe hand-off body on 500 as well, so show it if present.
    if (isValidAnswer(data)) {
      return { data };
    }
    return { error: "The support service is unavailable right now. Please try again in a moment." };
  } catch (error) {
    if (error.name === "AbortError") {
      return { error: "The request took too long. Please try again." };
    }
    return { error: "We couldn't reach the support service. Check your connection and try again." };
  } finally {
    clearTimeout(timer);
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  if (busy) return;

  const question = input.value.trim();
  if (question === "") {
    setHint("Please type a question first.", true);
    input.focus();
    return;
  }

  addMessage("user", (bubble) => bubble.appendChild(el("p", null, question)));
  input.value = "";
  autoResize();
  setHint(DEFAULT_HINT, false);
  setBusy(true);
  const loading = showLoading();

  const result = await askQuestion(question);

  loading.remove();
  if (result.data) {
    renderAnswer(result.data);
  } else {
    showError(result.error);
  }
  setBusy(false);
  input.focus();
}

form.addEventListener("submit", handleSubmit);

input.addEventListener("input", () => {
  autoResize();
  updateSendButton();
  if (hint.classList.contains("warn") && input.value.trim() !== "") {
    setHint(DEFAULT_HINT, false);
  }
});

input.addEventListener("keydown", (event) => {
  // Enter sends; Shift+Enter adds a new line; ignore Enter while typing with an IME.
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.focus();
