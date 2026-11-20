// Chat page logic. Plain JavaScript, no framework.
// All customer and model text is inserted with textContent (never as raw HTML),
// so it is always shown as text and can't run as HTML or script. That includes
// messages restored from an earlier visit.

const CHAT_URL = "/chat";
const CONVERSATIONS_URL = "/conversations/";
const REQUEST_TIMEOUT_MS = 60000;
const RESTORE_TIMEOUT_MS = 15000;
const DEFAULT_HINT = "Press Enter to send · Shift+Enter for a new line";

// The conversation ID is kept in sessionStorage: it survives a reload of this
// tab but is forgotten when the tab is closed. Only IDs of the shape the
// server generates are ever stored or sent.
const CONVERSATION_KEY = "support.conversationId";
const CONVERSATION_ID = /^[0-9a-f]{32}$/;

const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const newConversationButton = document.getElementById("new-conversation");
const messages = document.getElementById("messages");
const greeting = document.getElementById("greeting");
const hint = document.getElementById("hint");

let busy = false;
let conversationId = loadConversationId();

// --- Conversation ID --------------------------------------------------------

function loadConversationId() {
  try {
    const value = window.sessionStorage.getItem(CONVERSATION_KEY);
    return value && CONVERSATION_ID.test(value) ? value : null;
  } catch {
    return null; // Storage blocked (e.g. privacy settings): one conversation per page load.
  }
}

function setConversationId(value) {
  conversationId = typeof value === "string" && CONVERSATION_ID.test(value) ? value : null;
  try {
    if (conversationId) {
      window.sessionStorage.setItem(CONVERSATION_KEY, conversationId);
    } else {
      window.sessionStorage.removeItem(CONVERSATION_KEY);
    }
  } catch {
    // Not stored: the conversation still continues until the page is reloaded.
  }
}

// --- Rendering ----------------------------------------------------------------

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
  // kind: "user" | "answer" | "handoff" | "error" | "loading"
  const row = el("div", "message " + (kind === "user" ? "user" : "assistant " + kind));
  const bubble = el("div", "bubble");
  build(bubble);
  row.appendChild(bubble);
  messages.appendChild(row);
  scrollToBottom();
  return row;
}

function addUserMessage(text) {
  addMessage("user", (bubble) => bubble.appendChild(el("p", null, text)));
}

function showNotice(text) {
  messages.appendChild(el("p", "notice", text));
  scrollToBottom();
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
    if (!source || typeof source !== "object") continue;
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

function showError(text, label) {
  addMessage("error", (bubble) => {
    bubble.appendChild(el("span", "label", label || "Something went wrong"));
    bubble.appendChild(el("p", null, text));
  });
}

function showLoading(text) {
  return addMessage("loading", (bubble) => {
    const typing = el("span", "typing");
    typing.appendChild(el("span", null, text || "Searching our help articles"));
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
  newConversationButton.disabled = value;
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

function detailOr(data, fallback) {
  return data && typeof data.detail === "string" ? data.detail : fallback;
}

// --- Requests -----------------------------------------------------------------

async function askQuestion(question) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const body = conversationId ? { question, conversation_id: conversationId } : { question };

  try {
    const response = await fetch(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
    if (response.status === 429) {
      const message = detailOr(
        data,
        "We're receiving a lot of questions right now. Please wait a minute and try again."
      );
      return { error: message, label: "Please try again later" };
    }
    if (response.status === 404) {
      // The conversation expired, or the server restarted and lost it.
      return {
        error:
          "Your earlier conversation is no longer available, so your question was not sent. " +
          "Press Send to ask it again in a new conversation.",
        label: "Conversation ended",
        notSent: true,
        conversationGone: true,
      };
    }
    if (response.status === 409) {
      return {
        error: detailOr(data, "This conversation has reached its message limit.") +
          " Your question is still in the box below.",
        label: "Conversation full",
        notSent: true,
        conversationFull: true,
      };
    }
    if (response.status === 503) {
      return {
        error: detailOr(data, "The support service is unavailable right now. Please try again in a moment."),
        label: "Please try again later",
        notSent: true,
      };
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

  addUserMessage(question);
  input.value = "";
  autoResize();
  setHint(DEFAULT_HINT, false);
  setBusy(true);
  const loading = showLoading();

  const result = await askQuestion(question);

  loading.remove();
  if (result.data) {
    if (CONVERSATION_ID.test(result.data.conversation_id || "")) {
      setConversationId(result.data.conversation_id);
    }
    renderAnswer(result.data);
  } else {
    showError(result.error, result.label);
    if (result.conversationGone) {
      setConversationId(null); // The next message starts a new conversation.
    }
    if (result.notSent) {
      input.value = question; // Nothing was saved: let the customer send it again.
      autoResize();
    }
    if (result.conversationFull) {
      setHint("Select “New conversation” to keep asking questions.", true);
    }
  }
  setBusy(false);
  input.focus();
}

// --- Restoring and starting conversations ---------------------------------------

function isValidStoredMessage(message) {
  return (
    message !== null &&
    typeof message === "object" &&
    (message.role === "customer" || message.role === "assistant") &&
    typeof message.content === "string"
  );
}

async function restoreConversation() {
  if (!conversationId) return;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), RESTORE_TIMEOUT_MS);
  setBusy(true);
  const loading = showLoading("Loading your conversation");
  try {
    const response = await fetch(CONVERSATIONS_URL + encodeURIComponent(conversationId), {
      signal: controller.signal,
    });
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    loading.remove();

    if (response.ok && data && Array.isArray(data.messages)) {
      for (const message of data.messages.filter(isValidStoredMessage)) {
        if (message.role === "customer") {
          addUserMessage(message.content);
        } else {
          renderAnswer({ answer: message.content, sources: message.sources, escalate: message.escalate === true });
        }
      }
      return;
    }
    // Not restorable: start fresh rather than continue a conversation the
    // customer can't see.
    setConversationId(null);
    showNotice(
      response.status === 404
        ? "Your previous conversation is no longer available. A new conversation will start."
        : "We couldn't load your previous conversation. A new conversation will start."
    );
  } catch {
    loading.remove();
    setConversationId(null);
    showNotice("We couldn't load your previous conversation. A new conversation will start.");
  } finally {
    clearTimeout(timer);
    setBusy(false);
    input.focus();
  }
}

function startNewConversation() {
  if (busy) return;
  setConversationId(null);
  messages.replaceChildren(greeting);
  input.value = "";
  autoResize();
  setHint(DEFAULT_HINT, false);
  updateSendButton();
  input.focus();
}

form.addEventListener("submit", handleSubmit);
newConversationButton.addEventListener("click", startNewConversation);

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
restoreConversation();
