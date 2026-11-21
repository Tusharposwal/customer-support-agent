// Chat page logic. Plain JavaScript, no framework.
// All customer and model text is inserted with textContent (never as raw HTML),
// so it is always shown as text and can't run as HTML or script. That includes
// messages restored from an earlier visit.

const CHAT_URL = "/chat";
const CONVERSATIONS_URL = "/conversations/";
const CONFIRM_URL = "/cancellations/confirm";
const FEEDBACK_URL = "/feedback";
const FEEDBACK_TIMEOUT_MS = 15000;
// Shown while the conversation has an open hand-off record. Truthful on
// purpose: the record is internal, no support system is connected, so we
// never say that someone has been contacted or assigned.
const HANDOFF_STATUS_TEXT =
  "This conversation needs help from our support team and has been recorded for them. " +
  "No one has been assigned to it yet.";
// Longer than the server's worst case: a follow-up just after a restart can
// need rewrite 8 s + two embeddings 10 s each + rerank 5 s + chat 25 s = 58 s,
// and each budget limits each network phase separately, so it can run over.
// Giving up first would show "took too long" for a reply the server then saves.
const REQUEST_TIMEOUT_MS = 90000;
const RESTORE_TIMEOUT_MS = 15000;
const CONFIRM_TIMEOUT_MS = 15000;
const DEFAULT_HINT = "Press Enter to send · Shift+Enter for a new line";

// The conversation ID is kept in sessionStorage: it survives a reload of this
// tab but is forgotten when the tab is closed. Only IDs of the shape the
// server generates are ever stored or sent.
const CONVERSATION_KEY = "support.conversationId";
const CONVERSATION_ID = /^[0-9a-f]{32}$/;

// Cancellation offers: the action ID that confirms an offer is kept only in
// this tab's sessionStorage (like the conversation ID), keyed by order number
// and expiry time, so a pending offer can still be confirmed after a reload.
// The server never returns it again. It is sent only to CONFIRM_URL, never to
// /chat. "kept" records that the customer chose to keep the order.
const ACTIONS_KEY = "support.cancellationActions";
const ACTION_ID = CONVERSATION_ID;
const ORDER_ID = /^ORD-\d{4}$/;
const KEPT = "kept";

const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const newConversationButton = document.getElementById("new-conversation");
const messages = document.getElementById("messages");
const greeting = document.getElementById("greeting");
const hint = document.getElementById("hint");
const handoffStatus = document.getElementById("handoff-status");

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

function loadActions() {
  try {
    const parsed = JSON.parse(window.sessionStorage.getItem(ACTIONS_KEY) || "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function storedAction(key) {
  const value = loadActions()[key];
  return value === KEPT || (typeof value === "string" && ACTION_ID.test(value)) ? value : null;
}

function storeAction(key, value) {
  try {
    const actions = loadActions();
    if (value === null) {
      delete actions[key];
    } else {
      actions[key] = value;
    }
    window.sessionStorage.setItem(ACTIONS_KEY, JSON.stringify(actions));
  } catch {
    // Not stored: the offer can still be confirmed until the page is reloaded.
  }
}

function clearActions() {
  try {
    window.sessionStorage.removeItem(ACTIONS_KEY);
  } catch {
    // Nothing to clear.
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

// data: { answer, sources, escalate, cancellation, feedback: { conversationId, messageIndex, rating } }
function renderAnswer(data) {
  if (data.escalate) {
    addMessage("handoff", (bubble) => {
      bubble.appendChild(el("span", "label", "Needs human help"));
      bubble.appendChild(el("p", null, data.answer));
      if (data.feedback) renderFeedback(bubble, data.feedback);
    });
    return;
  }

  addMessage("answer", (bubble) => {
    bubble.appendChild(el("p", null, data.answer));
    renderSources(bubble, data.sources);
    if (data.cancellation) renderCancellation(bubble, data.cancellation);
    if (data.feedback) renderFeedback(bubble, data.feedback);
  });
}

function showHandoffStatus(open) {
  if (!handoffStatus) return;
  handoffStatus.textContent = open ? HANDOFF_STATUS_TEXT : "";
  handoffStatus.hidden = !open;
}

// --- Feedback on a reply ----------------------------------------------------------

function feedbackTarget(conversation, messageIndex, rating) {
  // Only a reply the server numbered, in a known conversation, can be rated.
  if (!conversation || !CONVERSATION_ID.test(conversation)) return null;
  if (!Number.isInteger(messageIndex) || messageIndex < 0) return null;
  return {
    conversationId: conversation,
    messageIndex,
    rating: rating === "positive" || rating === "negative" ? rating : null,
  };
}

function renderFeedback(bubble, target) {
  const box = el("div", "feedback");
  box.appendChild(el("span", "feedback-label", "Was this helpful?"));
  const buttons = {
    positive: el("button", "feedback-button", "Yes"),
    negative: el("button", "feedback-button", "No"),
  };
  const status = el("span", "feedback-status");
  status.setAttribute("role", "status");
  let current = target.rating;
  let sending = false;

  function show() {
    for (const [rating, button] of Object.entries(buttons)) {
      button.setAttribute("aria-pressed", String(rating === current));
      button.className = "feedback-button" + (rating === current ? " selected" : "");
      button.disabled = sending;
    }
  }

  for (const [rating, button] of Object.entries(buttons)) {
    button.type = "button";
    button.addEventListener("click", async () => {
      // One request at a time; the rating already saved needs no request.
      if (sending || rating === current) return;
      sending = true;
      show();
      status.textContent = "Saving…";
      const saved = await sendFeedback(target.conversationId, target.messageIndex, rating);
      sending = false;
      if (saved.ok) {
        current = rating;
        status.textContent = "Thanks for your feedback.";
      } else {
        status.textContent = saved.message;
      }
      show();
    });
    box.appendChild(button);
  }
  box.appendChild(status);
  show();
  bubble.appendChild(box);
}

async function sendFeedback(conversation, messageIndex, rating) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FEEDBACK_TIMEOUT_MS);
  const failed = "We couldn't save your feedback. Please try again.";
  try {
    const response = await fetch(FEEDBACK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Only the rating and which reply it is for: nothing about the customer.
      body: JSON.stringify({ conversation_id: conversation, message_index: messageIndex, rating }),
      signal: controller.signal,
    });
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (response.status === 200 && data && data.rating === rating) return { ok: true };
    if (response.status === 404) return { ok: false, message: detailOr(data, failed) };
    return { ok: false, message: failed };
  } catch {
    return { ok: false, message: failed };
  } finally {
    clearTimeout(timer);
  }
}

// --- Cancellation confirmation ---------------------------------------------------

function formatTime(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function offerKey(orderId, expiresAt) {
  return orderId + "|" + expiresAt;
}

// offer: { orderId, expiresAt, status, actionId, conversationId }
// status: "pending" (buttons shown when actionId is known), "kept",
// "cancelled", "expired" or "unavailable".
function renderCancellation(bubble, offer) {
  const box = el("div", "cancellation");
  box.appendChild(el("p", "cancellation-title", "Cancel order " + offer.orderId + "?"));
  const time = formatTime(offer.expiresAt);
  const expiry = el("p", "cancellation-expiry",
    time ? "This cancellation request expires at " + time + "." : "This cancellation request expires soon.");
  box.appendChild(expiry);
  const status = el("p", "cancellation-status");
  status.setAttribute("role", "status");
  box.appendChild(status);
  bubble.appendChild(box);

  function finish(state, text) {
    // A final state: the buttons go away and the outcome stays.
    box.className = "cancellation " + state;
    status.textContent = text;
    actions.remove(); // A no-op when the buttons were never shown.
    expiry.remove();
  }

  const actions = el("div", "cancellation-actions");
  if (offer.status === "cancelled") {
    return finish("done", "Order " + offer.orderId + " has been cancelled.");
  }
  if (offer.status === "expired") {
    return finish("expired", "This cancellation request has expired. Nothing was cancelled.");
  }
  if (offer.status === "kept") {
    return finish("kept", "You chose to keep order " + offer.orderId + ". Nothing was cancelled.");
  }
  if (!offer.actionId) {
    return finish("unavailable",
      "This request can't be confirmed from this page any more. Order " + offer.orderId +
      " has not been cancelled. Ask again if you still want to cancel it.");
  }

  const confirmButton = el("button", "danger", "Confirm cancellation");
  const keepButton = el("button", "secondary", "Keep order");
  confirmButton.type = "button";
  keepButton.type = "button";
  actions.append(confirmButton, keepButton);
  box.appendChild(actions);
  const key = offerKey(offer.orderId, offer.expiresAt);
  let confirming = false;

  keepButton.addEventListener("click", () => {
    if (confirming) return;
    storeAction(key, KEPT);
    finish("kept", "You chose to keep order " + offer.orderId + ". Nothing was cancelled.");
  });

  confirmButton.addEventListener("click", async () => {
    if (confirming) return; // One request at a time: double clicks do nothing.
    confirming = true;
    confirmButton.disabled = true;
    keepButton.disabled = true;
    box.className = "cancellation pending";
    status.textContent = "Cancelling order " + offer.orderId + "…";

    const result = await confirmCancellation(offer.conversationId, offer.actionId);
    confirming = false;
    if (result.retry) {
      // Nothing changed on the server (or we can't tell): the customer may try again.
      confirmButton.disabled = false;
      keepButton.disabled = false;
      box.className = "cancellation failed";
      status.textContent = result.message;
      return;
    }
    storeAction(key, null);
    finish(result.state, result.message);
  });
}

async function confirmCancellation(conversation, actionId) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CONFIRM_TIMEOUT_MS);
  const retryMessage =
    "We couldn't confirm the cancellation, so your order has not been cancelled. Please try again.";
  try {
    const response = await fetch(CONFIRM_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversation, action_id: actionId }),
      signal: controller.signal,
    });
    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }
    if (response.status === 200 && data && data.status === "cancelled" && typeof data.order_id === "string") {
      if (data.already_confirmed === true) {
        return { state: "done", message: "Order " + data.order_id + " was already cancelled. Nothing else was changed." };
      }
      return { state: "done", message: typeof data.message === "string" ? data.message : "Your order has been cancelled." };
    }
    if (response.status === 410) {
      return { state: "expired", message: detailOr(data, "This cancellation request has expired. Nothing was cancelled.") };
    }
    if (response.status === 404 || response.status === 409) {
      return { state: "failed", message: detailOr(data, "This order can't be cancelled. Nothing was cancelled.") };
    }
    return { retry: true, message: response.status === 503 ? detailOr(data, retryMessage) : retryMessage };
  } catch (error) {
    if (error.name === "AbortError") {
      return { retry: true, message: "The request took too long. Your order may not have been cancelled; please try again." };
    }
    return { retry: true, message: retryMessage };
  } finally {
    clearTimeout(timer);
  }
}

function pendingFromChat(data) {
  // The /chat response's pending_cancellation, if it is well formed.
  const pending = data.pending_cancellation;
  if (!pending || typeof pending !== "object" || data.escalate) return null;
  if (!ACTION_ID.test(pending.action_id || "") || !ORDER_ID.test(pending.order_id || "") ||
      typeof pending.expires_at !== "string") {
    return null;
  }
  return pending;
}

function offerFromRestore(message) {
  // A restored reply's cancellation status, joined with the action ID this tab kept.
  const c = message.cancellation;
  if (!c || typeof c !== "object" || !ORDER_ID.test(c.order_id || "") || typeof c.expires_at !== "string") {
    return null;
  }
  const stored = storedAction(offerKey(c.order_id, c.expires_at));
  let status = c.status;
  if (status === "pending" && stored === KEPT) status = "kept";
  if (!["pending", "cancelled", "expired", "kept"].includes(status)) return null;
  return {
    orderId: c.order_id,
    expiresAt: c.expires_at,
    status,
    actionId: status === "pending" && stored !== KEPT ? stored : null,
    conversationId,
  };
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

function isSavedReply(data) {
  return (
    isValidAnswer(data) &&
    CONVERSATION_ID.test(data.conversation_id || "") &&
    Number.isInteger(data.message_index)
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
    // A reply the server saved always comes with its conversation and position.
    // A 5xx body without them (the server's last-resort error) was NOT saved:
    // show it as an error and keep the question, never as a recorded hand-off.
    if (response.status >= 500 && !isSavedReply(data)) {
      return {
        error: isValidAnswer(data) ? data.answer
          : "The support service is unavailable right now. Please try again in a moment.",
        label: "Please try again later",
        notSent: true,
      };
    }
    // The server sends a saved hand-off body on 500 as well, so show it if present.
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
    const pending = pendingFromChat(result.data);
    let cancellation = null;
    if (pending && conversationId) {
      storeAction(offerKey(pending.order_id, pending.expires_at), pending.action_id);
      cancellation = {
        orderId: pending.order_id,
        expiresAt: pending.expires_at,
        status: "pending",
        actionId: pending.action_id,
        conversationId,
      };
    }
    // Only these fields are rendered; the action ID never becomes message text.
    renderAnswer({
      answer: result.data.answer,
      sources: result.data.sources,
      escalate: result.data.escalate,
      cancellation,
      feedback: feedbackTarget(conversationId, result.data.message_index, null),
    });
    // An escalated reply is saved together with the conversation's hand-off
    // record; only a saved reply (with its position) proves that.
    if (result.data.escalate === true && isSavedReply(result.data)) showHandoffStatus(true);
  } else {
    showError(result.error, result.label);
    if (result.conversationGone) {
      setConversationId(null); // The next message starts a new conversation.
      clearActions();
      showHandoffStatus(false);
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
          renderAnswer({
            answer: message.content,
            sources: message.sources,
            escalate: message.escalate === true,
            cancellation: message.escalate === true ? null : offerFromRestore(message),
            feedback: feedbackTarget(conversationId, message.message_index, message.feedback),
          });
        }
      }
      showHandoffStatus(Boolean(data.handoff && data.handoff.status === "open"));
      return;
    }
    // Not restorable: start fresh rather than continue a conversation the
    // customer can't see.
    setConversationId(null);
    clearActions();
    showNotice(
      response.status === 404
        ? "Your previous conversation is no longer available. A new conversation will start."
        : "We couldn't load your previous conversation. A new conversation will start."
    );
  } catch {
    loading.remove();
    setConversationId(null);
    clearActions();
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
  clearActions();
  showHandoffStatus(false);
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
