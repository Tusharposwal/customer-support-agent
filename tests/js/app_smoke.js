// Behavioural checks of static/app.js with a tiny fake DOM: no browser, no network,
// no npm packages. Run by tests/test_frontend.py when Node.js is installed.
// Any use of innerHTML / insertAdjacentHTML / outerHTML throws.
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const SCRIPT = fs.readFileSync(process.argv[2], "utf8");
const ID = "0123456789abcdef0123456789abcdef";
const NEW_ID = "fedcba9876543210fedcba9876543210";
const XSS = '<img src=x onerror="alert(1)"><script>alert(2)</script>';

class Node {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.parent = null; this._text = "";
    this.className = ""; this.attributes = {}; this.style = {}; this.listeners = {};
    this.value = ""; this.disabled = false; this.scrollTop = 0; this.scrollHeight = 0;
    const self = this;
    this.classList = {
      toggle(name, on) { const set = new Set(self.className.split(" ").filter(Boolean)); on ? set.add(name) : set.delete(name); self.className = [...set].join(" "); },
      contains(name) { return self.className.split(" ").includes(name); },
    };
  }
  set textContent(v) { this.children = []; this._text = String(v); }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set innerHTML(v) { throw new Error("innerHTML used"); }
  set outerHTML(v) { throw new Error("outerHTML used"); }
  insertAdjacentHTML() { throw new Error("insertAdjacentHTML used"); }
  appendChild(child) { if (child.parent) child.remove(); child.parent = this; this.children.push(child); return child; }
  append(...nodes) { nodes.forEach((n) => this.appendChild(n)); }
  remove() { if (this.parent) { this.parent.children = this.parent.children.filter((c) => c !== this); this.parent = null; } }
  replaceChildren(...nodes) { this.children.forEach((c) => (c.parent = null)); this.children = []; this.append(...nodes); }
  setAttribute(k, v) { this.attributes[k] = v; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  focus() {}
  requestSubmit() { return this.listeners.submit({ preventDefault() {} }); }
  all() { return [this, ...this.children.flatMap((c) => c.all())]; }
}

function setup({ stored, fetchImpl, actions }) {
  const byId = {};
  for (const id of ["chat-form", "question", "send", "new-conversation", "messages", "hint", "handoff-status"]) byId[id] = new Node("div");
  byId["handoff-status"].hidden = true;
  const greeting = new Node("div"); greeting.id = "greeting"; greeting.textContent = "Hi!";
  byId.greeting = greeting;
  byId.messages.appendChild(greeting);
  const storage = new Map(stored ? [["support.conversationId", stored]] : []);
  if (actions) storage.set("support.cancellationActions", JSON.stringify(actions));
  const requests = [];
  const context = {
    document: { getElementById: (id) => byId[id], createElement: (tag) => new Node(tag) },
    window: { sessionStorage: {
      getItem: (k) => (storage.has(k) ? storage.get(k) : null),
      setItem: (k, v) => storage.set(k, String(v)),
      removeItem: (k) => storage.delete(k),
    } },
    fetch: async (url, options = {}) => { requests.push({ url, options }); return fetchImpl(url, options); },
    AbortController, URL, setTimeout, clearTimeout, console,
  };
  vm.createContext(context);
  vm.runInContext(SCRIPT, context);
  return { byId, storage, requests, greeting };
}

const respond = (status, body) => ({ status, ok: status >= 200 && status < 300, json: async () => body });
const settle = () => new Promise((r) => setTimeout(r, 20));
const texts = (byId) => byId.messages.children.map((c) => c.textContent);

// --- Cancellation helpers ---
const ACTION = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const EXPIRES = "2026-11-21T10:15:00+00:00";
const OFFER_TEXT = "Order ORD-1001 is still processing, so it can be cancelled, but it has not been cancelled yet.";
const offerReply = (extra = {}) => ({
  answer: OFFER_TEXT, sources: [], escalate: false, reason: "cancellation_available", conversation_id: NEW_ID,
  pending_cancellation: { action_id: ACTION, order_id: "ORD-1001", expires_at: EXPIRES }, ...extra,
});
const buttons = (byId) => byId.messages.all().filter((n) => n.tagName === "button");
const button = (byId, label) => buttons(byId).find((b) => b.textContent === label);
const box = (byId) => byId.messages.all().find((n) => n.className.split(" ").includes("cancellation"));
const confirms = (requests) => requests.filter((r) => r.url === "/cancellations/confirm");
const actionsIn = (storage) => JSON.parse(storage.get("support.cancellationActions") || "{}");
const KEY = "ORD-1001|" + EXPIRES;

// A page with one fresh cancellation offer from /chat; confirm requests get `confirmImpl`.
async function offered(confirmImpl, chatImpl = () => respond(200, offerReply())) {
  const page = setup({
    stored: null,
    fetchImpl: (url, o) => (url === "/cancellations/confirm" ? confirmImpl(o) : chatImpl(o)),
  });
  page.byId.question.value = "Please cancel ORD-1001";
  await page.byId["chat-form"].requestSubmit();
  return page;
}

(async () => {
  // 1. Restore an earlier conversation, including HTML-looking text, as plain text.
  {
    const { byId, requests } = setup({
      stored: ID,
      fetchImpl: () => respond(200, { conversation_id: ID, message_limit: 40, messages: [
        { role: "customer", content: XSS },
        { role: "assistant", content: "Answer " + XSS, sources: [
          { id: "KB-1", title: XSS, url: "https://support.example.com/a" },
          { id: "KB-2", title: "bad", url: "javascript:alert(3)" }], escalate: false, reason: "answered" },
        { role: "assistant", content: "handed off", sources: [], escalate: true, reason: "x" },
        { role: "system", content: "ignored" },
      ] }),
    });
    await settle();
    assert.strictEqual(requests[0].url, "/conversations/" + ID);
    assert.deepStrictEqual(texts(byId).slice(0, 2), ["Hi!", XSS]);
    const all = byId.messages.all();
    const links = all.filter((n) => n.tagName === "a");
    assert.strictEqual(links.length, 1, "javascript: URL must not become a link");
    assert.strictEqual(links[0].href, "https://support.example.com/a");
    assert.ok(!all.some((n) => ["img", "script"].includes(n.tagName)), "no element created from text");
    assert.ok(texts(byId).some((t) => t.includes("Needs human help")));
    assert.ok(!texts(byId).some((t) => t.includes("ignored")));
    assert.strictEqual(byId.question.disabled, false);
    console.log("ok restore + escaping");
  }

  // 2. A follow-up sends the stored ID; a first message sends none and stores the new ID.
  {
    const { byId, storage, requests } = setup({
      stored: null,
      fetchImpl: (url, o) => respond(200, { answer: "A", sources: [], escalate: false, reason: "answered", conversation_id: NEW_ID }),
    });
    await settle();
    assert.strictEqual(requests.length, 0, "no restore without an ID");
    byId.question.value = "first";
    await byId["chat-form"].requestSubmit();
    assert.deepStrictEqual(JSON.parse(requests[0].options.body), { question: "first" });
    assert.strictEqual(storage.get("support.conversationId"), NEW_ID);
    byId.question.value = "second";
    await byId["chat-form"].requestSubmit();
    assert.deepStrictEqual(JSON.parse(requests[1].options.body), { question: "second", conversation_id: NEW_ID });
    console.log("ok first message + follow-up");
  }

  // 3. 404 on a follow-up: ID forgotten, question kept in the box, next send starts fresh.
  {
    let status = 404;
    const { byId, storage, requests } = setup({
      stored: ID,
      fetchImpl: (url, o) => (o.method === "POST" ? respond(status, { detail: "gone" }) : respond(200, { messages: [] })),
    });
    await settle();
    byId.question.value = "follow-up";
    await byId["chat-form"].requestSubmit();
    assert.ok(!storage.has("support.conversationId"));
    assert.strictEqual(byId.question.value, "follow-up");
    status = 200;
    await byId["chat-form"].requestSubmit();
    assert.deepStrictEqual(JSON.parse(requests.at(-1).options.body), { question: "follow-up" });
    console.log("ok 404 recovery");
  }

  // 4. 409 and 503 keep the ID and the question; nothing is shown as answered.
  for (const status of [409, 503]) {
    const { byId, storage } = setup({
      stored: ID,
      fetchImpl: (url, o) => (o.method === "POST" ? respond(status, { detail: "Server says no." }) : respond(200, { messages: [] })),
    });
    await settle();
    byId.question.value = "q";
    await byId["chat-form"].requestSubmit();
    assert.strictEqual(storage.get("support.conversationId"), ID);
    assert.strictEqual(byId.question.value, "q");
    assert.ok(texts(byId).at(-1).includes("Server says no."));
  }
  console.log("ok 409 / 503");

  // 5. Restore failure: start fresh instead of continuing an invisible conversation.
  for (const impl of [() => respond(404, { detail: "x" }), () => respond(503, {}), () => { throw new TypeError("offline"); }]) {
    const { byId, storage } = setup({ stored: ID, fetchImpl: impl });
    await settle();
    assert.ok(!storage.has("support.conversationId"));
    assert.ok(texts(byId).at(-1).includes("A new conversation will start."));
  }
  console.log("ok restore failures");

  // 6. Malformed stored IDs are ignored and never sent.
  {
    const { requests, storage } = setup({ stored: "../../admin", fetchImpl: () => respond(200, {}) });
    await settle();
    assert.strictEqual(requests.length, 0);
    console.log("ok malformed stored ID ignored");
  }

  // 7. New conversation: back to the greeting, ID forgotten.
  {
    const { byId, storage } = setup({
      stored: ID,
      fetchImpl: () => respond(200, { messages: [{ role: "customer", content: "old" }] }),
    });
    await settle();
    assert.strictEqual(byId.messages.children.length, 2);
    byId["new-conversation"].listeners.click();
    assert.deepStrictEqual(texts(byId), ["Hi!"]);
    assert.ok(!storage.has("support.conversationId"));
    console.log("ok new conversation");
  }
  // 8. A pending cancellation is shown with the order, the expiry and two buttons;
  //    the action ID is kept for the confirm call only, never shown or sent to /chat.
  {
    const { byId, storage, requests } = await offered(() => respond(500, {}),
      () => respond(200, offerReply()));
    const component = box(byId);
    assert.ok(component, "confirmation component shown");
    assert.ok(component.textContent.includes("Cancel order ORD-1001?"));
    assert.ok(component.textContent.includes("expires at"));
    assert.deepStrictEqual(buttons(byId).map((b) => b.textContent), ["Confirm cancellation", "Keep order"]);
    assert.ok(!byId.messages.textContent.includes(ACTION), "action ID never displayed");
    assert.deepStrictEqual(actionsIn(storage), { [KEY]: ACTION });
    byId.question.value = "yes";
    await byId["chat-form"].requestSubmit();
    assert.deepStrictEqual(JSON.parse(requests.at(-1).options.body), { question: "yes", conversation_id: NEW_ID });
    assert.ok(requests.every((r) => r.url === "/cancellations/confirm" || !String(r.options.body).includes(ACTION)));
    assert.strictEqual(confirms(requests).length, 0, "typing yes never confirms");
    console.log("ok pending cancellation display");
  }

  // 8b. A malformed pending_cancellation shows no buttons.
  {
    const { byId } = await offered(() => respond(500, {}), () => respond(200, offerReply({
      pending_cancellation: { action_id: "not-an-id", order_id: "ORD-1001", expires_at: EXPIRES } })));
    assert.strictEqual(buttons(byId).length, 0);
    assert.ok(texts(byId).at(-1).includes(OFFER_TEXT));
    console.log("ok malformed offer ignored");
  }

  // 9. Confirm: one call with both IDs, success shown, buttons removed, ID forgotten.
  {
    const { byId, storage, requests } = await offered(() => respond(200, {
      status: "cancelled", order_id: "ORD-1001", cancelled_at: EXPIRES, already_confirmed: false,
      message: "Order ORD-1001 has been cancelled." }));
    await button(byId, "Confirm cancellation").listeners.click();
    assert.strictEqual(confirms(requests).length, 1);
    assert.deepStrictEqual(JSON.parse(confirms(requests)[0].options.body), { conversation_id: NEW_ID, action_id: ACTION });
    assert.strictEqual(confirms(requests)[0].options.method, "POST");
    assert.ok(box(byId).textContent.includes("Order ORD-1001 has been cancelled."));
    assert.ok(box(byId).className.includes("done"));
    assert.strictEqual(buttons(byId).length, 0);
    assert.deepStrictEqual(actionsIn(storage), {});
    console.log("ok successful confirmation");
  }

  // 10. Duplicate clicks while the request is in flight do nothing.
  {
    let release;
    const { byId, requests } = await offered(() => new Promise((resolve) => { release = resolve; }));
    const confirm = button(byId, "Confirm cancellation");
    const keep = button(byId, "Keep order");
    const first = confirm.listeners.click();
    confirm.listeners.click();
    confirm.listeners.click();
    keep.listeners.click(); // Keeping is ignored while confirming.
    assert.strictEqual(confirms(requests).length, 1);
    assert.strictEqual(confirm.disabled, true);
    assert.strictEqual(keep.disabled, true);
    assert.ok(box(byId).textContent.includes("Cancelling order ORD-1001"));
    release(respond(200, { status: "cancelled", order_id: "ORD-1001", already_confirmed: false,
      message: "Order ORD-1001 has been cancelled." }));
    await first;
    assert.strictEqual(confirms(requests).length, 1);
    assert.ok(box(byId).textContent.includes("has been cancelled"));
    console.log("ok duplicate-click prevention");
  }

  // 11. Keep order: nothing is sent, the choice is remembered for this tab.
  {
    const { byId, storage, requests } = await offered(() => { throw new Error("must not confirm"); });
    button(byId, "Keep order").listeners.click();
    assert.strictEqual(confirms(requests).length, 0);
    assert.strictEqual(buttons(byId).length, 0);
    assert.ok(box(byId).textContent.includes("You chose to keep order ORD-1001. Nothing was cancelled."));
    assert.deepStrictEqual(actionsIn(storage), { [KEY]: "kept" });
    console.log("ok keep order");
  }

  // 12. Expired (410): the server's explanation, no buttons.
  {
    const { byId, storage } = await offered(() => respond(410, { detail: "This cancellation request has expired, so nothing was cancelled." }));
    await button(byId, "Confirm cancellation").listeners.click();
    assert.ok(box(byId).textContent.includes("has expired, so nothing was cancelled."));
    assert.ok(box(byId).className.includes("expired"));
    assert.strictEqual(buttons(byId).length, 0);
    assert.deepStrictEqual(actionsIn(storage), {});
    console.log("ok expired action");
  }

  // 13. Already confirmed (a repeat): said so, nothing else changed.
  {
    const { byId } = await offered(() => respond(200, { status: "cancelled", order_id: "ORD-1001",
      cancelled_at: EXPIRES, already_confirmed: true, message: "Order ORD-1001 has been cancelled." }));
    await button(byId, "Confirm cancellation").listeners.click();
    assert.ok(box(byId).textContent.includes("Order ORD-1001 was already cancelled. Nothing else was changed."));
    assert.strictEqual(buttons(byId).length, 0);
    console.log("ok already confirmed");
  }

  // 14. Failures: 503 / network / malformed 200 can be retried; 409 and 404 are final.
  {
    const outcomes = [() => respond(503, { detail: "We can't save conversations right now." }),
      () => { throw new TypeError("offline"); },
      () => respond(200, { unexpected: true }),
      () => respond(200, { status: "cancelled", order_id: "ORD-1001", already_confirmed: false,
        message: "Order ORD-1001 has been cancelled." })];
    const { byId, requests } = await offered(() => outcomes.shift()());
    await button(byId, "Confirm cancellation").listeners.click();
    assert.ok(box(byId).textContent.includes("We can't save conversations right now."));
    assert.ok(box(byId).className.includes("failed"));
    for (const expected of ["has not been cancelled. Please try again.", "has not been cancelled. Please try again."]) {
      assert.strictEqual(button(byId, "Confirm cancellation").disabled, false, "retry allowed");
      assert.strictEqual(button(byId, "Keep order").disabled, false);
      await button(byId, "Confirm cancellation").listeners.click();
      assert.ok(box(byId).textContent.includes(expected));
    }
    await button(byId, "Confirm cancellation").listeners.click();
    assert.ok(box(byId).textContent.includes("has been cancelled."));
    assert.strictEqual(confirms(requests).length, 4);

    for (const [status, detail] of [[409, "This order can no longer be cancelled because its status has changed."],
      [404, "This cancellation request was not found, so nothing was cancelled."]]) {
      const page = await offered(() => respond(status, { detail }));
      await button(page.byId, "Confirm cancellation").listeners.click();
      assert.ok(box(page.byId).textContent.includes(detail));
      assert.strictEqual(buttons(page.byId).length, 0);
    }
    console.log("ok failed confirmation");
  }

  // 15. Reload: statuses come from GET /conversations, the action ID from this tab's storage.
  {
    const offerMessage = (status, expires, extra = {}) => ({
      role: "assistant", content: OFFER_TEXT, sources: [], escalate: false, reason: "cancellation_available",
      cancellation: { order_id: "ORD-1001", status, expires_at: expires, cancelled_at: null, ...extra } });
    const E = (n) => "2026-11-21T10:1" + n + ":00+00:00";
    const storage = { [`ORD-1001|${E(1)}`]: ACTION, [`ORD-1001|${E(3)}`]: "kept" };
    const restored = setup({
      stored: ID,
      actions: storage,
      fetchImpl: (url, o) => url === "/cancellations/confirm"
        ? respond(200, { status: "cancelled", order_id: "ORD-1001", already_confirmed: false, message: "Order ORD-1001 has been cancelled." })
        : respond(200, { conversation_id: ID, message_limit: 40, messages: [
          { role: "customer", content: "cancel ORD-1001" },
          offerMessage("pending", E(1)),  // pending, ID kept in this tab: buttons
          offerMessage("pending", E(2)),  // pending, no ID here (other tab / storage lost)
          offerMessage("pending", E(3)),  // the customer chose to keep it
          offerMessage("expired", E(4)),
          offerMessage("cancelled", E(5), { cancelled_at: E(5) }),
          { role: "assistant", content: "Plain answer", sources: [], escalate: false, reason: "answered", cancellation: null },
        ] }),
    });
    await settle();
    const boxes = restored.byId.messages.all().filter((n) => n.className.split(" ").includes("cancellation"));
    assert.strictEqual(boxes.length, 5);
    assert.deepStrictEqual(buttons(restored.byId).map((b) => b.textContent), ["Confirm cancellation", "Keep order"]);
    assert.ok(boxes[1].textContent.includes("can't be confirmed from this page any more"));
    assert.ok(boxes[2].textContent.includes("You chose to keep order ORD-1001"));
    assert.ok(boxes[3].textContent.includes("has expired. Nothing was cancelled."));
    assert.ok(boxes[4].textContent.includes("Order ORD-1001 has been cancelled."));
    await button(restored.byId, "Confirm cancellation").listeners.click();
    const sent = confirms(restored.requests);
    assert.deepStrictEqual(JSON.parse(sent[0].options.body), { conversation_id: ID, action_id: ACTION });
    assert.ok(boxes[0].textContent.includes("has been cancelled."));
    console.log("ok reload and restoration");
  }

  // 16. New conversation forgets stored action IDs.
  {
    const { byId, storage } = await offered(() => respond(500, {}));
    assert.ok(storage.has("support.cancellationActions"));
    byId["new-conversation"].listeners.click();
    assert.ok(!storage.has("support.cancellationActions"));
    console.log("ok new conversation clears offers");
  }
  // --- Milestone 4: hand-off status and feedback ---
  const HANDOFF = "I'm not able to answer this confidently from our help articles, so I've recorded your question as needing help from our support team.";
  const feedbackButtons = (byId) => buttons(byId).filter((b) => b.className.split(" ").includes("feedback-button"));
  const fb = (byId, label) => feedbackButtons(byId).find((b) => b.textContent === label);
  const feedbackPosts = (requests) => requests.filter((r) => r.url === "/feedback");
  const answered = (extra = {}) => ({ answer: "A", sources: [], escalate: false, reason: "answered",
    conversation_id: NEW_ID, message_index: 1, ...extra });
  async function chatted(reply, feedbackImpl = () => respond(200, {})) {
    const page = setup({
      stored: null,
      fetchImpl: (url, o) => (url === "/feedback" ? feedbackImpl(JSON.parse(o.body)) : respond(200, reply)),
    });
    page.byId.question.value = "q";
    await page.byId["chat-form"].requestSubmit();
    return page;
  }
  const saved = (body) => respond(200, { message_index: body.message_index, rating: body.rating, changed: true });

  // 17. An escalated reply: truthful label and status, never "assigned" or "contacted".
  {
    const { byId } = await chatted(answered({ answer: HANDOFF, escalate: true, reason: "insufficient_evidence" }));
    const status = byId["handoff-status"];
    assert.strictEqual(status.hidden, false);
    assert.ok(status.textContent.includes("recorded for them"));
    assert.ok(status.textContent.includes("No one has been assigned"));
    const page = byId.messages.textContent + status.textContent;
    assert.ok(texts(byId).at(-1).includes("Needs human help"));
    for (const claim of ["will follow up", "agent has been assigned", "assigned to you", "will contact", "has contacted",
      "Handed off to a human agent"]) {
      assert.ok(!page.includes(claim), "no false claim: " + claim);
    }
    assert.strictEqual(feedbackButtons(byId).length, 2, "hand-off replies can be rated too");
    console.log("ok hand-off status");
  }

  // 17b. An ordinary answer shows no hand-off status.
  {
    const { byId } = await chatted(answered());
    assert.strictEqual(byId["handoff-status"].hidden, true);
    console.log("ok no hand-off status for answers");
  }

  // 18. Feedback: one request per new rating; the saved rating is not resent.
  {
    const { byId, requests } = await chatted(answered(), saved);
    assert.deepStrictEqual(feedbackButtons(byId).map((b) => b.textContent), ["Yes", "No"]);
    await fb(byId, "Yes").listeners.click();
    assert.deepStrictEqual(JSON.parse(feedbackPosts(requests)[0].options.body),
      { conversation_id: NEW_ID, message_index: 1, rating: "positive" });
    assert.strictEqual(fb(byId, "Yes").attributes["aria-pressed"], "true");
    assert.strictEqual(fb(byId, "No").attributes["aria-pressed"], "false");
    assert.ok(texts(byId).at(-1).includes("Thanks for your feedback."));
    await fb(byId, "Yes").listeners.click();
    assert.strictEqual(feedbackPosts(requests).length, 1, "same rating: no request");
    await fb(byId, "No").listeners.click();
    assert.strictEqual(feedbackPosts(requests).length, 2);
    assert.strictEqual(JSON.parse(feedbackPosts(requests)[1].options.body).rating, "negative");
    assert.strictEqual(fb(byId, "No").attributes["aria-pressed"], "true");
    // Only the rating and the reply's position are sent: nothing personal.
    assert.deepStrictEqual(Object.keys(JSON.parse(feedbackPosts(requests)[1].options.body)).sort(),
      ["conversation_id", "message_index", "rating"]);
    console.log("ok feedback submission");
  }

  // 19. Duplicate clicks while saving send one request.
  {
    let release;
    const { byId, requests } = await chatted(answered(), () => new Promise((r) => { release = r; }));
    const first = fb(byId, "Yes").listeners.click();
    fb(byId, "Yes").listeners.click();
    fb(byId, "No").listeners.click();
    assert.strictEqual(feedbackPosts(requests).length, 1);
    assert.ok(fb(byId, "Yes").disabled && fb(byId, "No").disabled);
    release(respond(200, { message_index: 1, rating: "positive", changed: true }));
    await first;
    assert.strictEqual(fb(byId, "Yes").disabled, false);
    assert.strictEqual(feedbackPosts(requests).length, 1);
    console.log("ok feedback duplicate-click prevention");
  }

  // 20. Failures are reported and can be retried; nothing is shown as saved.
  {
    const outcomes = [() => respond(503, {}), () => { throw new TypeError("offline"); },
      () => respond(404, { detail: "There is no assistant reply to rate at that position in this conversation." }),
      saved];
    const { byId, requests } = await chatted(answered(), (body) => outcomes.shift()(body));
    for (const expected of ["We couldn't save your feedback.", "We couldn't save your feedback.", "no assistant reply to rate"]) {
      await fb(byId, "No").listeners.click();
      assert.ok(texts(byId).at(-1).includes(expected));
      assert.strictEqual(fb(byId, "No").attributes["aria-pressed"], "false");
      assert.strictEqual(fb(byId, "No").disabled, false);
    }
    await fb(byId, "No").listeners.click();
    assert.strictEqual(fb(byId, "No").attributes["aria-pressed"], "true");
    assert.strictEqual(feedbackPosts(requests).length, 4);
    console.log("ok feedback failures");
  }

  // 21. Replies without a server position (malformed) get no feedback controls.
  {
    const { byId } = await chatted(answered({ message_index: "1" }));
    assert.strictEqual(feedbackButtons(byId).length, 0);
    console.log("ok no feedback without a message index");
  }

  // 22. Reload restores ratings and the hand-off status.
  {
    const conversation = (handoff) => respond(200, { conversation_id: ID, message_limit: 40, handoff, messages: [
      { role: "customer", content: "q1", message_index: 0 },
      { role: "assistant", content: "A1", sources: [], escalate: false, reason: "answered", message_index: 1, feedback: "negative" },
      { role: "customer", content: "q2", message_index: 2 },
      { role: "assistant", content: HANDOFF, sources: [], escalate: true, reason: "llm_error", message_index: 3, feedback: null },
    ] });
    const open = setup({ stored: ID, fetchImpl: () => conversation({ status: "open", category: "service_problem",
      created_at: EXPIRES, updated_at: EXPIRES }) });
    await settle();
    assert.strictEqual(open.byId["handoff-status"].hidden, false);
    const rows = feedbackButtons(open.byId);
    assert.strictEqual(rows.length, 4);
    assert.deepStrictEqual(rows.map((b) => b.attributes["aria-pressed"]), ["false", "true", "false", "false"]);
    const none = setup({ stored: ID, fetchImpl: () => conversation(null) });
    await settle();
    assert.strictEqual(none.byId["handoff-status"].hidden, true);
    console.log("ok restore ratings and hand-off status");
  }

  // 23. New conversation clears the hand-off status.
  {
    const { byId } = await chatted(answered({ answer: HANDOFF, escalate: true, reason: "llm_error" }));
    byId["new-conversation"].listeners.click();
    assert.strictEqual(byId["handoff-status"].hidden, true);
    console.log("ok new conversation clears hand-off status");
  }
  // 24. Milestone 5: a 500 the server did NOT save (no conversation ID or
  //     position) is an error; the question stays and nothing claims a record.
  {
    const UNSAVED = "Something went wrong on our side, so your message was not saved. Please try again in a moment.";
    const { byId, storage } = setup({
      stored: null,
      fetchImpl: () => respond(500, { answer: UNSAVED, sources: [], escalate: true, reason: "server_error" }),
    });
    byId.question.value = "my question";
    await byId["chat-form"].requestSubmit();
    assert.ok(texts(byId).at(-1).includes(UNSAVED));
    assert.ok(!texts(byId).at(-1).includes("Needs human help"));
    assert.strictEqual(byId["handoff-status"].hidden, true, "no 'recorded' status for an unsaved reply");
    assert.strictEqual(byId.question.value, "my question", "the question is kept to send again");
    assert.ok(!storage.has("support.conversationId"));
    assert.strictEqual(feedbackButtons(byId).length, 0);
    console.log("ok unsaved 500 is an error");
  }

  // 25. A 500 the server DID save (the agent crashed inside /chat): a recorded hand-off.
  {
    const { byId, storage } = setup({
      stored: null,
      fetchImpl: () => respond(500, answered({ answer: HANDOFF, escalate: true, reason: "server_error" })),
    });
    byId.question.value = "q";
    await byId["chat-form"].requestSubmit();
    assert.ok(texts(byId).at(-1).includes("Needs human help"));
    assert.strictEqual(byId["handoff-status"].hidden, false);
    assert.strictEqual(byId.question.value, "");
    assert.strictEqual(storage.get("support.conversationId"), NEW_ID);
    console.log("ok saved 500 is a recorded hand-off");
  }
  console.log("ALL FRONTEND SMOKE CHECKS PASSED");
})().catch((error) => { console.error(error); process.exit(1); });
