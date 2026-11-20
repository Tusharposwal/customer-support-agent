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

function setup({ stored, fetchImpl }) {
  const byId = {};
  for (const id of ["chat-form", "question", "send", "new-conversation", "messages", "hint"]) byId[id] = new Node("div");
  const greeting = new Node("div"); greeting.id = "greeting"; greeting.textContent = "Hi!";
  byId.greeting = greeting;
  byId.messages.appendChild(greeting);
  const storage = new Map(stored ? [["support.conversationId", stored]] : []);
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
    assert.ok(texts(byId).some((t) => t.includes("Handed off to a human agent")));
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
  console.log("ALL FRONTEND SMOKE CHECKS PASSED");
})().catch((error) => { console.error(error); process.exit(1); });
