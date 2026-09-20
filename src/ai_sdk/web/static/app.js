const state = {
  mode: "single",
  adaptiveStrategy: "auto",
  conversationId: null,
  conversations: [],
  deleteConversationId: null,
  activeRunId: null,
  running: false,
  citations: [],
  autoScroll: true,
};

const elements = {
  activityList: document.querySelector("#activity-list"),
  adaptiveStrategyControl: document.querySelector("#adaptive-strategy-control"),
  adaptiveStrategyCopy: document.querySelector("#adaptive-strategy-copy"),
  cancelButton: document.querySelector("#cancel-button"),
  chatForm: document.querySelector("#chat-form"),
  composerState: document.querySelector("#composer-state"),
  conversationList: document.querySelector("#conversation-list"),
  conversationNavigation: document.querySelector("#conversation-navigation"),
  conversationTitle: document.querySelector("#conversation-title"),
  deleteDialog: document.querySelector("#delete-conversation-dialog"),
  deleteDialogCancel: document.querySelector("#delete-dialog-cancel"),
  deleteDialogConfirm: document.querySelector("#delete-dialog-confirm"),
  deleteDialogCopy: document.querySelector("#delete-dialog-copy"),
  documentCount: document.querySelector("#document-count"),
  documentList: document.querySelector("#document-list"),
  emptyState: document.querySelector("#empty-state"),
  fileInput: document.querySelector("#file-input"),
  messageInput: document.querySelector("#message-input"),
  messages: document.querySelector("#messages"),
  modeCopy: document.querySelector("#mode-copy"),
  provider: document.querySelector("#provider"),
  chatListToggle: document.querySelector("#chat-list-toggle"),
  headerNewChatButton: document.querySelector("#header-new-chat-button"),
  newChatButton: document.querySelector("#new-chat-button"),
  routeSummary: document.querySelector("#route-summary"),
  sendButton: document.querySelector("#send-button"),
  sourceList: document.querySelector("#source-list"),
  systemStatus: document.querySelector("#system-status"),
  toast: document.querySelector("#toast"),
  uploadMessage: document.querySelector("#upload-message"),
  uploadZone: document.querySelector("#upload-zone"),
};

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

document.querySelectorAll(".strategy-option").forEach((button) => {
  button.addEventListener("click", () => setAdaptiveStrategy(button.dataset.strategy));
});

document.querySelectorAll(".prompt-card").forEach((button) => {
  button.addEventListener("click", () => {
    elements.messageInput.value = button.textContent.trim();
    elements.messageInput.focus();
    resizeComposer();
  });
});

elements.fileInput.addEventListener("change", () => {
  if (elements.fileInput.files[0]) uploadFile(elements.fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  elements.uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadZone.classList.remove("dragging");
  });
});

elements.uploadZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
});

elements.messageInput.addEventListener("input", resizeComposer);
elements.messages.addEventListener("scroll", () => {
  const distanceFromBottom = elements.messages.scrollHeight
    - elements.messages.scrollTop
    - elements.messages.clientHeight;
  state.autoScroll = distanceFromBottom < 96;
});
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});
elements.chatForm.addEventListener("submit", sendMessage);
elements.cancelButton.addEventListener("click", cancelRun);
elements.newChatButton.addEventListener("click", startNewChat);
elements.headerNewChatButton.addEventListener("click", startNewChat);
elements.chatListToggle.addEventListener("click", toggleConversationList);
elements.deleteDialogCancel.addEventListener("click", closeDeleteDialog);
elements.deleteDialogConfirm.addEventListener("click", confirmDeleteConversation);
elements.deleteDialog.addEventListener("click", (event) => {
  if (event.target === elements.deleteDialog) closeDeleteDialog();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.deleteDialog.classList.contains("hidden")) {
    closeDeleteDialog();
  }
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".conversation-item")) closeConversationMenus();
});

const messageObserver = new MutationObserver(() => {
  if (state.running && state.autoScroll) scheduleMessageScroll();
});
messageObserver.observe(elements.messages, {
  childList: true,
  characterData: true,
  subtree: true,
});

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

async function refreshConversations({ selectLatest = false } = {}) {
  try {
    state.conversations = await request("/api/conversations");
    if (selectLatest && !state.conversationId && state.conversations.length) {
      await loadConversation(state.conversations[0].conversation_id);
      return;
    }
    if (state.conversationId) {
      const active = state.conversations.find(
        (conversation) => conversation.conversation_id === state.conversationId
      );
      if (active) {
        elements.conversationTitle.textContent = active.title;
      } else {
        state.conversationId = null;
        elements.conversationTitle.textContent = "New conversation";
        renderConversationList();
        return;
      }
    }
    renderConversationList();
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderConversationList() {
  elements.conversationList.replaceChildren();
  if (!state.conversations.length) {
    const empty = document.createElement("div");
    empty.className = "conversation-list-empty";
    empty.textContent = "No saved chats yet.";
    elements.conversationList.append(empty);
    return;
  }

  state.conversations.forEach((conversation) => {
    const item = document.createElement("div");
    item.className = "conversation-item";
    item.classList.toggle(
      "active",
      conversation.conversation_id === state.conversationId
    );
    item.dataset.conversationId = conversation.conversation_id;

    const select = document.createElement("button");
    select.className = "conversation-select";
    select.type = "button";
    select.disabled = state.running;
    select.setAttribute("aria-label", `Open ${conversation.title}`);
    const title = document.createElement("strong");
    title.textContent = conversation.title;
    const detail = document.createElement("small");
    const turns = Math.floor(conversation.message_count / 2);
    const model = conversation.last_mode === "adaptive"
      ? `Adaptive · ${conversation.last_adaptive_strategy}`
      : (conversation.last_provider || "Single");
    detail.textContent = `${turns} ${turns === 1 ? "turn" : "turns"} · ${model}`;
    select.append(title, detail);
    select.addEventListener("click", () => loadConversation(conversation.conversation_id));

    const menuToggle = document.createElement("button");
    menuToggle.className = "conversation-menu-toggle";
    menuToggle.type = "button";
    menuToggle.disabled = state.running;
    menuToggle.setAttribute("aria-label", `Manage ${conversation.title}`);
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.textContent = "•••";

    const menu = document.createElement("div");
    menu.className = "conversation-menu";
    const rename = document.createElement("button");
    rename.type = "button";
    rename.textContent = "Rename";
    rename.addEventListener("click", () => beginRename(item, conversation));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => openDeleteDialog(conversation));
    menu.append(rename, remove);
    menuToggle.addEventListener("click", (event) => {
      event.stopPropagation();
      const willOpen = !menu.classList.contains("visible");
      closeConversationMenus();
      menu.classList.toggle("visible", willOpen);
      menuToggle.setAttribute("aria-expanded", String(willOpen));
    });

    item.append(select, menuToggle, menu);
    elements.conversationList.append(item);
  });
}

function closeConversationMenus() {
  document.querySelectorAll(".conversation-menu.visible").forEach((menu) => {
    menu.classList.remove("visible");
    menu.parentElement.querySelector(".conversation-menu-toggle")
      ?.setAttribute("aria-expanded", "false");
  });
}

async function loadConversation(conversationId) {
  if (state.running || conversationId === state.conversationId) return;
  try {
    const conversation = await request(
      `/api/conversations/${encodeURIComponent(conversationId)}`
    );
    state.conversationId = conversation.conversation_id;
    elements.conversationTitle.textContent = conversation.title;
    elements.messages.replaceChildren();
    conversation.messages.forEach((message) => {
      const bubble = appendMessage(message.role, message.content);
      if (message.role === "assistant") {
        bubble.className = "bubble markdown-body";
        renderMarkdown(bubble, message.content);
      }
    });
    if (!conversation.messages.length) elements.emptyState.classList.remove("hidden");
    if (conversation.last_provider) elements.provider.value = conversation.last_provider;
    setAdaptiveStrategy(conversation.last_adaptive_strategy);
    setMode(conversation.last_mode);
    clearEvidence();
    renderConversationList();
    state.autoScroll = true;
    scheduleMessageScroll(true);
  } catch (error) {
    showToast(error.message, true);
    await refreshConversations();
  }
}

function startNewChat() {
  if (state.running) return;
  state.conversationId = null;
  elements.conversationTitle.textContent = "New conversation";
  elements.messages.replaceChildren();
  elements.emptyState.classList.remove("hidden");
  clearEvidence();
  renderConversationList();
  elements.messageInput.focus();
}

function toggleConversationList() {
  const collapsed = elements.conversationNavigation.classList.toggle("collapsed");
  elements.chatListToggle.setAttribute("aria-expanded", String(!collapsed));
}

function beginRename(item, conversation) {
  if (state.running) return;
  closeConversationMenus();
  const input = document.createElement("input");
  input.className = "conversation-rename-input";
  input.type = "text";
  input.maxLength = 80;
  input.value = conversation.title;
  input.setAttribute("aria-label", "Conversation title");
  item.append(input);
  input.focus();
  input.select();

  let finished = false;
  const cancel = () => {
    if (finished) return;
    finished = true;
    input.remove();
  };
  const save = async () => {
    if (finished) return;
    const title = input.value.trim();
    if (!title) {
      showToast("Conversation title cannot be blank.", true);
      input.focus();
      return;
    }
    finished = true;
    try {
      await request(`/api/conversations/${encodeURIComponent(conversation.conversation_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      await refreshConversations();
      showToast("Conversation renamed.");
    } catch (error) {
      showToast(error.message, true);
      renderConversationList();
    }
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      save();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      cancel();
    }
  });
  input.addEventListener("blur", cancel);
}

function openDeleteDialog(conversation) {
  if (state.running) return;
  closeConversationMenus();
  state.deleteConversationId = conversation.conversation_id;
  elements.deleteDialogCopy.textContent = `“${conversation.title}” and its messages will be permanently removed. Documents and long-term memory will stay available.`;
  elements.deleteDialog.classList.remove("hidden");
  elements.deleteDialogConfirm.focus();
}

function closeDeleteDialog() {
  state.deleteConversationId = null;
  elements.deleteDialog.classList.add("hidden");
}

async function confirmDeleteConversation() {
  const conversationId = state.deleteConversationId;
  if (!conversationId || state.running) return;
  elements.deleteDialogConfirm.disabled = true;
  try {
    await request(`/api/conversations/${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
    });
    const wasActive = conversationId === state.conversationId;
    closeDeleteDialog();
    if (wasActive) startNewChat();
    await refreshConversations();
    showToast("Conversation deleted.");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.deleteDialogConfirm.disabled = false;
  }
}

async function refreshStatus() {
  try {
    const status = await request("/api/status");
    const readyProviders = status.providers.filter((provider) => provider.ready);
    elements.systemStatus.className = `status-pill ${readyProviders.length ? "ready" : "warning"}`;
    elements.systemStatus.innerHTML = "";
    const dot = document.createElement("span");
    dot.className = "status-dot";
    elements.systemStatus.append(dot, document.createTextNode(
      readyProviders.length ? `${readyProviders.length}/3 providers ready` : "Add provider keys"
    ));

    status.providers.forEach((provider) => {
      const option = elements.provider.querySelector(`option[value="${provider.provider}"]`);
      if (option) {
        option.disabled = !provider.ready;
        option.textContent = `${provider.display_name}${provider.ready ? " · Ready" : " · Not configured"}`;
      }
    });
    if (elements.provider.selectedOptions[0]?.disabled && readyProviders[0]) {
      elements.provider.value = readyProviders[0].provider;
    }
    const metrics = status.adaptive_metrics;
    document.querySelector("#metric-runs").textContent = metrics.total_runs;
    document.querySelector("#metric-success").textContent = metrics.total_runs
      ? `${Math.round((metrics.successful_runs / metrics.total_runs) * 100)}%`
      : "—";
  } catch (error) {
    elements.systemStatus.className = "status-pill warning";
    elements.systemStatus.textContent = "Server unavailable";
  }
}

async function refreshDocuments() {
  try {
    const documents = await request("/api/documents");
    elements.documentCount.textContent = documents.length;
    elements.documentList.replaceChildren();
    documents.forEach((documentItem) => {
      const item = document.createElement("div");
      item.className = "document-item";

      const type = document.createElement("span");
      type.className = "document-type";
      type.textContent = documentItem.format;

      const copy = document.createElement("div");
      copy.className = "document-copy";
      const name = document.createElement("strong");
      name.textContent = documentItem.source;
      const metadata = document.createElement("small");
      metadata.textContent = documentItem.page_count
        ? `${documentItem.page_count} pages · ${documentItem.chunk_count} chunks`
        : `${documentItem.chunk_count} chunks`;
      copy.append(name, metadata);

      const remove = document.createElement("button");
      remove.className = "document-remove";
      remove.type = "button";
      remove.title = "Remove document";
      remove.textContent = "×";
      remove.addEventListener("click", () => removeDocument(documentItem.document_id));
      item.append(type, copy, remove);
      elements.documentList.append(item);
    });
  } catch (error) {
    showToast(error.message, true);
  }
}

async function uploadFile(file) {
  elements.uploadMessage.className = "inline-message";
  elements.uploadMessage.textContent = `Indexing ${file.name}…`;
  const body = new FormData();
  body.append("file", file);
  try {
    await request("/api/documents", { method: "POST", body });
    elements.uploadMessage.textContent = `${file.name} is ready for retrieval.`;
    await refreshDocuments();
    showToast(`${file.name} indexed and ready for retrieval.`);
  } catch (error) {
    elements.uploadMessage.className = "inline-message error";
    elements.uploadMessage.textContent = error.message;
  } finally {
    elements.fileInput.value = "";
  }
}

async function removeDocument(documentId) {
  try {
    await request(`/api/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
    await refreshDocuments();
    showToast("Document removed.");
  } catch (error) {
    showToast(error.message, true);
  }
}

function setMode(mode) {
  if (state.running) return;
  state.mode = mode;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  const adaptive = mode === "adaptive";
  elements.provider.disabled = adaptive;
  elements.adaptiveStrategyControl.classList.toggle("hidden", !adaptive);
  elements.modeCopy.textContent = adaptive
    ? "Choose automatic routing or force the complete three-provider workflow."
    : "One provider answers with the shared RAG context.";
}

function setAdaptiveStrategy(strategy) {
  if (state.running || !["auto", "maximum"].includes(strategy)) return;
  state.adaptiveStrategy = strategy;
  document.querySelectorAll(".strategy-option").forEach((button) => {
    button.classList.toggle("active", button.dataset.strategy === strategy);
  });
  elements.adaptiveStrategyCopy.textContent = strategy === "maximum"
    ? "Always runs Gemini → Claude → OpenAI. Uses more time and provider requests."
    : "Chooses the smallest suitable route for lower cost and latency.";
}

async function sendMessage(event) {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message || state.running) return;

  if (!state.conversationId && elements.messages.children.length) {
    elements.messages.replaceChildren();
    elements.emptyState.classList.remove("hidden");
  }
  state.autoScroll = true;
  appendMessage("user", message);
  const answerBubble = appendMessage("assistant", "", true);
  elements.messageInput.value = "";
  resizeComposer();
  resetEvidence();
  setRunning(true);

  const payload = { message, mode: state.mode };
  if (state.mode === "single") payload.provider = elements.provider.value;
  if (state.mode === "adaptive") payload.adaptive_strategy = state.adaptiveStrategy;
  if (state.conversationId) payload.conversation_id = state.conversationId;

  let terminalEventReceived = false;
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || `Request failed (${response.status}).`);
    }
    state.activeRunId = response.headers.get("X-Run-ID");
    state.conversationId = response.headers.get("X-Conversation-ID") || state.conversationId;
    renderConversationList();
    await consumeEventStream(response.body, (type, data) => {
      if (["answer", "cancelled", "error"].includes(type)) {
        terminalEventReceived = true;
      }
      handleStreamEvent(type, data, answerBubble);
    });
    if (!terminalEventReceived) {
      clearPendingState(answerBubble);
      answerBubble.className = "bubble error";
      answerBubble.textContent = "The response stream ended before a final answer arrived.";
      scheduleMessageScroll(true);
    }
  } catch (error) {
    clearPendingState(answerBubble);
    answerBubble.className = "bubble error";
    answerBubble.textContent = error.message;
    scheduleMessageScroll(true);
  } finally {
    setRunning(false);
    state.activeRunId = null;
    await Promise.all([refreshStatus(), refreshConversations()]);
  }
}

async function consumeEventStream(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    blocks.forEach((block) => {
      let type = "message";
      let data = null;
      block.split("\n").forEach((line) => {
        if (line.startsWith("event: ")) type = line.slice(7);
        if (line.startsWith("data: ")) data = JSON.parse(line.slice(6));
      });
      if (data !== null) onEvent(type, data);
    });
    if (done) break;
  }
}

function handleStreamEvent(type, data, answerBubble) {
  if (type === "run") {
    state.activeRunId = data.run_id;
    state.conversationId = data.conversation_id || state.conversationId;
  }
  if (type === "route") {
    const routeName = data.route.replaceAll("_", " ");
    elements.routeSummary.className = "route-summary active";
    elements.routeSummary.querySelector("strong").textContent = routeName;
    elements.composerState.textContent = `${routeName} route selected`;
  }
  if (type === "stage") addActivity(data);
  if (type === "citations") {
    state.citations = data;
    renderSources(data);
  }
  if (type === "answer") {
    clearPendingState(answerBubble);
    answerBubble.className = "bubble markdown-body";
    renderMarkdown(answerBubble, data.content);
    scheduleMessageScroll(true);
  }
  if (type === "cancelled") {
    clearPendingState(answerBubble);
    answerBubble.className = "bubble error";
    answerBubble.textContent = "The workflow was cancelled at a safe boundary.";
    scheduleMessageScroll(true);
  }
  if (type === "error") {
    clearPendingState(answerBubble);
    answerBubble.className = "bubble error";
    answerBubble.textContent = data.message;
    scheduleMessageScroll(true);
  }
}

function addActivity(data) {
  if (elements.activityList.querySelector(".activity-placeholder")) {
    elements.activityList.replaceChildren();
  }
  const item = document.createElement("div");
  const shortStatus = data.status.replace("stage_", "").replace("workflow_", "");
  item.className = `activity-item ${shortStatus}`;
  const node = document.createElement("span");
  node.className = "activity-node";
  const label = document.createElement("strong");
  label.textContent = data.stage_id || "workflow";
  const status = document.createElement("small");
  status.textContent = shortStatus;
  item.append(node, label, status);
  elements.activityList.append(item);
}

function renderSources(citations) {
  elements.sourceList.replaceChildren();
  if (!citations.length) {
    const empty = document.createElement("div");
    empty.className = "source-placeholder";
    empty.textContent = "No indexed source was retrieved for this answer.";
    elements.sourceList.append(empty);
    return;
  }
  citations.forEach((citation) => {
    const card = document.createElement("article");
    card.className = "source-card";
    const header = document.createElement("header");
    const name = document.createElement("strong");
    name.textContent = citation.source;
    const position = document.createElement("span");
    position.textContent = `[${citation.position}]`;
    header.append(name, position);
    const detail = document.createElement("small");
    const page = citation.page ? `Page ${citation.page} · ` : "";
    detail.textContent = `${page}relevance ${citation.score.toFixed(3)}`;
    card.append(header, detail);
    elements.sourceList.append(card);
  });
}

function resetEvidence() {
  elements.routeSummary.className = "route-summary neutral";
  elements.routeSummary.querySelector("strong").textContent = state.mode === "adaptive"
    ? (state.adaptiveStrategy === "maximum" ? "Full route · maximum" : "Selecting route")
    : `${elements.provider.value} · single`;
  elements.activityList.innerHTML = '<div class="activity-placeholder">Waiting for execution progress…</div>';
  elements.sourceList.innerHTML = '<div class="source-placeholder">Retrieving relevant context…</div>';
}

function clearEvidence() {
  state.citations = [];
  elements.routeSummary.className = "route-summary neutral";
  elements.routeSummary.querySelector("strong").textContent = "Waiting for a request";
  elements.activityList.innerHTML = '<div class="activity-placeholder">Stage progress appears here without exposing prompts or handoff content.</div>';
  elements.sourceList.innerHTML = '<div class="source-placeholder">Citations will be attached to the next grounded answer.</div>';
}

function appendMessage(role, content, pending = false) {
  elements.emptyState.classList.add("hidden");
  const row = document.createElement("div");
  row.className = `message ${role}${pending ? " pending-response" : ""}`;
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  if (role === "user") {
    avatar.textContent = "YOU";
  } else {
    avatar.classList.add("assistant-avatar");
    avatar.append(createConvergenceCoreIcon());
  }
  const bubble = document.createElement("div");
  bubble.className = `bubble${pending ? " pending" : ""}`;
  if (pending) {
    bubble.setAttribute("aria-label", "Generating response");
    bubble.append(createThinkingIndicator());
  } else {
    bubble.textContent = content;
  }
  row.append(avatar, bubble);
  elements.messages.append(row);
  scheduleMessageScroll(true);
  return bubble;
}

function createThinkingIndicator() {
  const indicator = document.createElement("span");
  indicator.className = "thinking-indicator";
  indicator.setAttribute("aria-hidden", "true");
  indicator.append(createConvergenceCoreIcon({ waiting: true }));
  return indicator;
}

function createConvergenceCoreIcon({ waiting = false } = {}) {
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.classList.add("convergence-icon");
  svg.classList.add(waiting ? "waiting-convergence" : "avatar-convergence");
  svg.setAttribute("viewBox", "0 0 256 256");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");

  const segments = document.createElementNS(namespace, "g");
  segments.setAttribute("class", "convergence-segments");
  [
    "m150 47 54 31v61",
    "m190 184-62 36-62-36",
    "M52 139V78l54-31",
  ].forEach((definition) => {
    const segment = document.createElementNS(namespace, "path");
    segment.setAttribute("d", definition);
    segments.append(segment);
  });
  svg.append(segments);

  if (!waiting) {
    const core = document.createElementNS(namespace, "path");
    core.setAttribute("class", "convergence-core");
    core.setAttribute("d", "m128 108 20 20-20 20-20-20z");
    svg.append(core);
    return svg;
  }

  const core = document.createElementNS(namespace, "path");
  core.setAttribute("class", "convergence-core wandering-core");
  core.setAttribute("d", "m128 108 20 20-20 20-20-20z");
  svg.append(core);
  startWanderingCore(core);
  return svg;
}

const wanderingCoreTimers = new WeakMap();

function startWanderingCore(core) {
  const motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (motionPreference.matches) return;

  const zones = [
    { x: -16, y: -10 },
    { x: 16, y: -10 },
    { x: 0, y: 17 },
    { x: 0, y: 0 },
  ];
  let zoneIndex = 3;

  const schedule = (delay) => {
    const timer = window.setTimeout(() => {
      wanderingCoreTimers.delete(core);
      move();
    }, delay);
    wanderingCoreTimers.set(core, timer);
  };

  const move = () => {
    if (!core.isConnected) return;
    if (motionPreference.matches) {
      core.style.transition = "none";
      core.style.transform = "none";
      return;
    }

    const microSaccade = zoneIndex !== 3 && Math.random() < 0.34;
    let nextZone = zoneIndex;
    if (!microSaccade) {
      if (zoneIndex !== 3 && Math.random() < 0.16) {
        nextZone = 3;
      } else {
        const candidates = [0, 1, 2].filter((index) => index !== zoneIndex);
        nextZone = candidates[Math.floor(Math.random() * candidates.length)];
      }
    }

    const target = zones[nextZone];
    const jitter = microSaccade ? 1.2 : 1.8;
    const x = target.x + (Math.random() - 0.5) * jitter * 2;
    const y = target.y + (Math.random() - 0.5) * jitter * 2;
    const scale = 0.99 + Math.random() * 0.02;
    const duration = microSaccade
      ? 70 + Math.random() * 55
      : 105 + Math.random() * 85;
    let pause = microSaccade
      ? 260 + Math.random() * 420
      : 620 + Math.random() * 980;
    if (!microSaccade && Math.random() < 0.14) {
      pause = 1800 + Math.random() * 800;
    }

    core.style.transition = `transform ${Math.round(duration)}ms cubic-bezier(0.16, 1, 0.3, 1)`;
    core.style.transform = `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px) scale(${scale.toFixed(3)})`;
    zoneIndex = nextZone;
    schedule(duration + pause);
  };

  schedule(240 + Math.random() * 420);
}

function stopWanderingCore(bubble) {
  const core = bubble.querySelector(".wandering-core");
  const timer = core ? wanderingCoreTimers.get(core) : undefined;
  if (timer !== undefined) window.clearTimeout(timer);
  if (core) wanderingCoreTimers.delete(core);
}

function clearPendingState(bubble) {
  stopWanderingCore(bubble);
  bubble.removeAttribute("aria-label");
  bubble.closest(".message")?.classList.remove("pending-response");
}

let messageScrollFrame = null;
function scheduleMessageScroll(force = false) {
  if (!force && !state.autoScroll) return;
  if (messageScrollFrame !== null) cancelAnimationFrame(messageScrollFrame);
  messageScrollFrame = requestAnimationFrame(() => {
    messageScrollFrame = requestAnimationFrame(() => {
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      elements.messages.scrollTo({
        top: elements.messages.scrollHeight,
        behavior: reducedMotion ? "auto" : "smooth",
      });
      messageScrollFrame = null;
    });
  });
}

function renderMarkdown(container, markdown) {
  container.replaceChildren();
  const lines = String(markdown || "").replaceAll("\r\n", "\n").split("\n");
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    const fence = line.match(/^\s*```([\w+#.-]*)\s*$/);
    if (fence) {
      const code = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      container.append(createCodeBlock(code.join("\n"), fence[1]));
      continue;
    }

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const element = document.createElement(`h${heading[1].length}`);
      appendInlineMarkdown(element, heading[2]);
      container.append(element);
      index += 1;
      continue;
    }

    const listItem = line.match(/^\s*([-*+] |\d+[.)] )(.*)$/);
    if (listItem) {
      const ordered = /^\d/.test(listItem[1]);
      const list = document.createElement(ordered ? "ol" : "ul");
      while (index < lines.length) {
        const item = lines[index].match(/^\s*([-*+] |\d+[.)] )(.*)$/);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        const entry = document.createElement("li");
        appendInlineMarkdown(entry, item[2]);
        list.append(entry);
        index += 1;
      }
      container.append(list);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quote = document.createElement("blockquote");
      const quoteLines = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      appendInlineMarkdown(quote, quoteLines.join(" "));
      container.append(quote);
      continue;
    }

    if (/^\s*(?:---+|___+|\*\*\*+)\s*$/.test(line)) {
      container.append(document.createElement("hr"));
      index += 1;
      continue;
    }

    const paragraphLines = [line.trim()];
    index += 1;
    while (
      index < lines.length
      && lines[index].trim()
      && !isMarkdownBlockStart(lines[index])
    ) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, paragraphLines.join(" "));
    container.append(paragraph);
  }
}

function isMarkdownBlockStart(line) {
  return /^\s*```/.test(line)
    || /^(#{1,4})\s+/.test(line)
    || /^\s*([-*+] |\d+[.)] )/.test(line)
    || /^\s*>\s?/.test(line)
    || /^\s*(?:---+|___+|\*\*\*+)\s*$/.test(line);
}

function appendInlineMarkdown(parent, text) {
  const tokenPattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_([^_\n]+)_|\[[^\]\n]+\]\([^\s)]+\))/g;
  let cursor = 0;

  for (const match of text.matchAll(tokenPattern)) {
    parent.append(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    } else if (token.startsWith("**") || token.startsWith("__")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else if (token.startsWith("*") || token.startsWith("_")) {
      const emphasis = document.createElement("em");
      emphasis.textContent = token.slice(1, -1);
      parent.append(emphasis);
    } else {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      const safeUrl = linkMatch && safeLinkUrl(linkMatch[2]);
      if (safeUrl) {
        const link = document.createElement("a");
        link.textContent = linkMatch[1];
        link.href = safeUrl;
        link.target = "_blank";
        link.rel = "noreferrer noopener";
        parent.append(link);
      } else {
        parent.append(document.createTextNode(linkMatch ? linkMatch[1] : token));
      }
    }
    cursor = match.index + token.length;
  }
  parent.append(document.createTextNode(text.slice(cursor)));
}

function safeLinkUrl(value) {
  try {
    const url = new URL(value, window.location.origin);
    return ["http:", "https:", "mailto:"].includes(url.protocol) ? url.href : null;
  } catch (_) {
    return null;
  }
}

function createCodeBlock(source, language) {
  const wrapper = document.createElement("div");
  wrapper.className = "code-block";

  const header = document.createElement("div");
  header.className = "code-header";
  const label = document.createElement("span");
  const normalizedLanguage = normalizeLanguage(language);
  label.textContent = normalizedLanguage || "code";

  const copyButton = document.createElement("button");
  copyButton.className = "copy-code";
  copyButton.type = "button";
  copyButton.setAttribute("aria-label", "Copy code");
  copyButton.textContent = "Copy";
  copyButton.addEventListener("click", async () => {
    const copied = await copyText(source);
    copyButton.textContent = copied ? "Copied" : "Copy failed";
    copyButton.classList.toggle("copied", copied);
    window.setTimeout(() => {
      copyButton.textContent = "Copy";
      copyButton.classList.remove("copied");
    }, 1600);
  });
  header.append(label, copyButton);

  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.className = normalizedLanguage ? `language-${normalizedLanguage}` : "";
  appendHighlightedCode(code, source, normalizedLanguage);
  pre.append(code);
  wrapper.append(header, pre);
  return wrapper;
}

function normalizeLanguage(language) {
  const aliases = {
    js: "javascript",
    ts: "typescript",
    py: "python",
    sh: "bash",
    shell: "bash",
    yml: "yaml",
  };
  const normalized = String(language || "").toLowerCase();
  return aliases[normalized] || normalized;
}

function appendHighlightedCode(parent, source, language) {
  const keywords = {
    python: new Set(["and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try", "while", "with", "yield"]),
    javascript: new Set(["async", "await", "break", "case", "catch", "class", "const", "continue", "default", "delete", "do", "else", "export", "extends", "false", "finally", "for", "from", "function", "if", "import", "in", "instanceof", "let", "new", "null", "of", "return", "static", "super", "switch", "this", "throw", "true", "try", "typeof", "undefined", "var", "while", "yield"]),
    typescript: new Set(["any", "as", "async", "await", "boolean", "class", "const", "else", "enum", "export", "extends", "false", "for", "from", "function", "if", "implements", "import", "in", "interface", "keyof", "let", "new", "null", "number", "of", "private", "protected", "public", "readonly", "return", "static", "string", "super", "this", "throw", "true", "try", "type", "typeof", "undefined", "unknown", "void", "while"]),
    bash: new Set(["case", "do", "done", "elif", "else", "esac", "export", "fi", "for", "function", "if", "in", "local", "readonly", "select", "then", "until", "while"]),
    sql: new Set(["and", "as", "asc", "by", "case", "create", "delete", "desc", "distinct", "drop", "else", "end", "from", "group", "having", "in", "insert", "into", "is", "join", "limit", "not", "null", "on", "or", "order", "select", "set", "table", "then", "union", "update", "values", "when", "where"]),
  };
  const languageKeywords = keywords[language] || new Set();
  const commentMarkers = language === "python" || language === "bash" ? ["#"]
    : language === "sql" ? ["--"] : ["//"];
  const tokenPattern = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_$][\w$]*\b|\s+|.)/gs;
  const sourceLines = source.split("\n");

  sourceLines.forEach((line, lineIndex) => {
    const commentAt = findCommentStart(line, commentMarkers);
    const codePart = commentAt >= 0 ? line.slice(0, commentAt) : line;
    const commentPart = commentAt >= 0 ? line.slice(commentAt) : "";

    for (const match of codePart.matchAll(tokenPattern)) {
      const token = match[0];
      let className = "";
      if (/^["']/.test(token)) className = "syntax-string";
      else if (/^\d/.test(token)) className = "syntax-number";
      else if (languageKeywords.has(language === "sql" ? token.toLowerCase() : token)) className = "syntax-keyword";
      appendCodeToken(parent, token, className);
    }
    if (commentPart) appendCodeToken(parent, commentPart, "syntax-comment");
    if (lineIndex < sourceLines.length - 1) parent.append(document.createTextNode("\n"));
  });
}

function findCommentStart(line, markers) {
  let quote = null;
  let escaped = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === quote) quote = null;
      continue;
    }
    if (character === '"' || character === "'") {
      quote = character;
      continue;
    }
    if (markers.some((marker) => line.startsWith(marker, index))) return index;
  }
  return -1;
}

function appendCodeToken(parent, value, className) {
  if (!className) {
    parent.append(document.createTextNode(value));
    return;
  }
  const token = document.createElement("span");
  token.className = className;
  token.textContent = value;
  parent.append(token);
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch (_) {
    const fallback = document.createElement("textarea");
    fallback.value = value;
    fallback.setAttribute("readonly", "");
    fallback.style.position = "fixed";
    fallback.style.opacity = "0";
    document.body.append(fallback);
    fallback.select();
    const copied = document.execCommand("copy");
    fallback.remove();
    return copied;
  }
}

async function cancelRun() {
  if (!state.activeRunId) return;
  try {
    const result = await request(`/api/runs/${state.activeRunId}/cancel`, { method: "POST" });
    elements.composerState.textContent = result.accepted
      ? "Cancellation requested"
      : "This stage cannot be interrupted";
  } catch (error) {
    showToast(error.message, true);
  }
}

function setRunning(running) {
  state.running = running;
  elements.sendButton.disabled = running;
  elements.newChatButton.disabled = running;
  elements.headerNewChatButton.disabled = running;
  elements.provider.disabled = running || state.mode === "adaptive";
  document.querySelectorAll(".strategy-option").forEach((button) => {
    button.disabled = running;
  });
  elements.cancelButton.classList.toggle("hidden", !running || state.mode !== "adaptive");
  elements.composerState.textContent = running ? "Running" : "Ready";
  renderConversationList();
}

function resizeComposer() {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 160)}px`;
}

let toastTimer = null;
function showToast(message, error = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.className = `toast visible${error ? " error" : ""}`;
  toastTimer = setTimeout(() => { elements.toast.className = "toast"; }, 3200);
}

async function initialize() {
  setMode("single");
  await Promise.all([refreshStatus(), refreshDocuments()]);
  await refreshConversations({ selectLatest: true });
}

initialize();
