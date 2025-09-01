(() => {
  const API_ROOT = "http://localhost:5057"; // Flask API server
  
  // Check authentication
  const account = localStorage.getItem("aimhsa_account");
  if (!account) {
      window.location.href = '/login';
      return;
  }
  
  // Elements
  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("form");
  const queryInput = document.getElementById("query");
  const sendBtn = document.getElementById("send");
  const fileInput = document.getElementById("file");
  const composer = form; // composer container (used for inserting preview)
  const historyList = document.getElementById("historyList");
  const newChatBtn = document.getElementById("newChatBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");
  const logoutBtn = document.getElementById("logoutBtn");
  const usernameEl = document.getElementById("username");
  
  let convId = localStorage.getItem("aimhsa_conv") || null;
  let typingEl = null;
  let currentPreview = null;
  
  // Set username
  usernameEl.textContent = account === 'null' ? 'Guest' : account;
  
  // Inject runtime CSS for animations & preview (keeps frontend simple)
  (function injectStyles(){
    const css = `
      @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity:1; transform:none; } }
      .fade-in { animation: fadeIn 280ms ease both; }
      .typing { display:flex; align-items:center; gap:8px; padding:8px 12px; border-radius:10px; width:fit-content; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03); }
      .dots { display:inline-block; width:36px; text-align:center; }
      .dot { display:inline-block; width:6px; height:6px; margin:0 2px; background:var(--muted); border-radius:50%; opacity:0.25; animation: blink 1s infinite; }
      .dot:nth-child(2){ animation-delay: .2s; } .dot:nth-child(3){ animation-delay: .4s; }
      @keyframes blink { 0%{opacity:.25} 50%{opacity:1} 100%{opacity:.25} }
      .upload-preview { display:flex; align-items:center; gap:12px; padding:8px 10px; border-radius:8px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03); margin-right:auto; max-width:420px; }
      .upload-meta { display:flex; flex-direction:column; gap:4px; font-size:13px; color:var(--muted); }
      .upload-filename { font-weight:600; color:var(--text); }
      .upload-actions { display:flex; gap:8px; align-items:center; }
      .progress-bar { width:160px; height:8px; background:rgba(255,255,255,0.03); border-radius:6px; overflow:hidden; }
      .progress-inner { height:100%; width:0%; background:linear-gradient(90deg,var(--accent), #5b21b6); transition:width .2s ease; }
      .btn-small { padding:6px 8px; border-radius:8px; background:transparent; border:1px solid rgba(255,255,255,0.04); color:var(--muted); cursor:pointer; font-size:12px; }
      .sending { opacity:0.7; transform:scale(.98); transition:transform .12s ease, opacity .12s ease; }
      .msg.fade-in { transform-origin: left top; }
    `;
    const s = document.createElement("style");
    s.textContent = css;
    document.head.appendChild(s);
  })();

  // helper: ensure messages container scrolls to bottom after layout updates
  function ensureScroll() {
    const doScroll = () => {
      try {
        const last = messagesEl.lastElementChild;
        if (last && typeof last.scrollIntoView === "function") {
          last.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" });
        } else {
          messagesEl.scrollTop = messagesEl.scrollHeight;
        }
      } catch (e) {
        try { messagesEl.scrollTop = messagesEl.scrollHeight; } catch (_) {}
      }
    };
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setTimeout(doScroll, 40);
      });
    });
  }

  // Logout handler
  logoutBtn.addEventListener("click", () => {
      localStorage.removeItem("aimhsa_account");
      localStorage.removeItem("aimhsa_conv");
      window.location.href = '/login';
  });
  
  // Modern message display
  function appendMessage(role, text) {
      const msgDiv = document.createElement("div");
      msgDiv.className = `msg ${role === "user" ? "user" : "bot"}`;
      
      const contentDiv = document.createElement("div");
      contentDiv.className = "msg-content";
      
      const metaDiv = document.createElement("div");
      metaDiv.className = "msg-meta";
      metaDiv.textContent = role === "user" ? "You" : "AIMHSA";
      
      const textDiv = document.createElement("div");
      textDiv.className = "msg-text";
      textDiv.textContent = text;
      
      contentDiv.appendChild(metaDiv);
      contentDiv.appendChild(textDiv);
      msgDiv.appendChild(contentDiv);
      
      messagesEl.appendChild(msgDiv);
      ensureScroll();
      return msgDiv;
  }
  
  function createTypingIndicator() {
      if (typingEl) return;
      typingEl = document.createElement("div");
      typingEl.className = "msg bot";
      
      const contentDiv = document.createElement("div");
      contentDiv.className = "typing";
      
      const dotsDiv = document.createElement("div");
      dotsDiv.className = "typing-dots";
      dotsDiv.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
      
      contentDiv.appendChild(dotsDiv);
      typingEl.appendChild(contentDiv);
      messagesEl.appendChild(typingEl);
      ensureScroll();
  }
  
  function removeTypingIndicator() {
      if (!typingEl) return;
      typingEl.remove();
      typingEl = null;
  }
  
  async function api(path, opts) {
    const url = API_ROOT + path;
    const res = await fetch(url, opts);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || res.statusText);
    }
    return res.json();
  }

  async function initSession(useAccount = false) {
    const payload = {};
    if (useAccount && account) payload.account = account;
    try {
      const resp = await api("/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      convId = resp.id;
      localStorage.setItem("aimhsa_conv", convId);
      await loadHistory();
      await updateHistoryList();
    } catch (err) {
      console.error("session error", err);
      appendMessage("bot", "Could not start session. Try again.");
    }
  }

  // helper to generate a client-side conv id when needed (fallback)
  function newConvId() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
    return "conv-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2,8);
  }

  async function loadHistory() {
    if (!convId) return;
    try {
      const resp = await api("/history?id=" + encodeURIComponent(convId));
      messagesEl.innerHTML = "";
      const hist = resp.history || [];
      for (const m of hist) {
        appendMessage(m.role, m.content);
      }
      if (resp.attachments && resp.attachments.length) {
        resp.attachments.forEach(a => {
          appendMessage("bot", `Attachment (${a.filename}):\n` + (a.text.slice(0,400) + (a.text.length>400?"...[truncated]":"")));
        });
      }
      ensureScroll();
    } catch (err) {
      console.error("history load error", err);
    }
  }

  // Auto-resize textarea
  function autoResizeTextarea() {
    queryInput.style.height = 'auto';
    const scrollHeight = queryInput.scrollHeight;
    const maxHeight = 120; // Match CSS max-height
    queryInput.style.height = Math.min(scrollHeight, maxHeight) + 'px';
  }

  // Add textarea auto-resize listener
  queryInput.addEventListener('input', autoResizeTextarea);
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event('submit'));
    }
  });

  async function sendMessage(query) {
    if (!query) return;
    disableComposer(true);
    appendMessage("user", query);
    createTypingIndicator();
    queryInput.value = "";
    autoResizeTextarea(); // Reset textarea height
    
    try {
      // include account so server can bind new convs to the logged-in user
      const payload = { id: convId, query, history: [] };
      if (account) payload.account = account;
      const resp = await api("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      removeTypingIndicator();
      appendMessage("assistant", resp.answer || "(no answer)");
      if (resp.id && resp.id !== convId) {
        convId = resp.id;
        localStorage.setItem("aimhsa_conv", convId);
      }
      // refresh server-side conversation list for signed-in users
      if (account) await updateHistoryList();
    } catch (err) {
      console.error("ask error", err);
      removeTypingIndicator();
      appendMessage("bot", "Error contacting server. Try again.");
    } finally {
      disableComposer(false);
    }
  }

  // show upload preview block when a file is selected
  function showUploadPreview(file) {
    clearUploadPreview();
    const preview = document.createElement("div");
    preview.className = "upload-preview fade-in";
    preview.dataset.name = file.name;

    const icon = document.createElement("div");
    icon.style.fontSize = "20px";
    icon.textContent = "📄";

    const meta = document.createElement("div");
    meta.className = "upload-meta";
    const fname = document.createElement("div");
    fname.className = "upload-filename";
    fname.textContent = file.name;
    const fsize = document.createElement("div");
    fsize.className = "small";
    fsize.textContent = `${(file.size/1024).toFixed(1)} KB`;

    meta.appendChild(fname);
    meta.appendChild(fsize);

    const actions = document.createElement("div");
    actions.className = "upload-actions";
    const progress = document.createElement("div");
    progress.className = "progress-bar";
    const inner = document.createElement("div");
    inner.className = "progress-inner";
    progress.appendChild(inner);

    const removeBtn = document.createElement("button");
    removeBtn.className = "btn-small";
    removeBtn.type = "button";
    removeBtn.textContent = "Remove";
    removeBtn.addEventListener("click", () => {
      fileInput.value = "";
      clearUploadPreview();
    });

    actions.appendChild(progress);
    actions.appendChild(removeBtn);

    preview.appendChild(icon);
    preview.appendChild(meta);
    preview.appendChild(actions);

    // insert preview at left of composer (before send button)
    composer.insertBefore(preview, composer.firstChild);
    currentPreview = { el: preview, inner };
  }

  function updateUploadProgress(pct) {
    if (!currentPreview) return;
    currentPreview.inner.style.width = Math.max(0, Math.min(100, pct)) + "%";
  }

  function clearUploadPreview() {
    if (currentPreview && currentPreview.el) currentPreview.el.remove();
    currentPreview = null;
  }

  // Use XHR for upload to track progress
  function uploadPdf(file) {
    if (!file) return;
    disableComposer(true);
    showUploadPreview(file);

    const url = API_ROOT + "/upload_pdf";
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);

    xhr.upload.onprogress = function(e) {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        updateUploadProgress(pct);
      }
    };

    xhr.onload = function() {
      disableComposer(false);
      try {
        const resText = xhr.responseText || "{}";
        const data = JSON.parse(resText);
        if (xhr.status >= 200 && xhr.status < 300) {
          convId = data.id;
          localStorage.setItem("aimhsa_conv", convId);
          appendMessage("bot", `Uploaded ${data.filename}. What would you like to know about this document?`);
          clearUploadPreview();
          if (account) updateHistoryList();
        } else {
          appendMessage("bot", "PDF upload failed: " + (data.error || xhr.statusText));
        }
      } catch (err) {
        appendMessage("bot", "Upload parsing error");
      }
    };

    xhr.onerror = function() {
      disableComposer(false);
      appendMessage("bot", "PDF upload error");
    };

    const fd = new FormData();
    fd.append("file", file, file.name);
    if (convId) fd.append("id", convId);
    if (account) fd.append("account", account);
    xhr.send(fd);
  }

  function disableComposer(disabled) {
    if (disabled) {
      sendBtn.disabled = true;
      sendBtn.classList.add("sending");
      fileInput.disabled = true;
      queryInput.disabled = true;
    } else {
      sendBtn.disabled = false;
      sendBtn.classList.remove("sending");
      fileInput.disabled = false;
      queryInput.disabled = false;
    }
  }

  // New chat: require account (server enforces too)
  newChatBtn.addEventListener('click', async () => {
    if (!account) {
      appendMessage("bot", "Please sign in to create and view saved conversations.");
      return;
    }
    try {
      const payload = { account };
      const resp = await api("/conversations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (resp && resp.id) {
        convId = resp.id;
        localStorage.setItem("aimhsa_conv", convId);
        messagesEl.innerHTML = '';
        await updateHistoryList();
      }
    } catch (e) {
      console.error("failed to create conversation", e);
      appendMessage("bot", "Could not start new conversation. Try again.");
    }
  });

  // Clear only visual messages
  clearChatBtn.addEventListener("click", () => {
    if (!convId) return;
    
    if (confirm("Clear current messages? This will only clear the visible chat.")) {
      messagesEl.innerHTML = "";
      appendMessage("bot", "Messages cleared. How can I help you?");
    }
  });

  // Clear server-side history
  clearHistoryBtn.addEventListener("click", async () => {
    if (!convId) return;
    
    if (confirm("Are you sure? This will permanently clear all saved messages and attachments.")) {
      try {
        await api("/clear_chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: convId })
        });
        
        // Clear both messages and conversation history
        messagesEl.innerHTML = "";
        historyList.innerHTML = "";
        
        // Add default "no conversations" message
        const note = document.createElement('div');
        note.className = 'small';
        note.style.padding = '12px';
        note.style.color = 'var(--muted)';
        note.textContent = 'No conversations yet. Start a new chat!';
        historyList.appendChild(note);
        
        appendMessage("bot", "Chat history cleared. How can I help you?");
        
        // Start a new conversation
        const payload = { account };
        const resp = await api("/conversations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        
        if (resp && resp.id) {
          convId = resp.id;
          localStorage.setItem("aimhsa_conv", convId);
          await updateHistoryList();
        }
      } catch (err) {
        console.error("Failed to clear chat history", err);
        appendMessage("bot", "Failed to clear chat history on server. Try again.");
      }
    }
  });

  // show preview when file selected
  fileInput.addEventListener("change", (e) => {
    const f = fileInput.files[0];
    if (f) showUploadPreview(f);
    else clearUploadPreview();
  });

  const app = document.querySelector('.app');
  
  // Replace existing drag/drop handlers with:
  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    if (!e.dataTransfer.types.includes('Files')) return;
    app.classList.add('dragging');
  });

  document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    // Only remove if actually leaving the app
    if (e.target === document || e.target === app) {
      app.classList.remove('dragging');
    }
  });

  document.addEventListener('dragover', (e) => {
    e.preventDefault();
  });

  document.addEventListener('drop', (e) => {
    e.preventDefault();
    app.classList.remove('dragging');
    
    const files = Array.from(e.dataTransfer.files);
    const pdfFile = files.find(f => f.type === 'application/pdf');
    
    if (pdfFile) {
      fileInput.files = e.dataTransfer.files;
      const event = new Event('change');
      fileInput.dispatchEvent(event);
      uploadPdf(pdfFile);
    } else {
      appendMessage('bot', 'Please drop a PDF file.');
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = queryInput.value.trim();
    if (!q && !fileInput.files[0]) return;
    
    const file = fileInput.files[0];
    if (file) {
      uploadPdf(file);
      fileInput.value = "";
    } else {
      // ensure a convId exists for anonymous users too
      if (!convId) {
        convId = newConvId();
        localStorage.setItem("aimhsa_conv", convId);
      }
      sendMessage(q);
    }
  });

  // require signed-in account for server-backed conversations; otherwise show prompt
  async function updateHistoryList() {
    historyList.innerHTML = '';
    if (!account || account === 'null') {
      const note = document.createElement('div');
      note.className = 'small';
      note.style.padding = '12px';
      note.style.color = 'var(--text-muted)';
      note.textContent = 'Sign in to view and manage your conversation history.';
      historyList.appendChild(note);
      newChatBtn.disabled = true;
      newChatBtn.title = "Sign in to create server-backed conversations";
      return;
    }
    newChatBtn.disabled = false;
    newChatBtn.title = "";

    try {
      const q = "?account=" + encodeURIComponent(account);
      const resp = await api("/conversations" + q, { method: "GET" });
      const entries = resp.conversations || [];
      for (const historyData of entries) {
        const item = document.createElement('div');
        item.className = 'history-item' + (historyData.id === convId ? ' active' : '');
        const preview = document.createElement('div');
        preview.className = 'history-preview';
        preview.textContent = historyData.preview || 'New chat';
        item.appendChild(preview);
        item.addEventListener('click', () => switchConversation(historyData.id));
        historyList.appendChild(item);
      }
    } catch (e) {
      console.warn("failed to load conversations", e);
      const errNote = document.createElement('div');
      errNote.className = 'small';
      errNote.style.padding = '12px';
      errNote.style.color = 'var(--muted)';
      errNote.textContent = 'Unable to load conversations.';
      historyList.appendChild(errNote);
    }
  }

  // switch conversation -> set convId, persist selection and load history
  async function switchConversation(newConvId) {
    if (!newConvId || newConvId === convId) return;
    convId = newConvId;
    localStorage.setItem("aimhsa_conv", convId);
    await loadHistory();
    await updateHistoryList();
  }

  // initial load: start session (account-bound when available) and refresh history list
  (async () => {
    if (account) {
      await initSession(true);
    } else {
      await initSession(false);
    }
    await updateHistoryList();
  })();
})();