/**
 * Smart NLP Chatbot - Frontend Engine
 * Manages conversation, interactive messaging, and dynamic rendering of all 5 NLP cards.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatMessages = document.getElementById("chatMessages");
    const welcomeHero = document.getElementById("welcomeHero");
    const typingIndicator = document.getElementById("typingIndicator");
    const charCounter = document.getElementById("charCounter");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const newChatBtn = document.getElementById("newChatBtn");
    const togglePanelBtn = document.getElementById("togglePanelBtn");
    const copyJsonBtn = document.getElementById("copyJsonBtn");
    const toast = document.getElementById("toast");
    const historyList = document.getElementById("historyList");
    const historyCount = document.getElementById("historyCount");
    const activeQueryCard = document.getElementById("activeQueryCard");
    const activeQueryText = document.getElementById("activeQueryText");
    const nlpStatusText = document.getElementById("nlpStatusText");

    // Responsive Mobile Controls
    const sidebarLeft = document.getElementById("sidebarLeft");
    const panelRight = document.getElementById("panelRight");
    const openSidebarLeft = document.getElementById("openSidebarLeft");
    const closeSidebarLeft = document.getElementById("closeSidebarLeft");
    const closePanelBtn = document.getElementById("closePanelBtn");

    // Session State
    let conversationHistory = [];
    let currentNlpData = null;
    let isProcessing = false;

    // -------------------------------------------------------------------------
    // Toast Notification Utility
    // -------------------------------------------------------------------------
    function showToast(message) {
        if (!toast) return;
        toast.textContent = message;
        toast.classList.add("show");
        setTimeout(() => {
            toast.classList.remove("show");
        }, 2200);
    }

    // -------------------------------------------------------------------------
    // Auto-resizing Textarea & Character Counter
    // -------------------------------------------------------------------------
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = `${Math.min(userInput.scrollHeight, 140)}px`;
        const len = userInput.value.length;
        charCounter.textContent = `${len} / 1500`;
        sendBtn.disabled = len === 0 || isProcessing;
    });

    // Enter key sends, Shift+Enter creates a new line
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled && !isProcessing) {
                chatForm.dispatchEvent(new Event("submit"));
            }
        }
    });

    // -------------------------------------------------------------------------
    // Mobile Navigation & Panel Toggling
    // -------------------------------------------------------------------------
    if (openSidebarLeft) {
        openSidebarLeft.addEventListener("click", () => {
            sidebarLeft.classList.add("open");
        });
    }
    if (closeSidebarLeft) {
        closeSidebarLeft.addEventListener("click", () => {
            sidebarLeft.classList.remove("open");
        });
    }
    if (togglePanelBtn) {
        togglePanelBtn.addEventListener("click", () => {
            panelRight.classList.toggle("open");
            togglePanelBtn.classList.toggle("active");
        });
    }
    if (closePanelBtn) {
        closePanelBtn.addEventListener("click", () => {
            panelRight.classList.remove("open");
            togglePanelBtn.classList.remove("active");
        });
    }

    // -------------------------------------------------------------------------
    // Prompt Chips and Technique Clicks
    // -------------------------------------------------------------------------
    document.querySelectorAll(".chip-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const text = btn.getAttribute("data-text");
            if (text) {
                userInput.value = text;
                userInput.dispatchEvent(new Event("input"));
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    });

    document.querySelectorAll(".tech-item").forEach((item) => {
        item.addEventListener("click", () => {
            const query = item.getAttribute("data-query");
            if (query) {
                userInput.value = query;
                userInput.dispatchEvent(new Event("input"));
                chatForm.dispatchEvent(new Event("submit"));
                if (window.innerWidth <= 980) {
                    sidebarLeft.classList.remove("open");
                }
            }
        });
    });

    // -------------------------------------------------------------------------
    // New Chat & Clear Chat Handlers
    // -------------------------------------------------------------------------
    function resetChat() {
        chatMessages.innerHTML = "";
        if (welcomeHero) {
            chatMessages.appendChild(welcomeHero);
            welcomeHero.style.display = "block";
        }
        userInput.value = "";
        userInput.style.height = "auto";
        charCounter.textContent = "0 / 1500";
        activeQueryCard.style.display = "none";
        nlpStatusText.textContent = "Awaiting input";
        resetNlpCards();
    }

    newChatBtn.addEventListener("click", () => {
        resetChat();
        showToast("Started a new chat session");
        if (window.innerWidth <= 980) {
            sidebarLeft.classList.remove("open");
        }
    });

    clearChatBtn.addEventListener("click", () => {
        resetChat();
        showToast("Conversation cleared");
    });

    // -------------------------------------------------------------------------
    // Copy Full NLP JSON
    // -------------------------------------------------------------------------
    copyJsonBtn.addEventListener("click", () => {
        if (!currentNlpData) {
            showToast("No NLP data to copy yet!");
            return;
        }
        const jsonStr = JSON.stringify(currentNlpData, null, 2);
        navigator.clipboard.writeText(jsonStr).then(() => {
            showToast("Full NLP JSON copied to clipboard! 📋");
        }).catch(() => {
            showToast("Failed to copy JSON");
        });
    });

    // -------------------------------------------------------------------------
    // Format Markdown Helper (Bold, Lists, Line Breaks)
    // -------------------------------------------------------------------------
    function formatBotMessage(text) {
        if (!text) return "";
        let formatted = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Bold **text**
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        // Italics *text*
        formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");
        // Bullet points
        formatted = formatted.replace(/^• (.*?)$/gm, "<li>$1</li>");
        formatted = formatted.replace(/(<li>.*?<\/li>)/gs, "<ul>$1</ul>");
        // Line breaks
        formatted = formatted.replace(/\n\n/g, "</p><p>");
        formatted = formatted.replace(/\n/g, "<br>");

        return `<p>${formatted}</p>`;
    }

    // -------------------------------------------------------------------------
    // Append Message to Chat
    // -------------------------------------------------------------------------
    function appendMessage(role, text, timestamp, nlpPayload = null) {
        if (welcomeHero && welcomeHero.style.display !== "none") {
            welcomeHero.style.display = "none";
        }

        const row = document.createElement("div");
        row.className = `message-row message-${role}`;

        const avatar = document.createElement("div");
        avatar.className = `msg-avatar ${role}-avatar`;
        avatar.textContent = role === "user" ? "👤" : "🤖";

        const contentWrapper = document.createElement("div");
        contentWrapper.className = "msg-content-wrapper";

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";

        if (role === "user") {
            bubble.textContent = text;
        } else {
            bubble.innerHTML = formatBotMessage(text);
        }

        const meta = document.createElement("div");
        meta.className = "msg-meta";
        meta.innerHTML = `<span>${timestamp || "Just now"}</span>`;

        if (role === "user" && nlpPayload) {
            const inspectBtn = document.createElement("button");
            inspectBtn.className = "inspect-nlp-btn";
            inspectBtn.textContent = "View NLP Analysis";
            inspectBtn.addEventListener("click", () => {
                renderNlpDashboard(nlpPayload, text);
                if (window.innerWidth <= 980) {
                    panelRight.classList.add("open");
                    togglePanelBtn.classList.add("active");
                }
            });
            meta.appendChild(inspectBtn);
        }

        contentWrapper.appendChild(bubble);
        contentWrapper.appendChild(meta);

        row.appendChild(avatar);
        row.appendChild(contentWrapper);

        chatMessages.appendChild(row);
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: "smooth"
        });
    }

    // -------------------------------------------------------------------------
    // Add Query to Session History List
    // -------------------------------------------------------------------------
    function addHistoryItem(text, nlpPayload) {
        if (!historyList) return;
        const emptyNotice = historyList.querySelector(".history-empty");
        if (emptyNotice) {
            emptyNotice.remove();
        }

        const li = document.createElement("li");
        li.className = "history-item";
        li.title = text;
        li.innerHTML = `
            <span>💬</span>
            <span style="overflow:hidden; text-overflow:ellipsis;">${text}</span>
        `;

        li.addEventListener("click", () => {
            renderNlpDashboard(nlpPayload, text);
            document.querySelectorAll(".history-item").forEach(el => el.classList.remove("active"));
            li.classList.add("active");
            if (window.innerWidth <= 980) {
                sidebarLeft.classList.remove("open");
                panelRight.classList.add("open");
                togglePanelBtn.classList.add("active");
            }
        });

        historyList.prepend(li);
        conversationHistory.push({ text, nlp: nlpPayload });
        if (historyCount) {
            historyCount.textContent = conversationHistory.length;
        }
    }

    // -------------------------------------------------------------------------
    // Submit Message Handler
    // -------------------------------------------------------------------------
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || isProcessing) return;

        isProcessing = true;
        sendBtn.disabled = true;
        const timeNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Display user message in chat
        appendMessage("user", text, timeNow);

        // Clear input field
        userInput.value = "";
        userInput.style.height = "auto";
        charCounter.textContent = "0 / 1500";

        // Show typing indicator
        typingIndicator.style.display = "flex";
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: "smooth"
        });

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });

            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }

            const data = await response.json();

            // Hide typing indicator
            typingIndicator.style.display = "none";

            // Render bot reply
            appendMessage("bot", data.response, data.timestamp || timeNow);

            // Store active NLP data
            currentNlpData = data.nlp;

            // Render right side panel
            renderNlpDashboard(data.nlp, text);

            // Add to session history
            addHistoryItem(text, data.nlp);

        } catch (err) {
            typingIndicator.style.display = "none";
            appendMessage("bot", `⚠️ An error occurred while communicating with the local NLP engine: ${err.message}. Please verify the Flask server is running.`, timeNow);
        } finally {
            isProcessing = false;
            sendBtn.disabled = userInput.value.trim().length === 0;
            userInput.focus();
        }
    });

    // -------------------------------------------------------------------------
    // Reset NLP Cards
    // -------------------------------------------------------------------------
    function resetNlpCards() {
        currentNlpData = null;

        // Card 1
        document.getElementById("tokenCountBadge").textContent = "0 tokens";
        document.getElementById("tokenWordsCount").textContent = "0";
        document.getElementById("tokenPunctCount").textContent = "0";
        document.getElementById("tokenTotalCount").textContent = "0";
        document.getElementById("tokenChipsContainer").innerHTML = '<div class="empty-state-card">Send a message to view tokens.</div>';

        // Card 2
        document.getElementById("stemChangedBadge").textContent = "0 changed";
        document.querySelector("#stemmingTable tbody").innerHTML = '<tr><td colspan="3" class="table-empty">Send a message to view stemming results.</td></tr>';

        // Card 3
        document.getElementById("lemmaChangedBadge").textContent = "0 changed";
        document.querySelector("#lemmatizationTable tbody").innerHTML = '<tr><td colspan="4" class="table-empty">Send a message to view lemmatization results.</td></tr>';

        // Card 4
        document.getElementById("nerCountBadge").textContent = "0 entities";
        document.getElementById("nerHighlightBox").innerHTML = '<span class="empty-state-card">Send a message to detect named entities.</span>';
        document.getElementById("nerEntitiesList").innerHTML = "";

        // Card 5
        document.getElementById("sentimentBadge").textContent = "NEUTRAL";
        document.getElementById("sentimentIcon").textContent = "😐";
        document.getElementById("sentimentCategory").textContent = "NEUTRAL";
        document.getElementById("sentimentCompound").textContent = "Compound Score: 0.0000";
        document.getElementById("sentimentMood").textContent = "Balanced / Neutral";
        document.getElementById("posScoreVal").textContent = "0.0%";
        document.getElementById("posScoreBar").style.width = "0%";
        document.getElementById("neuScoreVal").textContent = "100.0%";
        document.getElementById("neuScoreBar").style.width = "100%";
        document.getElementById("negScoreVal").textContent = "0.0%";
        document.getElementById("negScoreBar").style.width = "0%";
    }

    // -------------------------------------------------------------------------
    // Render Complete NLP Dashboard
    // -------------------------------------------------------------------------
    function renderNlpDashboard(nlp, rawText) {
        if (!nlp) return;
        currentNlpData = nlp;

        // Active Query Banner
        if (activeQueryCard && activeQueryText) {
            activeQueryCard.style.display = "block";
            activeQueryText.textContent = rawText;
        }
        if (nlpStatusText) {
            nlpStatusText.textContent = "Analysis Complete";
        }

        // ==========================================
        // 1. TOKENIZATION
        // ==========================================
        const tok = nlp.tokenization || {};
        document.getElementById("tokenCountBadge").textContent = `${tok.total_tokens || 0} tokens`;
        document.getElementById("tokenWordsCount").textContent = tok.word_count || 0;
        document.getElementById("tokenPunctCount").textContent = tok.punctuation_count || 0;
        document.getElementById("tokenTotalCount").textContent = tok.total_tokens || 0;

        const chipsContainer = document.getElementById("tokenChipsContainer");
        chipsContainer.innerHTML = "";

        if (tok.details && tok.details.length > 0) {
            tok.details.forEach((item, index) => {
                const chip = document.createElement("span");
                chip.className = `token-chip type-${item.type}`;
                chip.innerHTML = `
                    <span class="token-index">${index + 1}</span>
                    <span>${escapeHtml(item.token)}</span>
                `;
                chipsContainer.appendChild(chip);
            });
        } else {
            chipsContainer.innerHTML = '<div class="empty-state-card">No tokens extracted.</div>';
        }

        // ==========================================
        // 2. STEMMING
        // ==========================================
        const stem = nlp.stemming || {};
        document.getElementById("stemChangedBadge").textContent = `${stem.changed_count || 0} modified`;

        const stemTbody = document.querySelector("#stemmingTable tbody");
        stemTbody.innerHTML = "";

        if (stem.results && stem.results.length > 0) {
            stem.results.forEach((row) => {
                const tr = document.createElement("tr");
                const statusHtml = row.changed
                    ? '<span class="badge-modified">Modified</span>'
                    : '<span class="badge-unchanged">Unchanged</span>';

                tr.innerHTML = `
                    <td><strong>${escapeHtml(row.original)}</strong></td>
                    <td style="color: #93C5FD; font-family: monospace;">${escapeHtml(row.stemmed)}</td>
                    <td>${statusHtml}</td>
                `;
                stemTbody.appendChild(tr);
            });
        } else {
            stemTbody.innerHTML = '<tr><td colspan="3" class="table-empty">No words stemmed.</td></tr>';
        }

        // ==========================================
        // 3. LEMMATIZATION
        // ==========================================
        const lemma = nlp.lemmatization || {};
        document.getElementById("lemmaChangedBadge").textContent = `${lemma.changed_count || 0} modified`;

        const lemmaTbody = document.querySelector("#lemmatizationTable tbody");
        lemmaTbody.innerHTML = "";

        if (lemma.results && lemma.results.length > 0) {
            lemma.results.forEach((row) => {
                const tr = document.createElement("tr");
                const statusHtml = row.changed
                    ? '<span class="badge-modified">Modified</span>'
                    : '<span class="badge-unchanged">Unchanged</span>';

                tr.innerHTML = `
                    <td><strong>${escapeHtml(row.original)}</strong></td>
                    <td><span class="pos-tag-pill" title="${escapeHtml(row.pos_category)}">${escapeHtml(row.pos_tag)}</span></td>
                    <td style="color: #67E8F9; font-weight: 600;">${escapeHtml(row.lemma)}</td>
                    <td>${statusHtml}</td>
                `;
                lemmaTbody.appendChild(tr);
            });
        } else {
            lemmaTbody.innerHTML = '<tr><td colspan="4" class="table-empty">No words lemmatized.</td></tr>';
        }

        // ==========================================
        // 4. NAMED ENTITY RECOGNITION
        // ==========================================
        const ner = nlp.ner || {};
        const entityCount = ner.total_entities || 0;
        document.getElementById("nerCountBadge").textContent = `${entityCount} detected`;

        const nerHighlightBox = document.getElementById("nerHighlightBox");
        const nerEntitiesList = document.getElementById("nerEntitiesList");
        nerEntitiesList.innerHTML = "";

        if (ner.has_entities && ner.entities && ner.entities.length > 0) {
            nerHighlightBox.innerHTML = ner.highlighted_html || escapeHtml(rawText);

            ner.entities.forEach((ent) => {
                const row = document.createElement("div");
                row.className = "ner-entity-row";
                row.style.borderLeftColor = ent.color || "#8B5CF6";

                row.innerHTML = `
                    <div class="ner-entity-left">
                        <span style="font-size: 16px;">${ent.icon || "🏷️"}</span>
                        <div>
                            <span class="ner-entity-name">${escapeHtml(ent.text)}</span>
                            <div style="font-size: 10px; color: #71717A;">${escapeHtml(ent.description || "")}</div>
                        </div>
                    </div>
                    <span class="ner-entity-label" style="background-color: ${ent.color || '#8B5CF6'}; color: #FFFFFF;">
                        ${escapeHtml(ent.label)}
                    </span>
                `;
                nerEntitiesList.appendChild(row);
            });
        } else {
            nerHighlightBox.innerHTML = escapeHtml(rawText);
            nerEntitiesList.innerHTML = `
                <div class="empty-state-card" style="color: #A1A1AA; background: rgba(39, 39, 42, 0.4);">
                    No entities detected in this sentence.
                </div>
            `;
        }

        // ==========================================
        // 5. SENTIMENT ANALYSIS
        // ==========================================
        const sent = nlp.sentiment || {};
        const cat = sent.category || "NEUTRAL";

        const sentBadge = document.getElementById("sentimentBadge");
        sentBadge.textContent = cat;
        sentBadge.style.color = sent.color || "#60A5FA";
        sentBadge.style.backgroundColor = sent.bg || "rgba(96, 165, 250, 0.15)";

        document.getElementById("sentimentIcon").textContent = sent.icon || "😐";
        document.getElementById("sentimentCategory").textContent = cat;
        document.getElementById("sentimentCategory").style.color = sent.color || "#FFFFFF";
        document.getElementById("sentimentCompound").textContent = `Compound Score: ${sent.compound_display || (sent.compound !== undefined ? sent.compound.toFixed(4) : "0.0000")}`;
        document.getElementById("sentimentMood").textContent = sent.mood || "Balanced / Neutral";

        const banner = document.getElementById("sentimentBanner");
        banner.style.backgroundColor = sent.bg || "rgba(96, 165, 250, 0.12)";
        banner.style.borderColor = sent.color ? `${sent.color}40` : "rgba(96, 165, 250, 0.25)";

        const posP = sent.pos_percent !== undefined ? sent.pos_percent : 0;
        const neuP = sent.neu_percent !== undefined ? sent.neu_percent : 100;
        const negP = sent.neg_percent !== undefined ? sent.neg_percent : 0;

        document.getElementById("posScoreVal").textContent = `${posP}%`;
        document.getElementById("posScoreBar").style.width = `${posP}%`;

        document.getElementById("neuScoreVal").textContent = `${neuP}%`;
        document.getElementById("neuScoreBar").style.width = `${neuP}%`;

        document.getElementById("negScoreVal").textContent = `${negP}%`;
        document.getElementById("negScoreBar").style.width = `${negP}%`;
    }

    // Helper: HTML escape
    function escapeHtml(str) {
        if (str === null || str === undefined) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});

// -----------------------------------------------------------------------------
// Global Toggle Card Function (For Accordion Expand / Collapse)
// -----------------------------------------------------------------------------
function toggleCard(cardId) {
    const card = document.getElementById(cardId);
    if (card) {
        card.classList.toggle("collapsed");
    }
}
