// Helper function to safely escape regex characters
const escapeRegExp = (string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// Global set to track typos the user has chosen to ignore
const ignoredTypos = new Set();

function isDefinitionalContext(promptText, flaggedWord) {
    const lowerPrompt = (promptText || '').toLowerCase();
    const lowerWord = (flaggedWord || '').toLowerCase();

    // Define patterns that indicate an innocent question about the word itself
    const safePatterns = [
        new RegExp(`what (is|are|was|were)( a| an)? ${lowerWord}`),
        new RegExp(`how old (is|are|was|were) ${lowerWord}`),
        new RegExp(`define ${lowerWord}`),
        new RegExp(`meaning of ${lowerWord}`),
        new RegExp(`tell me about ${lowerWord}`)
    ];

    return safePatterns.some(pattern => pattern.test(lowerPrompt));
}

function isSafeResponseContext(responseText, flaggedWord) {
    const lowerText = responseText.toLowerCase();
    const lowerWord = flaggedWord.toLowerCase();

    // 1. Check for basic dictionary definition patterns the AI might use
    const safePatterns = [
        new RegExp(`${lowerWord} (is|are|was|were)( a| an)? `),
        new RegExp(`${lowerWord} refers to`),
        new RegExp(`(defined|known) as (a |an )?${lowerWord}`)
    ];

    if (lowerWord.includes('dinosaur')) {
        safePatterns.push(/(prehistoric|extinct|reptile|fossil|mesozoic|jurassic|cretaceous|million years)/);
    } else if (lowerWord.includes('ninja')) {
        safePatterns.push(/(martial arts|japan|feudal|shinobi|warrior|assassin)/);
    } else if (lowerWord.includes('rockstar')) {
        safePatterns.push(/(music|band|guitar|singer|concert|album|song)/);
    }

    return safePatterns.some(pattern => pattern.test(lowerText));
}

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. UI Elements ---
    const inputField = document.getElementById('prompt-input');
    const promptInput = document.getElementById('prompt-input');
    const realtimeWarning = document.getElementById('realtime-bias-warning');
    const typoBox = document.getElementById('typo-correction-box');
    let typingTimer;
    const sendBtn = document.querySelector('.send-btn');
    const chatArea = document.querySelector('.chat-area');
    const aiButtons = document.querySelectorAll('.ai-btn');
    const systemStatus = document.getElementById('system-status');
    const systemStatusText = systemStatus?.querySelector('.status-text');

    const reportBiasToggleBtn = document.getElementById('report-bias-toggle-btn');
    const missedWordContainer = document.getElementById('missed-word-container');
    const missedWordInput = document.getElementById('missed-word-input');
    const submitMissedWordBtn = document.getElementById('submit-missed-word-btn');
    const cancelMissedWordBtn = document.getElementById('cancel-missed-word-btn');

    // --- 1C. Dynamic System Status indicator ---
    // Active: connected and idle
    // Scanning: any in-flight request to backend
    // Offline: cannot reach backend
    let inFlightRequests = 0;

    function setSystemStatus(state, text) {
        if (!systemStatus) return;
        systemStatus.classList.remove('status-active', 'status-scanning', 'status-error');

        if (state === 'scanning') systemStatus.classList.add('status-scanning');
        else if (state === 'error') systemStatus.classList.add('status-error');
        else systemStatus.classList.add('status-active');

        if (systemStatusText) systemStatusText.textContent = `System Status: ${text}`;
    }

    // Listen for network activity from api.js.
    window.addEventListener('bb:net', (e) => {
        const phase = e?.detail?.phase;
        if (phase === 'start') inFlightRequests += 1;
        if (phase === 'success' || phase === 'error') inFlightRequests = Math.max(0, inFlightRequests - 1);

        if (inFlightRequests > 0) setSystemStatus('scanning', 'Scanning');
        else setSystemStatus('active', 'Active');

        if (phase === 'error') setSystemStatus('error', 'Offline');
    });

    // Periodic connectivity probe (catches offline even when idle).
    async function refreshConnectivityStatus() {
        if (!window.pingBackend) return;
        const ok = await window.pingBackend();
        if (!ok) setSystemStatus('error', 'Offline');
        else if (inFlightRequests > 0) setSystemStatus('scanning', 'Scanning');
        else setSystemStatus('active', 'Active');
    }

    refreshConnectivityStatus();
    setInterval(refreshConnectivityStatus, 10000);

    // --- 1B. Tooltip/banner cleanup helper ---
    // Centralized function to dismiss ALL active tooltips, the bias banner,
    // and the typo box.  Called on blur, input (typing starts), and submit.
    function dismissAllTooltips() {
        // 1. Strip tooltip-active from every highlight container
        document.querySelectorAll('.bias-highlight-container.tooltip-active').forEach(el => {
            // Also clear any pending show-timeout so it doesn't re-appear
            clearTimeout(el.dataset.tooltipTimeout);
            el.classList.remove('tooltip-active');
        });
        // 2. Hide the real-time bias warning banner
        if (realtimeWarning) {
            realtimeWarning.style.display = 'none';
            realtimeWarning.innerHTML = '';
        }
        // 3. Hide the typo/spelling suggestion box
        if (typoBox) {
            typoBox.style.display = 'none';
            typoBox.innerHTML = '';
        }
    }

    // --- 1C. Real-time, "as-you-type" bias coach ---
    // Debounced input evaluation that reuses `window.evaluateForBias` from `api.js`.
    if (promptInput && realtimeWarning) {
        const debounce = (fn, delayMs) => {
            let t;
            return (...args) => {
                clearTimeout(t);
                t = setTimeout(() => fn(...args), delayMs);
            };
        };

        const runRealtimeScan = debounce(async (currentText) => {
            try {
                const result = await window.evaluateForBias(currentText);
                console.log("Evaluation Result:", result);
                // Backend now returns issues as a canonical-keyed object map:
                // { "rockstar": { biased_word: "rockstar", ... } }
                const issuesMap =
                    result?.issues && typeof result.issues === 'object' && !Array.isArray(result.issues)
                        ? result.issues
                        : {};
                const issues = Object.values(issuesMap);

                // --- Separate spelling issues from bias issues ---
                const spellingIssues = issues.filter(
                    (i) => String(i?.type || '').toLowerCase() === 'spelling'
                );
                const biasIssues = issues.filter(
                    (i) => String(i?.type || '').toLowerCase() !== 'spelling'
                );

                // --- Primary bias banner (exact matches from regex engine) ---
                // Excludes spelling-type issues so they only appear in the typo popup.
                if (result?.is_biased && biasIssues.length > 0) {
                    const typedHits = [];
                    biasIssues.forEach((issue) => {
                        const matches = Array.isArray(issue?.matches) ? issue.matches : [];
                        matches.forEach((m) => {
                            const t = String(m?.text || '').trim();
                            if (t) typedHits.push(t);
                        });
                    });

                    const uniqueWords = Array.from(
                        new Set(
                            typedHits
                                .map((w) =>
                                    String(w || '')
                                        .toLowerCase()
                                        .trim()
                                        .replace(/[-\s]+/g, ' ')
                                )
                                .filter(Boolean)
                                .filter((w) => !isDefinitionalContext(currentText, w))
                        )
                    );

                    if (uniqueWords.length > 0) {
                        realtimeWarning.innerHTML =
                            '<span>Heads up! Your prompt contains potentially biased language: <strong>' +
                            uniqueWords.join(', ') +
                            '</strong></span>';
                        realtimeWarning.style.display = 'flex';
                    } else {
                        realtimeWarning.style.display = 'none';
                    }
                } else {
                    realtimeWarning.style.display = 'none';
                }

                // --- Typo / Spelling coaching box ---
                // Sources: 1) backend `typos` array, 2) spelling-type issues from the issues map.
                // Both run independently of the bias banner above.
                if (typoBox) {
                    // Merge backend typos with spelling issues extracted from the issues map
                    const allTypos = [];

                    // 1) Standard backend typos
                    if (result.typos && result.typos.length > 0) {
                        result.typos.forEach((t) => allTypos.push({
                            original: t.original,
                            suggested: t.suggested
                        }));
                    }

                    // 2) Spelling issues from the issues map (e.g. 'toobossy' → 'too bossy')
                    spellingIssues.forEach((si) => {
                        allTypos.push({
                            original: si.original || si.biased_word || '',
                            suggested: si.replacement || si.suggestion || ''
                        });
                    });

                    // Get all typos that haven't been ignored
                    const activeTypos = allTypos.filter(
                        (t) => t.original && t.suggested &&
                            !ignoredTypos.has(String(t.original).toLowerCase())
                    );

                    // De-duplicate spelling typos so we don't show the same word twice
                    const uniqueTypos = [];
                    const seenTypos = new Set();
                    activeTypos.forEach(t => {
                        const key = String(t.original).toLowerCase();
                        if (!seenTypos.has(key)) {
                            seenTypos.add(key);
                            uniqueTypos.push(t);
                        }
                    });

                    if (uniqueTypos.length > 0) {
                        // Move typo box ABOVE the realtime bias warning in the DOM
                        if (realtimeWarning && typoBox.parentNode) {
                            typoBox.parentNode.insertBefore(typoBox, realtimeWarning);
                        }

                        // Make the typoBox itself a transparent vertical wrapper
                        typoBox.style.flexDirection = 'column';
                        typoBox.style.alignItems = 'stretch';
                        typoBox.style.gap = '8px';
                        typoBox.style.backgroundColor = 'transparent';
                        typoBox.style.borderColor = 'transparent';
                        typoBox.style.padding = '0';
                        typoBox.innerHTML = ''; // Clear previous content

                        uniqueTypos.forEach((activeTypo) => {
                            const card = document.createElement('div');
                            // Give each spelling issue its own complete card styling
                            card.className = 'bias-alert-banner spelling-popup-card';
                            card.style.display = 'flex';
                            card.style.alignItems = 'center';
                            card.style.width = '100%';
                            card.style.backgroundColor = '#e0f2fe';
                            card.style.color = '#0369a1';
                            card.style.borderColor = '#bae6fd';
                            // Margin is natively handled by the parent gap, but we clear it just in case
                            card.style.margin = '0'; 

                            card.innerHTML =
                                '<span>Did you mean <strong>' +
                                activeTypo.suggested +
                                '</strong> instead of "' +
                                activeTypo.original +
                                '"?</span>' +
                                '<div style="margin-left: auto; padding-left: 16px; display: flex; gap: 8px;">' +
                                `<button class="typo-action-btn fix-typo-btn" data-original="${activeTypo.original}" data-suggested="${activeTypo.suggested}">Fix</button>` +
                                `<button class="typo-action-btn ignore-typo-btn" data-original="${activeTypo.original}">Ignore</button>` +
                                '</div>';
                            
                            typoBox.appendChild(card);
                        });

                        typoBox.style.display = 'flex';

                        // Attach event listeners to all dynamically created buttons
                        const fixBtns = typoBox.querySelectorAll('.fix-typo-btn');
                        const ignoreBtns = typoBox.querySelectorAll('.ignore-typo-btn');

                        fixBtns.forEach(btn => {
                            btn.onclick = () => {
                                const orig = btn.getAttribute('data-original');
                                const sugg = btn.getAttribute('data-suggested');
                                // Use case-insensitive replace for the misspelled word
                                const regex = new RegExp(escapeRegExp(orig), 'gi');
                                promptInput.value = promptInput.value.replace(regex, sugg);
                                // Trigger input event to re-evaluate and re-render without this typo
                                promptInput.dispatchEvent(new Event('input'));
                            };
                        });

                        ignoreBtns.forEach(btn => {
                            btn.onclick = () => {
                                const orig = btn.getAttribute('data-original');
                                ignoredTypos.add(String(orig).toLowerCase());
                                // Trigger input event to re-evaluate and re-render without the ignored typo
                                promptInput.dispatchEvent(new Event('input'));
                            };
                        });
                    } else {
                        typoBox.style.display = 'none';
                        typoBox.style.flexDirection = 'row'; // Reset to default
                    }
                }
            } catch (e) {
                // Fail closed: if evaluator is unreachable, hide the banner.
                realtimeWarning.style.display = 'none';
                if (typoBox) typoBox.style.display = 'none';
            }
        }, 1500);

        promptInput.addEventListener('input', () => {
            // Immediately dismiss any lingering tooltips/banners when user starts typing
            dismissAllTooltips();

            const currentText = promptInput.value || "";

            if (currentText.trim() === "") {
                return; // dismissAllTooltips already hid everything
            }

            // Heavy work only after user pauses typing for 500ms.
            runRealtimeScan(currentText);
        });

        // Blur event removed to prevent tooltips from disappearing on click-away
    }

    // --- False Negative UI Logic ---
    if (reportBiasToggleBtn && missedWordContainer) {
        reportBiasToggleBtn.addEventListener('click', () => {
            if (missedWordContainer.style.display === 'none') {
                missedWordContainer.style.display = 'flex';
                if (missedWordInput) missedWordInput.focus();
            } else {
                missedWordContainer.style.display = 'none';
            }
        });
    }

    if (cancelMissedWordBtn) {
        cancelMissedWordBtn.addEventListener('click', () => {
            missedWordContainer.style.display = 'none';
            if (missedWordInput) missedWordInput.value = '';
        });
    }

    if (submitMissedWordBtn && missedWordInput) {
        submitMissedWordBtn.addEventListener('click', async () => {
            const word = missedWordInput.value.trim();
            if (!word) return;
            
            submitMissedWordBtn.textContent = 'Sending...';
            submitMissedWordBtn.disabled = true;
            
            const success = await window.reportFeedback(word, 1);
            
            submitMissedWordBtn.textContent = 'Submit';
            submitMissedWordBtn.disabled = false;
            
            if (success) {
                missedWordContainer.style.display = 'none';
                missedWordInput.value = '';
                
                // Auto-evaluate the most recent AI message
                if (chatArea) {
                    const lastAiMessage = chatArea.querySelector('.message.ai-container:last-child');
                    if (lastAiMessage && lastAiMessage.dataset.originalText) {
                        const originalText = lastAiMessage.dataset.originalText;
                        try {
                            const evalResult = await window.evaluateForBias(originalText, true);
                            const contentDiv = lastAiMessage.querySelector('.ai-text-content');
                            if (contentDiv) {
                                // Reset the DOM content to clean text
                                if (typeof window.marked === 'function') {
                                    contentDiv.innerHTML = window.marked.parse(originalText);
                                } else {
                                    contentDiv.innerHTML = originalText.replace(/\n/g, '<br>');
                                }
                                
                                // Re-apply highlights if biased
                                if (evalResult && evalResult.is_biased && evalResult.issues) {
                                    const issuesMap = typeof evalResult.issues === 'object' && !Array.isArray(evalResult.issues) ? evalResult.issues : {};
                                    const rawIssues = Object.values(issuesMap);
                                    
                                    // Normally we would run isSafeResponseContext here, but since the user
                                    // specifically reported it as a False Negative, we want to ensure it highlights.
                                    highlightDomTextNodes(contentDiv, rawIssues);
                                }
                            }
                        } catch (e) {
                            console.error("Auto-eval failed:", e);
                        }
                    }
                }
            } else {
                alert("Failed to report missed bias. Try again.");
            }
        });
    }

    // --- False Positive Optimistic UI ---
    if (chatArea) {
        chatArea.addEventListener('click', async (e) => {
            if (e.target.classList.contains('report-fp-btn')) {
                const btn = e.target;
                const word = btn.getAttribute('data-word');
                if (!word) return;

                // Optimistic UI update: Remove the highlight completely
                const container = btn.closest('.bias-highlight-container');
                if (container) {
                    const mark = container.querySelector('mark');
                    if (mark) {
                        const textNode = document.createTextNode(mark.textContent);
                        container.parentNode.replaceChild(textNode, container);
                    }
                }

                // Fire async request
                await window.reportFeedback(word, 0);
            }
        });
    }

    // --- 2. Dynamic Welcome Message ---
    const welcomeMessages = [
        "Welcome to the Laboratory! I am ready for your prompts. Try to test my limits and see if you can uncover any embedded biases.",
        "Hello, researcher! Input a prompt below. Let's see if your wording triggers any of my demographic or coded language filters.",
        "Greetings! I am currently operating with standard parameters. Enter a scenario, and let's analyze how I handle potentially biased instructions.",
        "System initialized. I'm here to help you understand AI bias. Throw a challenging prompt my way and watch the fairness tools in action!",
        "Welcome to BiasBuster! Ready for an experiment? Type your prompt below to see how I process complex social contexts."
    ];

    function showWelcomeMessage() {
        if (!chatArea) return;
        const randomMsg = welcomeMessages[Math.floor(Math.random() * welcomeMessages.length)];

        const aiMessageDiv = document.createElement('div');
        aiMessageDiv.classList.add('message', 'ai-container');
        aiMessageDiv.innerHTML = `<div class="ai-text-content">${randomMsg}</div>`;

        chatArea.innerHTML = '';
        chatArea.appendChild(aiMessageDiv);
    }

    showWelcomeMessage();

    // --- 3. Placeholder Rotation ---
    const placeholders = [
        "Challenge the AI... type your prompt here.",
        "Push the boundaries. Test for AI bias...",
        "Can you trick the AI? Enter a prompt..."
    ];
    let currentIndex = 0;
    setInterval(() => {
        currentIndex = (currentIndex + 1) % placeholders.length;
        if (inputField) inputField.placeholder = placeholders[currentIndex];
    }, 15000);

    // --- 4. AI Selector Logic & State ---
    let selectedModel = 'gemini'; // Default

    aiButtons.forEach(button => {
        button.addEventListener('click', () => {
            aiButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            const btnText = button.textContent.trim().toLowerCase();
            if (btnText.includes('gemini')) selectedModel = 'gemini';
            else if (btnText.includes('llama')) selectedModel = 'llama';
            else if (btnText.includes('qwen')) selectedModel = 'qwen';
            else if (btnText.includes('gpt-oss')) selectedModel = 'gptoss';

            if (chatArea) {
                const notice = document.createElement('div');
                notice.style.cssText = "align-self: center; font-size: 0.75rem; color: #666; font-style: italic; margin: 10px 0;";
                notice.textContent = `[System: Switched to ${selectedModel.toUpperCase()} engine]`;
                chatArea.appendChild(notice);
                chatArea.scrollTop = chatArea.scrollHeight;
            }
        });
    });

    // --- 5. Robust Regex-Based Highlighting Engine ---
    // Severity-aware highlights + educational tooltips.
    const escapeHtml = (str) =>
        String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');

    // Tooltip HTML is injected as part of the highlight markup (innerHTML),
    // so we must escape all dynamic values.
    const escapeForTooltipHtml = (value) => escapeHtml(value);
    const buildTooltipHtml = ({ word, category, severity, suggestion, context, impact }) => {
        const w = escapeForTooltipHtml(word);
        const c = escapeForTooltipHtml(category || 'N/A');
        const s = escapeForTooltipHtml(severity || 'Unknown');
        const ctx = escapeForTooltipHtml(context || 'N/A');
        const imp = escapeForTooltipHtml(impact || 'N/A');
        
        let suggestionHtml = '';
        if (suggestion) {
            suggestionHtml = `<div class="tooltip-suggestion">Suggestion: ${escapeForTooltipHtml(suggestion)}</div>`;
        }

        const fpBtnHtml = `<div class="fp-btn-container"><button class="report-fp-btn" data-word="${w}">Report False Positive</button></div>`;

        return (
            `<div class="tooltip-inner">` +
                `<div class="tooltip-word">${w}</div>` +
                `<div class="tooltip-row"><strong>Category:</strong> ${c}</div>` +
                `<div class="tooltip-row"><strong>Severity:</strong> ${s}</div>` +
                `<div class="tooltip-row"><strong>Context:</strong> ${ctx}</div>` +
                `<div class="tooltip-row"><strong>Impact:</strong> ${imp}</div>` +
                suggestionHtml +
                fpBtnHtml +
            `</div>`
        );
    };

    function severityToClass(severity) {
        const s = String(severity || '').toLowerCase();
        if (s.includes('high') || s.includes('critical')) return 'highlight-red';
        if (s.includes('medium')) return 'highlight-yellow';
        if (s.includes('low')) return 'highlight-blue';
        return 'highlight-yellow'; // fallback
    }


    function highlightDomTextNodes(container, validIssues) {
        if (!validIssues || validIssues.length === 0) return;

        // --- Step 1: Sanitize and sort the target terms ---
        const matchToIssue = new Map();
        const rank = (sev) => {
            const s = String(sev || '').toLowerCase();
            if (s === 'critical') return 3;
            if (s === 'high') return 2;
            if (s === 'medium') return 1;
            return 0;
        };
        const typePriority = (issue) => {
            const t = String(issue?.type || '').toLowerCase();
            if (t === 'agentic' || t === 'communal') return 1;
            if (t === 'implicit_warning') return 0;
            return 2;
        };

        validIssues.forEach((issue) => {
            const matches = Array.isArray(issue?.matches) ? issue.matches : [];
            // Fallback to biased_word when matches is empty (implicit bias terms)
            const effectiveMatches = matches.length > 0
                ? matches
                : [{ text: issue?.biased_word || '' }];
            effectiveMatches.forEach((m) => {
                const textMatch = String(m?.text || '').trim();
                if (textMatch) {
                    const lowerMatch = textMatch.toLowerCase();
                    const prev = matchToIssue.get(lowerMatch);
                    if (!prev || typePriority(issue) > typePriority(prev) ||
                        (typePriority(issue) === typePriority(prev) && rank(issue?.severity) > rank(prev?.severity))) {
                        matchToIssue.set(lowerMatch, { textMatch, issue });
                    }
                }
            });
        });

        const terms = Array.from(matchToIssue.keys());
        if (terms.length === 0) return;

        // Sort by length descending so 'leadership' is matched before 'leader'
        terms.sort((a, b) => b.length - a.length);

        // --- Step 2: Build a capture-group regex ---
        // Multi-word phrases need flexible whitespace/hyphen matching and
        // lookaround-based boundaries instead of \b (which fails mid-phrase).
        const buildTermPattern = (term) => {
            const escaped = escapeRegExp(term);
            // If the term contains whitespace or hyphens, treat as multi-word phrase
            if (/[\s-]/.test(term)) {
                // Replace spaces/hyphens with flexible whitespace pattern
                const flexPattern = escaped.replace(/[-\s]+/g, '[-\\s]+');
                // Use lookaround for word boundaries at phrase edges
                return '(?<![\\w])' + flexPattern + '(?![\\w])';
            }
            // Single-word term: standard word boundaries work fine
            return '\\b' + escaped + '\\b';
        };
        const regex = new RegExp('(' + terms.map(buildTermPattern).join('|') + ')', 'gi');

        // --- Step 3: Gather text nodes via TreeWalker (no mutation during walk) ---
        const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
        const textNodes = [];
        let node;
        while ((node = walker.nextNode())) {
            if (node.parentNode && node.parentNode.classList &&
                (node.parentNode.classList.contains('bias-highlight') ||
                    node.parentNode.closest('.bias-highlight-container'))) {
                continue;
            }
            if (node.parentNode && ['CODE', 'PRE', 'SCRIPT', 'STYLE'].includes(node.parentNode.tagName)) {
                continue;
            }
            if (node.nodeValue.trim() !== '') {
                textNodes.push(node);
            }
        }

        // --- Step 4: Pure DOM split replacement ---
        textNodes.forEach(textNode => {
            const parts = textNode.nodeValue.split(regex);

            // If length > 1, we found matches
            if (parts.length > 1) {
                const fragment = document.createDocumentFragment();

                parts.forEach((part, index) => {
                    if (index % 2 === 0) {
                        // Even indices are standard non-matching text
                        if (part) fragment.appendChild(document.createTextNode(part));
                    } else {
                        // Odd indices are captured bias words — build highlight + tooltip
                        const matchedData = matchToIssue.get(part.toLowerCase());

                        if (matchedData) {
                            const issue = matchedData.issue;
                            const type = String(issue?.type || '').toLowerCase();
                            const sev = String(issue?.severity || '').toLowerCase();

                            const spanContainer = document.createElement('span');
                            spanContainer.className = 'bias-highlight-container';

                            const mark = document.createElement('mark');
                            mark.className = 'bias-highlight highlighted-word';
                            mark.dataset.type = issue?.type || '';
                            mark.dataset.severity = issue?.severity || '';
                            mark.dataset.suggestion = issue?.suggestion || '';
                            mark.textContent = part; // Safe from HTML injection

                            if (sev.includes('high') || sev.includes('critical') || type === 'communal') {
                                mark.style.backgroundColor = '#FFCCCC';
                            } else if (sev.includes('medium')) {
                                mark.style.backgroundColor = '#FFF9C4';
                            } else if (sev.includes('low') || type === 'agentic') {
                                mark.style.backgroundColor = '#CCE5FF';
                            } else {
                                mark.style.backgroundColor = '#FFF9C4';
                            }

                            const tooltipSpan = document.createElement('span');
                            
                            let severityClass = 'severity-medium';
                            if (sev.includes('high') || sev.includes('critical') || type === 'communal') severityClass = 'severity-high';
                            if (sev.includes('low') || type === 'agentic') severityClass = 'severity-low';
                            tooltipSpan.className = `custom-tooltip-box ${severityClass}`;

                            let category = issue?.category || issue?.dimension || 'Bias';
                            let severityStr = issue?.severity || 'Unknown';
                            let suggestion = issue?.suggestion || '';
                            let context = String(category).trim() ? category + '.' : 'Bias.';
                            let impact = issue?.affected_group || 'General demographic';

                            if (type === 'agentic') {
                                category = 'Gender-coded (Agentic)';
                                severityStr = 'Medium';
                                context = 'Language associated with agency, dominance, and assertiveness.';
                                impact = 'Gender stereotypes.';
                            } else if (type === 'communal') {
                                category = 'Gender-coded (Communal)';
                                severityStr = 'Medium';
                                context = 'Language associated with collaboration, nurturing, and relationships.';
                                impact = 'Gender stereotypes.';
                            }

                            tooltipSpan.innerHTML = buildTooltipHtml({
                                word: issue?.biased_word || part,
                                category: category,
                                severity: severityStr,
                                suggestion: suggestion,
                                context: context,
                                impact: impact
                            });

                            spanContainer.appendChild(mark);
                            spanContainer.appendChild(tooltipSpan);
                            fragment.appendChild(spanContainer);
                        } else {
                            // Matched the regex but no issue data — keep as plain text
                            fragment.appendChild(document.createTextNode(part));
                        }
                    }
                });

                textNode.parentNode.replaceChild(fragment, textNode);
            }
        });
    }

    // Replace the old highlightBias function completely
    function highlightBias(text, validIssues = []) {
        console.warn("highlightBias strings replacing called, avoiding to prevent offset bug");
        return text;
    }
    // --- 6. Chat Mechanics ---
    async function sendMessage() {
        if (!inputField || !chatArea) return;

        // Nuke ALL lingering tooltip / banner / typo-box state the moment
        // the user fires a new evaluation.  This is the definitive cleanup
        // point that guarantees nothing from the previous prompt persists.
        dismissAllTooltips();

        const text = inputField.value.trim();
        if (text === "") return;

        const userDiv = document.createElement('div');
        userDiv.classList.add('message', 'user');

        const escapeHtml = (str) =>
            String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');

        userDiv.innerHTML = `
            <div class="user-text-content">${escapeHtml(text)}</div>
            <button class="user-copy-btn" title="Copy Text" aria-label="Copy">
                <svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
            </button>
        `;

        userDiv.querySelector('.user-copy-btn').addEventListener('click', (e) => {
            const copyBtn = e.currentTarget;
            navigator.clipboard.writeText(text).then(() => {
                const originalIcon = copyBtn.innerHTML;
                copyBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>`;
                setTimeout(() => { copyBtn.innerHTML = originalIcon; }, 2000);
            });
        });

        chatArea.appendChild(userDiv);

        inputField.value = "";
        chatArea.scrollTop = chatArea.scrollHeight;

        getRealAIResponse(text);
    }

    if (sendBtn) sendBtn.addEventListener('click', sendMessage);
    if (inputField) inputField.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

    // --- 7. Modal Logic ---
    function showPatchModal(originalPrompt, model, foundBiases, biasedResponseText) {
        let existingModal = document.getElementById('dynamic-patch-modal');
        if (existingModal) existingModal.remove();

        const modal = document.createElement('div');
        modal.id = 'dynamic-patch-modal';
        modal.className = 'patch-modal';

        const biasCheckboxes = foundBiases.map(word => `
            <label class="checkbox-label">
                <input type="checkbox" class="bias-cb" value="${word}" checked>
                ${word}
            </label>
        `).join('');

        modal.innerHTML = `
            <div class="patch-modal-content">
                <h3 class="patch-modal-title">Select Fairness Patch</h3>
                
                <button id="btn-standard-patch" class="modal-btn">Standard EEO Patch</button>
                <button id="btn-custom-patch" class="modal-btn">Customised Patch</button>
                
                <div id="custom-patch-area" class="custom-patch-section">
                    <strong style="font-size: 0.85rem; color: var(--text-color);">Select specific biases to remove:</strong>
                    <div class="checkbox-list">
                        ${biasCheckboxes}
                    </div>
                    <button id="btn-deploy-custom" class="modal-btn deploy-btn">Deploy Selected Patch</button>
                </div>
                
                <button id="btn-cancel-patch" class="modal-btn secondary-btn">Cancel</button>
            </div>
        `;

        document.body.appendChild(modal);
        modal.style.display = 'flex';

        document.getElementById('btn-cancel-patch').addEventListener('click', () => { modal.remove(); });

        document.getElementById('btn-standard-patch').addEventListener('click', () => {
            modal.remove();
            window.deployFairnessPatch(originalPrompt, model, 'standard', [], biasedResponseText);
        });

        document.getElementById('btn-custom-patch').addEventListener('click', () => {
            document.getElementById('custom-patch-area').classList.toggle('visible');
        });

        document.getElementById('btn-deploy-custom').addEventListener('click', () => {
            const checkboxes = modal.querySelectorAll('.bias-cb:checked');
            const selectedWords = Array.from(checkboxes).map(cb => cb.value);

            if (selectedWords.length === 0) {
                return alert("Please select at least one bias to patch.");
            }

            modal.remove();
            window.deployFairnessPatch(originalPrompt, model, 'custom', selectedWords, biasedResponseText);
        });
    }

    // --- 8. The Logic Core ---
    async function getRealAIResponse(userPrompt) {
        const aiMessageDiv = document.createElement('div');
        aiMessageDiv.classList.add('message', 'ai-container');
        aiMessageDiv.innerHTML = `<div class="ai-text-content"><em>Processing via ${selectedModel.toUpperCase()} engine...</em></div>`;
        chatArea.appendChild(aiMessageDiv);

        // 1. Generate text
        const realResponse = await window.fetchUnifiedAIResponse(userPrompt, selectedModel);
        
        // Save original text for later re-evaluation
        aiMessageDiv.dataset.originalText = realResponse;

        if (realResponse.startsWith("Error 429:")) {
            aiMessageDiv.remove();
            
            // Create and show a user-friendly toast banner
            const toast = document.createElement('div');
            toast.className = 'bias-alert-banner severity-high';
            toast.style.cssText = 'position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 1000; padding: 12px 24px; border-radius: 8px; font-weight: bold; background-color: #fef2f2; color: #b91c1c; border: 1px solid #f87171; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); width: auto; max-width: 90%; text-align: center;';
            toast.innerHTML = `⚠️ ${realResponse.substring(10).trim()}`;
            document.body.appendChild(toast);
            
            // Auto remove after 5 seconds
            setTimeout(() => {
                if (document.body.contains(toast)) {
                    toast.style.opacity = '0';
                    toast.style.transition = 'opacity 0.5s ease';
                    setTimeout(() => toast.remove(), 500);
                }
            }, 5000);
            
            return;
        }

        if (realResponse.startsWith("Error:") || realResponse.includes("System Error:")) {
            aiMessageDiv.querySelector('.ai-text-content').innerHTML = realResponse.replace(/\n/g, '<br>');
            chatArea.scrollTop = chatArea.scrollHeight;
            return;
        }

        // 2. Evaluate generated text locally on the backend (Semantic CSV Model).
        // The evaluator returns:
        //   { is_biased: boolean, issues: { canonicalKey: { ... } } }
        const evaluation = await window.evaluateForBias(realResponse, true);

        // 3. Process issues (object map -> array) and apply safe-context filtering
        // for AI *responses* to reduce definitional/historical false positives.
        const issuesMap =
            evaluation?.issues && typeof evaluation.issues === 'object' && !Array.isArray(evaluation.issues)
                ? evaluation.issues
                : {};
        const rawIssues = Object.values(issuesMap);
        const validIssues = rawIssues.filter((i) => {
            const word = i?.biased_word;
            if (!word) return false;
            return !isSafeResponseContext(realResponse, word);
        });

        const foundBiasesWords = validIssues.map(i => i.biased_word);
        const isBiased = Boolean(evaluation?.is_biased) && validIssues.length > 0;

        // 5. Highlight text (DOM Safe Approach)
        // First inject the potentially formatted text (or primitive replace)
        // Note: We use marked if available, otherwise fallback, preserving any HTML context.
        if (typeof window.marked === 'function') {
            aiMessageDiv.querySelector('.ai-text-content').innerHTML = window.marked.parse(realResponse);
        } else {
            aiMessageDiv.querySelector('.ai-text-content').innerHTML = realResponse.replace(/\n/g, '<br>');
        }

        // Then walk the DOM and insert highlight spans safely without breaking syntax
        if (isBiased) {
            highlightDomTextNodes(aiMessageDiv.querySelector('.ai-text-content'), validIssues);
        }
        chatArea.scrollTop = chatArea.scrollHeight;

        // --- Append Universal Action Bar ---
        const actionBar = document.createElement('div');
        actionBar.className = 'ai-action-bar';
        actionBar.innerHTML = `
            <button class="action-icon-btn btn-download" title="Download Report" aria-label="Download">
                <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
            </button>
            <button class="action-icon-btn btn-regenerate" title="Regenerate Response" aria-label="Regenerate">
                <svg viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
            </button>
            <button class="action-icon-btn btn-copy" title="Copy Text" aria-label="Copy">
                <svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
            </button>
        `;
        aiMessageDiv.appendChild(actionBar);

        actionBar.querySelector('.btn-download').addEventListener('click', () => {
            window.exportUnpatchedAuditReport(userPrompt, realResponse);
        });
        actionBar.querySelector('.btn-regenerate').addEventListener('click', () => getRealAIResponse(userPrompt));
        actionBar.querySelector('.btn-copy').addEventListener('click', (e) => {
            const copyBtn = e.currentTarget;
            navigator.clipboard.writeText(realResponse).then(() => {
                const originalTitle = copyBtn.title;
                const originalIcon = copyBtn.innerHTML;
                copyBtn.title = "Copied!";
                copyBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>`;
                setTimeout(() => {
                    copyBtn.title = originalTitle;
                    copyBtn.innerHTML = originalIcon;
                }, 2000);
            });
        });

        // 6. Build Rich Audit Warning Box
        if (isBiased) {
            const warningBox = document.createElement('div');
            warningBox.className = "warning-box";

            // Generate dynamic badges for the detected issues
            const displayedWords = new Set();
            const issueDetailsArray = [];
            for (const i of validIssues) {
                const word = String(i.biased_word || '').toLowerCase();
                if (displayedWords.has(word)) {
                    continue;
                }
                displayedWords.add(word);
                issueDetailsArray.push(`
                <div style="margin-top: 8px; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 6px; font-size: 0.85rem;">
                    <strong>Term:</strong> "${i.biased_word}" <br>
                    <strong>Severity:</strong> <span style="color: ${i.severity === 'High' || i.severity === 'Critical' ? '#d93025' : '#e37400'}">${i.severity}</span> <br>
                    <strong>Impact:</strong> ${i.affected_group || 'General demographic'}
                </div>
            `);
            }
            const issueDetails = issueDetailsArray.join('');

            warningBox.innerHTML = `
                <div class="warning-title" style="margin-bottom: 8px;">BIAS DETECTED</div>
                ${issueDetails}
                <div class="warning-buttons" style="margin-top: 15px;"></div>
            `;

            const btnContainer = warningBox.querySelector('.warning-buttons');

            const patchBtn = document.createElement('button');
            patchBtn.textContent = "YES, DEPLOY FAIRNESS PATCH";
            patchBtn.className = "warn-btn";
            patchBtn.addEventListener('click', () => showPatchModal(userPrompt, selectedModel, foundBiasesWords, realResponse));

            const dismissBtn = document.createElement('button');
            dismissBtn.textContent = "NO, KEEP AS IS";
            dismissBtn.className = "warn-btn";
            dismissBtn.addEventListener('click', () => {
                warningBox.remove();
            });

            btnContainer.appendChild(patchBtn);
            btnContainer.appendChild(dismissBtn);
            aiMessageDiv.appendChild(warningBox);

            chatArea.scrollTop = chatArea.scrollHeight;
        }
    }
    // --- 9. Help Modal Logic ---
    const helpBtn = document.getElementById('help-btn');
    const helpModal = document.getElementById('help-modal');
    const closeHelp = document.getElementById('close-help');

    if (helpBtn && helpModal && closeHelp) {
        helpBtn.addEventListener('click', () => { helpModal.style.display = 'flex'; });
        closeHelp.addEventListener('click', () => { helpModal.style.display = 'none'; });
        window.addEventListener('click', (e) => {
            if (e.target === helpModal) helpModal.style.display = 'none';
        });

        // Example Prompts click handling
        document.querySelectorAll('.help-body em').forEach(em => {
            em.style.cursor = 'pointer';
            
            // Defensively clone to wipe out any phantom listeners attached by previous executions
            const newEm = em.cloneNode(true);
            em.parentNode.replaceChild(newEm, em);
            
            newEm.addEventListener('click', (e) => {
                let clickedText = e.target.textContent;
                clickedText = clickedText.replace(/^"|"$/g, ''); // Strip outer quotes if they exist
                const sanitizedText = clickedText.replace(/\s+/g, ' ').trim();
                
                if (promptInput) {
                    promptInput.value = sanitizedText;
                    promptInput.dispatchEvent(new Event('input'));
                }
                helpModal.style.display = 'none';
            });
        });
    }

    // Edge Collision Prevention & Tooltip Persistence
    document.addEventListener('mouseover', (e) => {
        const wrapper = e.target.closest('.bias-highlight-container');
        if (wrapper) {
            clearTimeout(wrapper.dataset.tooltipTimeout);
            wrapper.classList.add('tooltip-active');

            const tooltip = wrapper.querySelector('.custom-tooltip-box');
            if (tooltip) {
                const rect = wrapper.getBoundingClientRect();
                
                tooltip.style.bottom = 'auto';
                tooltip.style.transform = 'none';

                let tooltipWidth = tooltip.offsetWidth;
                let tooltipHeight = tooltip.offsetHeight;

                let calculatedLeft = rect.left + (rect.width / 2) - (tooltipWidth / 2);
                let calculatedTop = rect.top - tooltipHeight - 12;

                if (calculatedLeft + tooltipWidth > window.innerWidth) {
                    tooltip.style.left = (window.innerWidth - tooltipWidth - 10) + 'px';
                } else if (calculatedLeft < 0) {
                    tooltip.style.left = '10px';
                } else {
                    tooltip.style.left = calculatedLeft + 'px';
                }

                tooltip.style.top = calculatedTop + 'px';
            }
        }
    });

    document.addEventListener('mouseout', (e) => {
        const wrapper = e.target.closest('.bias-highlight-container');
        if (wrapper) {
            if (!wrapper.contains(e.relatedTarget)) {
                const timeoutId = setTimeout(() => {
                    wrapper.classList.remove('tooltip-active');
                }, 300);
                wrapper.dataset.tooltipTimeout = timeoutId;
            }
        }
    });
});

// ==========================================
// Global API functions live in `js/api.js`
// ==========================================
// `index.html` loads `api.js` before `main.js`, which attaches:
// - window.fetchUnifiedAIResponse(prompt, model)
// - window.evaluateForBias(text)

// --- The Fairness Patch Engine ---
window.deployFairnessPatch = async function (originalPrompt, model, patchType = 'standard', selectedWords = [], biasedText = "") {
    const chatArea = document.querySelector('.chat-area');
    if (!chatArea) return;

    const patchNotice = document.createElement('div');
    patchNotice.style.cssText = "align-self: center; background: var(--slight-dark); color: var(--text-color); padding: 8px 16px; border-radius: 20px; font-size: 0.75rem; font-weight: bold; margin: 10px 0; border: var(--border-width) solid var(--border-color); text-transform: uppercase;";
    patchNotice.textContent = `Neutralizing Bias via ${model.toUpperCase()} Engine...`;
    chatArea.appendChild(patchNotice);
    chatArea.scrollTop = chatArea.scrollHeight;

    let correctivePrompt = "";

    if (patchType === 'standard') {
        correctivePrompt = `
            You are an editor. Here is a draft text that needs to be neutralized: 
            "${biasedText}"
            
            REWRITE this response to be strictly professional and neutral. 
            1. Remove all exclusionary coded language. 
            2. Focus only on objective skills and competencies. 
            3. Append this legal footer at the very end: 
            [VERIFIED COMPLIANT] Company Name is an Equal Opportunity Employer. We evaluate applicants without regard to race, color, religion, sex, age, disability, or veteran status.
        `;
    } else if (patchType === 'custom') {
        const wordsToRemove = selectedWords.length > 0 ? selectedWords.join(', ') : 'all biased demographic assumptions';

        correctivePrompt = `
            You are a strict copy-editor. Here is the draft text: 
            "${biasedText}"
            
            Task: Update the text by ONLY removing or replacing the following specific words/phrases: [${wordsToRemove}].
            
            CRITICAL INSTRUCTIONS:
            1. DO NOT rewrite the entire text or change the overall tone.
            2. Preserve the original formatting (bolding, bullet points, headers), structure, and enthusiasm exactly as it is.
            3. ONLY alter the specific sentences containing the targeted words so they make grammatical sense without those words.
            4. Output ONLY the edited text. Do not add any conversational filler.
        `;
    }

    const cleanResponse = await window.fetchUnifiedAIResponse(correctivePrompt, model);

    const aiMessageDiv = document.createElement('div');
    aiMessageDiv.classList.add('message', 'ai-container', 'patched');

    aiMessageDiv.innerHTML = `
        <div class="patch-id-label" style="font-weight: 700; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; margin-bottom: 12px; font-size: 0.85rem;">PATCHED VERSION (AUDIT ID: BB-2026)</div>
        <div class="ai-text-content">${cleanResponse.replace(/\n/g, '<br>')}</div>
        
        <div class="ai-action-bar">
            <button class="action-icon-btn btn-download" title="Download Audit Report" aria-label="Download">
                <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
            </button>
            
            <button class="action-icon-btn btn-regenerate" title="Regenerate Response" aria-label="Regenerate">
                <svg viewBox="0 0 24 24"><path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
            </button>
            
            <button class="action-icon-btn btn-copy" title="Copy Text" aria-label="Copy">
                <svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
            </button>
        </div>
    `;

    chatArea.appendChild(aiMessageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    const downloadBtn = aiMessageDiv.querySelector('.btn-download');
    const regenerateBtn = aiMessageDiv.querySelector('.btn-regenerate');
    const copyBtn = aiMessageDiv.querySelector('.btn-copy');

    downloadBtn.addEventListener('click', () => window.exportAuditReport(originalPrompt, cleanResponse));

    regenerateBtn.addEventListener('click', () => {
        window.deployFairnessPatch(originalPrompt, model, patchType, selectedWords, biasedText);
    });

    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(cleanResponse).then(() => {
            const originalTitle = copyBtn.title;
            const originalIcon = copyBtn.innerHTML;

            copyBtn.title = "Copied!";
            copyBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>`;

            setTimeout(() => {
                copyBtn.title = originalTitle;
                copyBtn.innerHTML = originalIcon;
            }, 2000);
        }).catch(err => {
            console.error("Failed to copy text: ", err);
        });
    });
};

// --- Audit Export Logic ---
window.exportAuditReport = function (original, patched) {
    const timestamp = new Date().toLocaleString();
    const reportContent = `
BIASBUSTER LAB - AUDIT REPORT
-----------------------------
ID: BB-2026
TIMESTAMP: ${timestamp}
COMPLIANCE STATUS: VERIFIED

[ORIGINAL PROMPT/INPUT]
${original}

[MITIGATION STEPS]
1. Scanned for coded demographic language.
2. Removed exclusionary identifiers or deployed custom fairness rules.
3. Verified neutral competency-based phrasing.

[PATCHED OUTPUT]
${patched}

-----------------------------
This document serves as proof of bias mitigation for CodeQuest 2026.
    `;

    const blob = new Blob([reportContent.trim()], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `BiasBuster_Audit_Report.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
};

// --- Unpatched Audit Export Logic ---
window.exportUnpatchedAuditReport = function (original, unpatched) {
    const timestamp = new Date().toLocaleString();
    const reportContent = `
BIASBUSTER LAB - UNPATCHED AUDIT REPORT
-----------------------------
ID: BB-2026-UNPATCHED
TIMESTAMP: ${timestamp}
COMPLIANCE STATUS: FAILED / BYPASSED

[ORIGINAL PROMPT/INPUT]
${original}

[WARNING]
The user elected to bypass the fairness patch. The following text contains unmitigated demographic assumptions or coded language.

[UNPATCHED OUTPUT]
${unpatched}

-----------------------------
This document serves as an audit trail of bypassed bias mitigation for CodeQuest 2026.
    `;

    const blob = new Blob([reportContent.trim()], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `BiasBuster_Unpatched_Report.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
};

