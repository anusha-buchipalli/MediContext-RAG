/**
 * ==============================================================================
 * AURA CLINICAL AI - ENTERPRISE CLIENT RUNTIME (VANILLA JAVASCRIPT)
 * ==============================================================================
 * Architecture:
 * 1. State Management (Audience persona, state array, session metrics)
 * 2. WebAudio Micro-Recording Fragment Pipeline (Groq Whisper v3)
 * 3. Clinical Inference Dispatch Pipeline (/ask)
 * 4. Upstream Emergency Triage Engine Rendering (High-Contrast Alert Cards)
 * 5. Posterior Hallucination & Compliance Refusal Handlers
 * 6. Intent Routing & Source Citations Drawer Telemetry
 * 7. Clinical Session Export Utility (Physician Handoff Log)
 * 8. PDF Knowledge Ingestion Pipeline
 * ==============================================================================
 */

(function () {
    "use strict";

    // ==========================================================================
    // 1. APPLICATION STATE
    // ==========================================================================

    const state = {
        audience: "Patient", // 'Patient' or 'Clinician'
        guardrailMode: "Calibrated", // 'Calibrated' or 'Strict'
        history: [], // Consultation state array containing all turns
        isRecording: false,
        mediaRecorder: null,
        audioChunks: [],
        recordingTimerInterval: null,
        recordingSeconds: 0,
        selectedPdfFile: null,
        sessionMetrics: {
            inquiries: 0,
            internalRag: 0,
            externalDdg: 0,
            emergencyTriaged: 0
        }
    };

    // ==========================================================================
    // 2. DOM ELEMENT CACHE
    // ==========================================================================

    const DOM = {
        // Mode switchers
        modePatientBtn: document.getElementById("mode-patient"),
        modeClinicianBtn: document.getElementById("mode-clinician"),
        guardrailCalibratedBtn: document.getElementById("guardrail-calibrated"),
        guardrailStrictBtn: document.getElementById("guardrail-strict"),

        // Chat stream & inputs
        streamContainer: document.getElementById("stream-container"),
        welcomeCard: document.getElementById("welcome-card"),
        messagesList: document.getElementById("messages-list"),
        chatForm: document.getElementById("chat-form"),
        userInput: document.getElementById("user-input"),
        btnSend: document.getElementById("btn-send"),
        btnClearChat: document.getElementById("btn-clear-chat"),

        // Voice micro-recording
        btnMicToggle: document.getElementById("btn-mic-toggle"),
        voiceRecordingBar: document.getElementById("voice-recording-bar"),
        recordingTimer: document.getElementById("recording-timer"),
        btnStopRecording: document.getElementById("btn-stop-recording"),
        btnCancelRecording: document.getElementById("btn-cancel-recording"),

        // Sidebar Telemetry & Citations
        sidebarRouteBadge: document.getElementById("sidebar-route-badge"),
        stepRouterStatus: document.getElementById("step-router-status"),
        branchInternal: document.getElementById("branch-internal"),
        branchExternal: document.getElementById("branch-external"),
        citationsSidebarCard: document.getElementById("citations-sidebar-card"),
        sidebarCitationsBody: document.getElementById("sidebar-citations-body"),
        citationsCountPill: document.getElementById("citations-count-pill"),

        // Metrics counters
        statQueryCount: document.getElementById("stat-query-count"),
        statRouteInternal: document.getElementById("stat-route-internal"),
        statRouteExternal: document.getElementById("stat-route-external"),
        statTriageCount: document.getElementById("stat-triage-count"),

        // Telemetry top bar
        statusSystemVal: document.getElementById("status-system-val"),
        statusPineconeVal: document.getElementById("status-pinecone-val"),
        statusGroqVal: document.getElementById("status-groq-val"),
        kbChunksVal: document.getElementById("kb-chunks-val"),
        kbNamespaceVal: document.getElementById("kb-namespace-val"),

        // Modals
        btnExportLog: document.getElementById("btn-export-log"),
        modalExport: document.getElementById("modal-export"),
        btnCloseExportModal: document.getElementById("btn-close-export-modal"),
        exportPreviewPane: document.getElementById("export-preview-pane"),
        btnCopyExport: document.getElementById("btn-copy-export"),
        btnDownloadJson: document.getElementById("btn-download-json"),
        btnDownloadTxt: document.getElementById("btn-download-txt"),

        btnUploadPdf: document.getElementById("btn-upload-pdf"),
        modalUpload: document.getElementById("modal-upload"),
        btnCloseUploadModal: document.getElementById("btn-close-upload-modal"),
        pdfDropZone: document.getElementById("pdf-drop-zone"),
        pdfFileInput: document.getElementById("pdf-file-input"),
        filePreviewBar: document.getElementById("file-preview-bar"),
        previewFilename: document.getElementById("preview-filename"),
        previewFilesize: document.getElementById("preview-filesize"),
        uploadProgressBox: document.getElementById("upload-progress-box"),
        uploadProgressBar: document.getElementById("upload-progress-bar"),
        uploadProgressText: document.getElementById("upload-progress-text"),
        btnSubmitUpload: document.getElementById("btn-submit-upload"),
        btnCancelUpload: document.getElementById("btn-cancel-upload")
    };

    // ==========================================================================
    // 3. INITIALIZATION & TELEMETRY SYNC
    // ==========================================================================

    function init() {
        bindEvents();
        fetchSystemStatus();
        autoResizeTextarea(DOM.userInput);
    }

    function fetchSystemStatus() {
        fetch("/api/status")
            .then(res => res.json())
            .then(data => {
                if (data.online) {
                    if (DOM.statusSystemVal) DOM.statusSystemVal.textContent = "ONLINE";
                    if (DOM.statusPineconeVal && data.total_vectors !== undefined) {
                        DOM.statusPineconeVal.textContent = `${(data.total_vectors / 1000).toFixed(1)}k VECTORS`;
                    }
                    if (DOM.kbChunksVal && data.total_vectors !== undefined) {
                        DOM.kbChunksVal.textContent = Number(data.total_vectors).toLocaleString();
                    }
                    if (DOM.kbNamespaceVal && data.pinecone_namespace) {
                        DOM.kbNamespaceVal.textContent = data.pinecone_namespace;
                    }
                    if (DOM.statusGroqVal && data.groq_chat_model) {
                        DOM.statusGroqVal.textContent = "GROQ LPU";
                    }
                }
            })
            .catch(err => {
                console.warn("[TELEMETRY WARNING] Could not fetch server telemetry status:", err);
            });
    }

    // ==========================================================================
    // 4. EVENT BINDING
    // ==========================================================================

    function bindEvents() {
        // Audience modifier switchers
        DOM.modePatientBtn.addEventListener("click", () => setAudienceMode("Patient"));
        DOM.modeClinicianBtn.addEventListener("click", () => setAudienceMode("Clinician"));

        // Guardrail compliance mode switchers
        if (DOM.guardrailCalibratedBtn) {
            DOM.guardrailCalibratedBtn.addEventListener("click", () => setGuardrailMode("Calibrated"));
        }
        if (DOM.guardrailStrictBtn) {
            DOM.guardrailStrictBtn.addEventListener("click", () => setGuardrailMode("Strict"));
        }

        // Chat form submit
        DOM.chatForm.addEventListener("submit", handleChatSubmit);

        // Input keyboard handling (Enter to submit, Shift+Enter for newline)
        DOM.userInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                DOM.chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
            }
        });

        DOM.userInput.addEventListener("input", () => {
            autoResizeTextarea(DOM.userInput);
        });

        // Quick suggestion chips
        document.querySelectorAll(".query-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                const query = chip.getAttribute("data-query");
                if (query) {
                    DOM.userInput.value = query;
                    autoResizeTextarea(DOM.userInput);
                    DOM.userInput.focus();
                }
            });
        });

        // Voice micro-recording
        DOM.btnMicToggle.addEventListener("click", toggleVoiceRecording);
        DOM.btnStopRecording.addEventListener("click", stopAndTranscribeAudio);
        DOM.btnCancelRecording.addEventListener("click", cancelVoiceRecording);

        // Clear chat
        DOM.btnClearChat.addEventListener("click", clearConsultation);

        // Export modal actions
        DOM.btnExportLog.addEventListener("click", openExportModal);
        DOM.btnCloseExportModal.addEventListener("click", () => DOM.modalExport.style.display = "none");
        DOM.btnCopyExport.addEventListener("click", copyExportToClipboard);
        DOM.btnDownloadTxt.addEventListener("click", downloadExportTxt);
        DOM.btnDownloadJson.addEventListener("click", downloadExportJson);

        // PDF upload modal actions
        DOM.btnUploadPdf.addEventListener("click", () => DOM.modalUpload.style.display = "flex");
        DOM.btnCloseUploadModal.addEventListener("click", () => DOM.modalUpload.style.display = "none");
        DOM.btnCancelUpload.addEventListener("click", () => DOM.modalUpload.style.display = "none");

        // Drag & drop handlers
        DOM.pdfDropZone.addEventListener("click", () => DOM.pdfFileInput.click());
        DOM.pdfFileInput.addEventListener("change", handleFileSelect);
        DOM.pdfDropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            DOM.pdfDropZone.classList.add("dragover");
        });
        DOM.pdfDropZone.addEventListener("dragleave", () => {
            DOM.pdfDropZone.classList.remove("dragover");
        });
        DOM.pdfDropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            DOM.pdfDropZone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                processSelectedFile(e.dataTransfer.files[0]);
            }
        });
        DOM.btnSubmitUpload.addEventListener("click", uploadPdfDocument);

        // Close modals on background click
        window.addEventListener("click", (e) => {
            if (e.target === DOM.modalExport) DOM.modalExport.style.display = "none";
            if (e.target === DOM.modalUpload) DOM.modalUpload.style.display = "none";
        });
    }

    // ==========================================================================
    // 5. AUDIENCE MODE CONTROLLER (PATIENT VS CLINICIAN)
    // ==========================================================================

    function setAudienceMode(mode) {
        state.audience = mode;
        if (mode === "Patient") {
            DOM.modePatientBtn.classList.add("active");
            DOM.modePatientBtn.setAttribute("aria-selected", "true");
            DOM.modeClinicianBtn.classList.remove("active");
            DOM.modeClinicianBtn.setAttribute("aria-selected", "false");
            DOM.userInput.placeholder = "Ask any health, symptom, or uploaded document question in plain language...";
        } else {
            DOM.modeClinicianBtn.classList.add("active");
            DOM.modeClinicianBtn.setAttribute("aria-selected", "true");
            DOM.modePatientBtn.classList.remove("active");
            DOM.modePatientBtn.setAttribute("aria-selected", "false");
            DOM.userInput.placeholder = "Enter clinical differential, diagnostic pathology, ICD metric, or lab query...";
        }
    }

    function setGuardrailMode(mode) {
        state.guardrailMode = mode;
        if (mode === "Calibrated") {
            if (DOM.guardrailCalibratedBtn) {
                DOM.guardrailCalibratedBtn.classList.add("active");
                DOM.guardrailCalibratedBtn.setAttribute("aria-selected", "true");
            }
            if (DOM.guardrailStrictBtn) {
                DOM.guardrailStrictBtn.classList.remove("active");
                DOM.guardrailStrictBtn.setAttribute("aria-selected", "false");
            }
        } else {
            if (DOM.guardrailStrictBtn) {
                DOM.guardrailStrictBtn.classList.add("active");
                DOM.guardrailStrictBtn.setAttribute("aria-selected", "true");
            }
            if (DOM.guardrailCalibratedBtn) {
                DOM.guardrailCalibratedBtn.classList.remove("active");
                DOM.guardrailCalibratedBtn.setAttribute("aria-selected", "false");
            }
        }
    }

    // ==========================================================================
    // 6. WEBAUDIO MICRO-RECORDING & GROQ WHISPER PIPELINE
    // ==========================================================================

    async function toggleVoiceRecording() {
        if (state.isRecording) {
            stopAndTranscribeAudio();
        } else {
            startVoiceRecording();
        }
    }

    async function startVoiceRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Microphone recording is not supported in this browser or environment.");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            state.audioChunks = [];

            // Determine supported mimeType
            let mimeType = "audio/webm";
            if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
                mimeType = "audio/webm;codecs=opus";
            } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
                mimeType = "audio/ogg;codecs=opus";
            } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
                mimeType = "audio/mp4";
            }

            state.mediaRecorder = new MediaRecorder(stream, { mimeType });

            state.mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    state.audioChunks.push(e.data);
                }
            };

            state.mediaRecorder.onstop = () => {
                stream.getTracks().forEach(track => track.stop());
            };

            state.mediaRecorder.start(250); // Slice into 250ms chunks
            state.isRecording = true;
            state.recordingSeconds = 0;

            DOM.btnMicToggle.classList.add("recording");
            DOM.voiceRecordingBar.classList.add("active");
            DOM.recordingTimer.textContent = "00:00";

            state.recordingTimerInterval = setInterval(() => {
                state.recordingSeconds++;
                const mins = String(Math.floor(state.recordingSeconds / 60)).padStart(2, "0");
                const secs = String(state.recordingSeconds % 60).padStart(2, "0");
                DOM.recordingTimer.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            console.error("[VOICE RECORDING ERROR]", err);
            alert(`Could not access microphone: ${err.message || err.name}. Please verify microphone permissions.`);
            resetRecordingState();
        }
    }

    function stopAndTranscribeAudio() {
        if (!state.isRecording || !state.mediaRecorder) return;

        clearInterval(state.recordingTimerInterval);
        state.isRecording = false;
        DOM.btnMicToggle.classList.remove("recording");

        DOM.recordingTimer.textContent = "TRANSCRIBING...";
        DOM.btnStopRecording.disabled = true;

        state.mediaRecorder.onstop = async () => {
            try {
                const audioBlob = new Blob(state.audioChunks, { type: state.mediaRecorder.mimeType || "audio/webm" });
                const formData = new FormData();
                formData.append("audio", audioBlob, "consultation_recording.webm");

                const response = await fetch("/api/voice-transcribe", {
                    method: "POST",
                    body: formData
                });

                const result = await response.json();

                if (result.success && result.transcription) {
                    const currentVal = DOM.userInput.value.trim();
                    DOM.userInput.value = currentVal ? `${currentVal} ${result.transcription}` : result.transcription;
                    autoResizeTextarea(DOM.userInput);
                    DOM.userInput.focus();
                } else {
                    alert(result.error || "Whisper was unable to discern medical speech from the recording.");
                }
            } catch (err) {
                console.error("[TRANSCRIBE ERROR]", err);
                alert(`Voice transcription failed: ${err.message}`);
            } finally {
                resetRecordingState();
            }
        };

        state.mediaRecorder.stop();
    }

    function cancelVoiceRecording() {
        if (state.mediaRecorder && state.isRecording) {
            state.mediaRecorder.onstop = null;
            state.mediaRecorder.stop();
        }
        resetRecordingState();
    }

    function resetRecordingState() {
        clearInterval(state.recordingTimerInterval);
        state.isRecording = false;
        state.audioChunks = [];
        state.recordingSeconds = 0;
        DOM.btnMicToggle.classList.remove("recording");
        DOM.voiceRecordingBar.classList.remove("active");
        DOM.btnStopRecording.disabled = false;
    }

    // ==========================================================================
    // 7. INFERENCE DISPATCH PIPELINE (/ask)
    // ==========================================================================

    async function handleChatSubmit(e) {
        e.preventDefault();
        const question = DOM.userInput.value.trim();
        if (!question) return;

        // Clear input and collapse welcome card
        DOM.userInput.value = "";
        autoResizeTextarea(DOM.userInput);
        if (DOM.welcomeCard) {
            DOM.welcomeCard.style.display = "none";
        }

        // Render User Inquiry Card
        const turnTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        renderUserMessage(question, turnTimestamp, state.audience);

        // Render Loading Placeholder
        const loadingId = `loading-${Date.now()}`;
        renderLoadingIndicator(loadingId);
        scrollToBottom();

        // Update UI state
        DOM.btnSend.disabled = true;
        updateSidebarPipelinePending();

        try {
            const response = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    audience: state.audience,
                    guardrail_mode: state.guardrailMode
                })
            });

            removeLoadingIndicator(loadingId);
            const data = await response.json();

            if (!response.ok && !data.triage_emergency) {
                throw new Error(data.error || "Server returned an inference evaluation error.");
            }

            // Check if Upstream Emergency Triage Engine triggered
            if (data.triage_emergency || data.status === "emergency_intercepted") {
                handleEmergencyAlert(data, question);
            } else {
                handleSuccessfulConsultation(data, question);
            }

        } catch (err) {
            console.error("[INFERENCE ERROR]", err);
            removeLoadingIndicator(loadingId);
            renderErrorMessage(err.message);
        } finally {
            DOM.btnSend.disabled = false;
            scrollToBottom();
        }
    }

    // ==========================================================================
    // 8. EMERGENCY TRIAGE & AI CARD RENDERING
    // ==========================================================================

    function handleEmergencyAlert(data, question) {
        state.sessionMetrics.emergencyTriaged++;
        state.sessionMetrics.inquiries++;
        updateSessionCounters();

        const triageData = data.triage_data || {};
        const matchedTerm = triageData.matched_term || "Acute Critical Red Flag";
        const alertTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Record in consultation history state array
        state.history.push({
            id: `turn-${Date.now()}`,
            timestamp: new Date().toISOString(),
            displayTime: alertTime,
            question: question,
            answer: data.answer,
            audience: state.audience,
            route: "EMERGENCY_TRIAGE_HALT",
            sources: [],
            triageEmergency: true,
            matchedTerm: matchedTerm,
            complianceVerified: true
        });

        // Render High-Contrast Emergency Card
        const emergencyCardHtml = `
            <div class="emergency-alert-card">
                <div class="emergency-card-header">
                    <div class="siren-icon-box">🚨</div>
                    <div class="emergency-title-block">
                        <h3>CRITICAL EMERGENCY TRIAGE ALERT</h3>
                        <p>Potential Life-Threatening Symptom Detected: <span class="emergency-matched-tag">${escapeHtml(matchedTerm.toUpperCase())}</span></p>
                    </div>
                </div>
                <div class="emergency-body">
                    <p><strong>Clinical Warning:</strong> ${escapeHtml(data.answer)}</p>
                </div>
                <div class="emergency-hotlines-grid">
                    <a href="tel:911" class="hotline-btn-card">
                        <span class="hotline-number">911 / 112 / 999</span>
                        <span class="hotline-label">Emergency Medical Response</span>
                    </a>
                    <a href="tel:988" class="hotline-btn-card">
                        <span class="hotline-number">988</span>
                        <span class="hotline-label">Suicide & Crisis Lifeline</span>
                    </a>
                    <a href="tel:18002221222" class="hotline-btn-card">
                        <span class="hotline-number">1-800-222-1222</span>
                        <span class="hotline-label">National Poison Control</span>
                    </a>
                </div>
                <div class="emergency-directive">
                    ⚠️ <strong>Immediate Directive:</strong> Artificial intelligence cannot diagnose, monitor, or treat emergency pathologies. Please cease conversational input and contact an emergency medical provider immediately.
                </div>
            </div>
        `;

        DOM.messagesList.insertAdjacentHTML("beforeend", emergencyCardHtml);

        // Update sidebar diagram
        if (DOM.sidebarRouteBadge) {
            DOM.sidebarRouteBadge.textContent = "EMERGENCY HALT";
            DOM.sidebarRouteBadge.style.color = "var(--red-emergency)";
        }
        if (DOM.stepRouterStatus) {
            DOM.stepRouterStatus.textContent = "INTERCEPTED";
        }
    }

    function handleSuccessfulConsultation(data, question) {
        state.sessionMetrics.inquiries++;
        if (data.route === "INTERNAL") {
            state.sessionMetrics.internalRag++;
        } else {
            state.sessionMetrics.externalDdg++;
        }
        updateSessionCounters();

        const turnTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const turnId = `turn-${Date.now()}`;

        // Save in consultation history state array
        state.history.push({
            id: turnId,
            timestamp: new Date().toISOString(),
            displayTime: turnTime,
            question: question,
            answer: data.answer,
            audience: data.audience || state.audience,
            guardrailMode: data.guardrail_mode || state.guardrailMode,
            route: data.route,
            sources: data.sources || [],
            triageEmergency: false,
            complianceVerified: data.compliance_verified !== false,
            hallucinationDetected: Boolean(data.hallucination_detected)
        });

        // Render AI Message Card
        renderAiMessage(data, turnTime, turnId);

        // Update Sidebar Intelligence & Citations
        updateSidebarTelemetry(data);
    }

    function renderUserMessage(text, time, audience) {
        const userCardHtml = `
            <div class="message-card-user">
                <div class="user-card-meta">
                    <span class="user-tag">[${escapeHtml(audience.toUpperCase())} INQUIRY]</span>
                    <span class="user-time">${time}</span>
                </div>
                <div class="user-text">${escapeHtml(text)}</div>
            </div>
        `;
        DOM.messagesList.insertAdjacentHTML("beforeend", userCardHtml);
    }

    function renderAiMessage(data, time, turnId) {
        const isInternal = data.route === "INTERNAL";
        const routeLabel = isInternal ? "INTERNAL RAG: PATIENT RECORD" : "EXTERNAL VERIFIED: NIH / WHO / CDC";
        const routeClass = isInternal ? "route-internal" : "route-external";
        const routeIcon = isInternal ? "📑" : "🌐";

        const compliancePassed = data.compliance_verified !== false && !data.hallucination_detected;
        const complianceBadgeHtml = compliancePassed
            ? `<span class="badge-compliance compliance-pass">🛡️ GROUNDED IN CONTEXT</span>`
            : `<span class="badge-compliance compliance-fail">⚠️ COMPLIANCE REFUSAL</span>`;

        const activeGuardrail = data.guardrail_mode || state.guardrailMode || "Calibrated";
        const guardrailBadgeHtml = activeGuardrail === "Calibrated"
            ? `<span class="badge-compliance badge-guardrail">💡 CALIBRATED (EDUCATIONAL)</span>`
            : `<span class="badge-compliance badge-guardrail">🛡️ STRICT (VERBATIM)</span>`;

        const sources = data.sources || [];
        const hasSources = sources.length > 0;
        const drawerId = `drawer-${turnId}`;

        // Format Markdown to clean HTML
        const formattedAnswer = formatMarkdownToHtml(data.answer);

        let sourcesDrawerHtml = "";
        if (hasSources) {
            const sourceItems = sources.map((s, idx) => {
                const title = s.title || s.source || `Clinical Excerpt #${idx + 1}`;
                const meta = s.page ? `Page ${s.page} | Score ${s.score || 'N/A'}` : (s.url ? `<a href="${s.url}" target="_blank" rel="noopener">Verified Link</a>` : "Document Archive");
                const snippet = s.excerpt || s.snippet || "";
                return `
                    <div class="inline-source-item">
                        <div class="source-item-header">
                            <span class="source-title">${escapeHtml(title)}</span>
                            <span class="source-meta">${meta}</span>
                        </div>
                        <div class="source-snippet">"${escapeHtml(snippet)}"</div>
                    </div>
                `;
            }).join("");

            sourcesDrawerHtml = `
                <div class="ai-card-footer">
                    <button type="button" class="btn-toggle-sources" onclick="window.AuraApp.toggleSourcesDrawer('${drawerId}')">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                        <span>View Authenticated Citations (${sources.length})</span>
                    </button>
                    <span class="ai-card-time">${time}</span>
                </div>
                <div class="inline-sources-drawer" id="${drawerId}">
                    ${sourceItems}
                </div>
            `;
        } else {
            sourcesDrawerHtml = `
                <div class="ai-card-footer">
                    <span class="source-meta">General Clinical Reasoning</span>
                    <span class="ai-card-time">${time}</span>
                </div>
            `;
        }

        const aiCardHtml = `
            <div class="message-card-ai" id="${turnId}">
                <div class="ai-card-header">
                    <div class="ai-meta-badges">
                        <span class="badge-route ${routeClass}">${routeIcon} ${routeLabel}</span>
                        <span class="badge-audience">AUDIENCE: ${escapeHtml((data.audience || state.audience).toUpperCase())}</span>
                        ${guardrailBadgeHtml}
                        ${complianceBadgeHtml}
                    </div>
                </div>
                <div class="ai-card-body">
                    ${formattedAnswer}
                </div>
                ${sourcesDrawerHtml}
            </div>
        `;

        DOM.messagesList.insertAdjacentHTML("beforeend", aiCardHtml);
    }

    function renderLoadingIndicator(id) {
        const loadingHtml = `
            <div class="message-card-ai" id="${id}" style="opacity: 0.85;">
                <div class="ai-card-header">
                    <div class="ai-meta-badges">
                        <span class="badge-route route-internal">⚡ CLINICAL INFERENCE RUNNING...</span>
                    </div>
                </div>
                <div class="ai-card-body" style="display: flex; align-items: center; gap: 12px; padding: 18px 22px;">
                    <span class="indicator-dot dot-cyan" style="animation: pulse-ring 1s infinite;"></span>
                    <span style="color: var(--text-secondary); font-size: 13px;">Analyzing clinical inquiry across triage rules, intent router, and reference context...</span>
                </div>
            </div>
        `;
        DOM.messagesList.insertAdjacentHTML("beforeend", loadingHtml);
    }

    function removeLoadingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function renderErrorMessage(msg) {
        const errorHtml = `
            <div class="compliance-refusal-card">
                <h4>⚠️ Clinical Inference Exception</h4>
                <p>${escapeHtml(msg)}</p>
                <small style="display: block; margin-top: 6px; color: var(--text-muted);">Please check server connectivity or verify that your API credentials in .env are active.</small>
            </div>
        `;
        DOM.messagesList.insertAdjacentHTML("beforeend", errorHtml);
    }

    // ==========================================================================
    // 9. SIDEBAR TELEMETRY & CITATIONS SYNC
    // ==========================================================================

    function updateSidebarPipelinePending() {
        if (DOM.sidebarRouteBadge) DOM.sidebarRouteBadge.textContent = "EVALUATING...";
        if (DOM.stepRouterStatus) DOM.stepRouterStatus.textContent = "ROUTING...";
        if (DOM.branchInternal) DOM.branchInternal.classList.remove("active");
        if (DOM.branchExternal) DOM.branchExternal.classList.remove("active");
    }

    function updateSidebarTelemetry(data) {
        const isInternal = data.route === "INTERNAL";

        if (DOM.sidebarRouteBadge) {
            DOM.sidebarRouteBadge.textContent = data.route;
            DOM.sidebarRouteBadge.style.color = isInternal ? "var(--cyan-primary)" : "#c084fc";
        }
        if (DOM.stepRouterStatus) {
            DOM.stepRouterStatus.textContent = "CLASSIFIED";
        }

        if (DOM.branchInternal && DOM.branchExternal) {
            if (isInternal) {
                DOM.branchInternal.classList.add("active");
                DOM.branchExternal.classList.remove("active");
            } else {
                DOM.branchExternal.classList.add("active");
                DOM.branchInternal.classList.remove("active");
            }
        }

        // Render in Citations Drawer
        renderSidebarCitations(data.sources || []);
    }

    function renderSidebarCitations(sources) {
        if (!DOM.sidebarCitationsBody) return;

        if (!sources || sources.length === 0) {
            DOM.citationsCountPill.textContent = "0 Sources";
            DOM.sidebarCitationsBody.innerHTML = `
                <div class="citations-empty-state">
                    <p>No external citations retrieved for this response.</p>
                </div>
            `;
            return;
        }

        DOM.citationsCountPill.textContent = `${sources.length} Sources`;
        DOM.sidebarCitationsBody.innerHTML = sources.map((s, idx) => {
            const title = s.title || s.source || `Citation #${idx + 1}`;
            const linkHtml = s.url ? `<a href="${s.url}" target="_blank" rel="noopener">${escapeHtml(title)}</a>` : `<strong>${escapeHtml(title)}</strong>`;
            const snippet = s.excerpt || s.snippet || "";
            return `
                <div class="citation-card-side">
                    ${linkHtml}
                    <div class="citation-side-text">"${escapeHtml(snippet.slice(0, 160))}..."</div>
                </div>
            `;
        }).join("");
    }

    function updateSessionCounters() {
        if (DOM.statQueryCount) DOM.statQueryCount.textContent = state.sessionMetrics.inquiries;
        if (DOM.statRouteInternal) DOM.statRouteInternal.textContent = state.sessionMetrics.internalRag;
        if (DOM.statRouteExternal) DOM.statRouteExternal.textContent = state.sessionMetrics.externalDdg;
        if (DOM.statTriageCount) DOM.statTriageCount.textContent = state.sessionMetrics.emergencyTriaged;
    }

    // ==========================================================================
    // 10. SESSION EXPORT UTILITY (PHYSICIAN CONSULTATION LOG)
    // ==========================================================================

    function openExportModal() {
        if (state.history.length === 0) {
            alert("No consultation interactions recorded in the current session.");
            return;
        }

        const logContent = generateFormattedClinicalReport();
        DOM.exportPreviewPane.textContent = logContent;
        DOM.modalExport.style.display = "flex";
    }

    function generateFormattedClinicalReport() {
        const now = new Date();
        const divider = "=".repeat(80);
        const subDivider = "-".repeat(80);

        let report = `${divider}\n`;
        report += `AURA CLINICAL AI - MEDICAL CONSULTATION & DECISION SUPPORT AUDIT LOG\n`;
        report += `DOCUMENT GENERATION TIMESTAMP : ${now.toISOString()}\n`;
        report += `ACTIVE AUDIENCE PROFILE       : ${state.audience.toUpperCase()}\n`;
        report += `ACTIVE COMPLIANCE GUARDRAIL   : ${state.guardrailMode.toUpperCase()}\n`;
        report += `TOTAL CONSULTATION TURNS      : ${state.history.length}\n`;
        report += `EMERGENCY TRIAGES ENCOUNTERED : ${state.sessionMetrics.emergencyTriaged}\n`;
        report += `DISCLAIMER: This document is an informational log of patient-AI interactions\n`;
        report += `intended for clinical review by a licensed healthcare professional.\n`;
        report += `${divider}\n\n`;

        state.history.forEach((turn, idx) => {
            report += `[INTERACTION #${idx + 1}] | TIMESTAMP: ${turn.timestamp} | ROUTE: ${turn.route}\n`;
            report += `AUDIENCE PROFILE: ${turn.audience} | GUARDRAIL MODE: ${turn.guardrailMode || state.guardrailMode}\n`;
            report += `PATIENT INQUIRY:\n${turn.question}\n\n`;
            report += `CLINICAL SYNTHESIS:\n${turn.answer}\n\n`;

            if (turn.sources && turn.sources.length > 0) {
                report += `AUTHENTICATED REFERENCE CITATIONS (${turn.sources.length}):\n`;
                turn.sources.forEach((s, sIdx) => {
                    report += `  [${sIdx + 1}] Source: ${s.title || s.source || 'Medical Document'}\n`;
                    if (s.page) report += `      Page/Score: Page ${s.page} (Match score: ${s.score || 'N/A'})\n`;
                    if (s.url) report += `      URL: ${s.url}\n`;
                    if (s.excerpt) report += `      Excerpt: "${s.excerpt.trim()}"\n`;
                });
            } else {
                report += `CITATIONS: Direct clinical reasoning or emergency halt.\n`;
            }

            report += `COMPLIANCE STATUS: ${turn.complianceVerified ? 'VERIFIED GROUNDED' : 'COMPLIANCE REFUSAL'}\n`;
            report += `${subDivider}\n\n`;
        });

        report += `${divider}\n`;
        report += `END OF CLINICAL REPORT - AURA AI HEALTHCARE INTELLIGENCE\n`;
        report += `${divider}\n`;

        return report;
    }

    function copyExportToClipboard() {
        const text = DOM.exportPreviewPane.textContent;
        navigator.clipboard.writeText(text).then(() => {
            DOM.btnCopyExport.textContent = "Copied to Clipboard!";
            setTimeout(() => {
                DOM.btnCopyExport.textContent = "Copy to Clipboard";
            }, 2000);
        }).catch(err => {
            alert(`Could not copy: ${err}`);
        });
    }

    function downloadExportTxt() {
        const text = DOM.exportPreviewPane.textContent;
        const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const filename = `Clinical_Consultation_Log_${new Date().toISOString().slice(0, 10)}.txt`;
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function downloadExportJson() {
        const sessionPayload = {
            exportTimestamp: new Date().toISOString(),
            audienceMode: state.audience,
            guardrailMode: state.guardrailMode,
            metrics: state.sessionMetrics,
            consultationHistory: state.history
        };
        const text = JSON.stringify(sessionPayload, null, 2);
        const blob = new Blob([text], { type: "application/json;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const filename = `Clinical_Session_State_${new Date().toISOString().slice(0, 10)}.json`;
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ==========================================================================
    // 11. PDF KNOWLEDGE BASE INGESTION
    // ==========================================================================

    function handleFileSelect(e) {
        if (e.target.files && e.target.files[0]) {
            processSelectedFile(e.target.files[0]);
        }
    }

    function processSelectedFile(file) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            alert("Only PDF files are supported for clinical document ingestion.");
            return;
        }

        state.selectedPdfFile = file;
        DOM.previewFilename.textContent = file.name;
        DOM.previewFilesize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
        DOM.filePreviewBar.style.display = "flex";
        DOM.btnSubmitUpload.disabled = false;
    }

    async function uploadPdfDocument() {
        if (!state.selectedPdfFile) return;

        const formData = new FormData();
        formData.append("file", state.selectedPdfFile);

        DOM.uploadProgressBox.style.display = "block";
        DOM.uploadProgressBar.style.width = "40%";
        DOM.uploadProgressText.textContent = "Extracting document pages and creating chunks...";
        DOM.btnSubmitUpload.disabled = true;

        try {
            DOM.uploadProgressBar.style.width = "70%";
            DOM.uploadProgressText.textContent = "Computing Hugging Face embeddings & upserting to Pinecone...";

            const response = await fetch("/api/upload-pdf", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Failed to ingest PDF into vector index.");
            }

            DOM.uploadProgressBar.style.width = "100%";
            DOM.uploadProgressText.textContent = `Success! ${data.chunks_upserted} chunks ingested into namespace '${data.namespace}'.`;

            setTimeout(() => {
                DOM.modalUpload.style.display = "none";
                state.selectedPdfFile = null;
                DOM.filePreviewBar.style.display = "none";
                DOM.uploadProgressBox.style.display = "none";
                DOM.uploadProgressBar.style.width = "0%";
                fetchSystemStatus(); // Refresh vector telemetry
            }, 1800);

        } catch (err) {
            console.error("[UPLOAD ERROR]", err);
            DOM.uploadProgressText.textContent = `Error: ${err.message}`;
            DOM.uploadProgressBar.style.backgroundColor = "var(--red-emergency)";
            DOM.btnSubmitUpload.disabled = false;
        }
    }

    // ==========================================================================
    // 12. HELPER UTILITIES
    // ==========================================================================

    function clearConsultation() {
        if (state.history.length === 0) return;
        if (confirm("Are you sure you want to clear the active clinical consultation stream?")) {
            state.history = [];
            DOM.messagesList.innerHTML = "";
            if (DOM.welcomeCard) DOM.welcomeCard.style.display = "block";
            renderSidebarCitations([]);
            updateSidebarPipelinePending();
            if (DOM.sidebarRouteBadge) DOM.sidebarRouteBadge.textContent = "STANDBY";
            if (DOM.stepRouterStatus) DOM.stepRouterStatus.textContent = "READY";
        }
    }

    function autoResizeTextarea(textarea) {
        if (!textarea) return;
        textarea.style.height = "auto";
        textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
    }

    function scrollToBottom() {
        DOM.streamContainer.scrollTop = DOM.streamContainer.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatMarkdownToHtml(markdown) {
        if (!markdown) return "";
        let html = markdown;

        // Escape raw HTML entities
        html = escapeHtml(html);

        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Headers
        html = html.replace(/^### (.*?)$/gm, '<h4>$1</h4>');
        html = html.replace(/^## (.*?)$/gm, '<h3>$1</h3>');
        html = html.replace(/^# (.*?)$/gm, '<h2>$1</h2>');

        // Bullet list items
        html = html.replace(/^\s*[\-\*]\s+(.*?)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>');

        // Numbered lists
        html = html.replace(/^\s*\d+\.\s+(.*?)$/gm, '<li>$1</li>');

        // Paragraphs & newlines
        html = html.split(/\n{2,}/).map(para => {
            para = para.trim();
            if (!para.startsWith('<h') && !para.startsWith('<ul') && !para.startsWith('<ol')) {
                return `<p>${para.replace(/\n/g, '<br>')}</p>`;
            }
            return para;
        }).join("");

        return html;
    }

    // Export global helpers for runtime manipulation and testing
    window.AuraApp = {
        toggleSourcesDrawer: function (drawerId) {
            const drawer = document.getElementById(drawerId);
            if (drawer) {
                drawer.classList.toggle("open");
            }
        },
        setAudienceMode: setAudienceMode,
        setGuardrailMode: setGuardrailMode,
        getState: function () {
            return { ...state };
        }
    };

    // Run on DOM loaded
    document.addEventListener("DOMContentLoaded", init);

})();