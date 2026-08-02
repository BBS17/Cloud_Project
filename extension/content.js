// API endpoint lives in background.js — update it there when deploying

const MIN_WORD_COUNT = 3;
const MAX_CHAR_COUNT = 5000;

let analyzeButton = null;
let resultPopup = null;

// Inject styles once so the button and popup can use CSS classes
function injectStyles() {
    if (document.getElementById("fact-checker-styles")) return;
    const style = document.createElement("style");
    style.id = "fact-checker-styles";
    style.textContent = `
        .fc-btn {
            position: absolute;
            z-index: 2147483647;
            padding: 5px 13px;
            border: none;
            border-radius: 20px;
            background: #1a1a1a;
            color: #fff;
            cursor: pointer;
            font-size: 12px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-weight: 600;
            letter-spacing: 0.3px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            transition: background 0.15s, transform 0.1s, box-shadow 0.15s;
            user-select: none;
        }
        .fc-btn:hover {
            background: #333;
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        }
        .fc-btn:active { transform: translateY(0); }

        .fc-popup {
            position: absolute;
            z-index: 2147483647;
            min-width: 210px;
            max-width: 280px;
            padding: 14px 16px 12px;
            border-radius: 12px;
            background: #fff;
            box-shadow: 0 8px 28px rgba(0,0,0,0.14), 0 1px 4px rgba(0,0,0,0.07);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 13px;
            line-height: 1.5;
            border: 1px solid rgba(0,0,0,0.06);
        }
        .fc-close {
            position: absolute;
            top: 9px;
            right: 11px;
            cursor: pointer;
            color: #bbb;
            font-size: 17px;
            line-height: 1;
            transition: color 0.1s;
        }
        .fc-close:hover { color: #444; }

        .fc-header { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
        .fc-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
        .fc-dot--truth   { background: #16a34a; }
        .fc-dot--misinfo { background: #dc2626; }
        .fc-label { font-weight: 700; font-size: 14px; }
        .fc-label--truth   { color: #16a34a; }
        .fc-label--misinfo { color: #dc2626; }

        .fc-confidence { color: #666; font-size: 12px; margin: 0; }

        .fc-bar-track {
            height: 4px;
            background: #e5e7eb;
            border-radius: 2px;
            margin-top: 9px;
            overflow: hidden;
        }
        .fc-bar-fill { height: 100%; border-radius: 2px; }
        .fc-bar-fill--truth   { background: #16a34a; }
        .fc-bar-fill--misinfo { background: #dc2626; }

        .fc-status { color: #999; font-size: 12px; font-style: italic; margin: 0; }
        .fc-error  { color: #dc2626; font-size: 12px; margin: 0; }
    `;
    document.head.appendChild(style);
}

// Remove existing analyze button
function removeAnalyzeButton() {
    if (analyzeButton) {
        analyzeButton.remove();
        analyzeButton = null;
    }
}

// Remove existing result popup
function removeResultPopup() {
    if (resultPopup) {
        resultPopup.remove();
        resultPopup = null;
    }
}

// Create the floating Analyze button
function showAnalyzeButton(selectedText, x, y) {
    removeAnalyzeButton();

    analyzeButton = document.createElement("button");
    analyzeButton.className = "fc-btn";
    analyzeButton.textContent = "Analyze";
    analyzeButton.style.left = `${x}px`;
    analyzeButton.style.top  = `${y}px`;

    analyzeButton.addEventListener("click", (event) => {
        event.stopPropagation();

        const clickX = event.pageX;
        const clickY = event.pageY;

        removeAnalyzeButton();
        showResultPopup({ type: "loading" }, clickX, clickY);

        // Background script makes the fetch (bypasses mixed-content restriction)
        chrome.runtime.sendMessage({ action: "predict", text: selectedText }, (response) => {
            if (chrome.runtime.lastError) {
                showResultPopup({ type: "error", message: chrome.runtime.lastError.message }, clickX, clickY);
                return;
            }
            if (response.error) {
                showResultPopup({ type: "error", message: response.error }, clickX, clickY);
                return;
            }
            const { label, confidence } = response.result;
            showResultPopup(
                { type: "result", label: label ?? "Unknown", confidence: confidence ?? 0 },
                clickX,
                clickY
            );
        });
    });

    document.body.appendChild(analyzeButton);
}

// Create the floating result popup
// payload: { type: "loading" } | { type: "result", label, confidence } | { type: "error", message }
function showResultPopup(payload, x, y) {
    removeResultPopup();

    resultPopup = document.createElement("div");
    resultPopup.className = "fc-popup";
    resultPopup.style.left = `${x}px`;
    resultPopup.style.top  = `${y + 35}px`;

    const closeBtn = document.createElement("span");
    closeBtn.className = "fc-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", removeResultPopup);
    resultPopup.appendChild(closeBtn);

    if (payload.type === "loading") {
        const status = document.createElement("p");
        status.className = "fc-status";
        status.textContent = "Analyzing…";
        resultPopup.appendChild(status);

    } else if (payload.type === "result") {
        const isMisinfo = payload.label === "Misinformation";
        const mod = isMisinfo ? "misinfo" : "truth";

        const header = document.createElement("div");
        header.className = "fc-header";

        const dot = document.createElement("span");
        dot.className = `fc-dot fc-dot--${mod}`;

        // textContent used for all API values — no innerHTML risk
        const labelEl = document.createElement("span");
        labelEl.className = `fc-label fc-label--${mod}`;
        labelEl.textContent = payload.label;

        header.appendChild(dot);
        header.appendChild(labelEl);
        resultPopup.appendChild(header);

        const confEl = document.createElement("p");
        confEl.className = "fc-confidence";
        confEl.textContent = `Confidence: ${payload.confidence}%`;
        resultPopup.appendChild(confEl);

        const track = document.createElement("div");
        track.className = "fc-bar-track";
        const fill = document.createElement("div");
        fill.className = `fc-bar-fill fc-bar-fill--${mod}`;
        fill.style.width = `${payload.confidence}%`;
        track.appendChild(fill);
        resultPopup.appendChild(track);

    } else if (payload.type === "error") {
        const errEl = document.createElement("p");
        errEl.className = "fc-error";
        errEl.textContent = payload.message;
        resultPopup.appendChild(errEl);
    }

    document.body.appendChild(resultPopup);
}

// Detect text selection
document.addEventListener("mouseup", (event) => {
    setTimeout(() => {
        const selectedText = window.getSelection().toString().trim();
        const wordCount = selectedText.split(/\s+/).filter(Boolean).length;

        if (selectedText.length > MAX_CHAR_COUNT) {
            // silently ignore — too large to analyse
            removeAnalyzeButton();
        } else if (wordCount >= MIN_WORD_COUNT) {
            injectStyles();
            showAnalyzeButton(selectedText, event.pageX + 10, event.pageY + 10);
        } else {
            removeAnalyzeButton();
        }
    }, 10);
});

// Hide button/popup when clicking elsewhere
document.addEventListener("mousedown", (event) => {
    const clickedAnalyzeButton = analyzeButton && analyzeButton.contains(event.target);
    const clickedResultPopup = resultPopup && resultPopup.contains(event.target);

    if (!clickedAnalyzeButton && !clickedResultPopup) {
        if (analyzeButton) {
            analyzeButton.remove();
            analyzeButton = null;
        }

        if (resultPopup) {
        resultPopup.remove();
        resultPopup = null;
        }

    }
});