//This file controls the front end extension of the application
//TODO: add features to prevent phrases less than 3 words, or set max words

//store references to dynamic elements
let analyzeButton = null;
let resultPopup = null;

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

// Create the floating Analyze button, TODO: work on styling, current styling is generic
function showAnalyzeButton(selectedText, x, y) {
    removeAnalyzeButton();

    analyzeButton = document.createElement("button");
    analyzeButton.textContent = "Analyze";

    //generic styling
    analyzeButton.style.position = "absolute";
    analyzeButton.style.left = `${x}px`;
    analyzeButton.style.top = `${y}px`;
    analyzeButton.style.zIndex = "999999";
    analyzeButton.style.padding = "6px 10px";
    analyzeButton.style.border = "none";
    analyzeButton.style.borderRadius = "6px";
    analyzeButton.style.background = "#111";
    analyzeButton.style.color = "white";
    analyzeButton.style.cursor = "pointer";
    analyzeButton.style.fontSize = "13px";
    analyzeButton.style.boxShadow = "0 2px 8px rgba(0,0,0,0.2)";

    analyzeButton.addEventListener("click", async (event) => {
    event.stopPropagation();

    const clickX = event.pageX;
    const clickY = event.pageY;

    removeAnalyzeButton();
    showResultPopup("Analyzing...", clickX, clickY);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: selectedText
        })
        });

        const data = await response.json();

        if (!response.ok) {
        showResultPopup(
            `Error: ${JSON.stringify(data, null, 2)}`,
            clickX,
            clickY
        );
        return;
        }

      // Change this formatting to match backend response shape
        const label = data.label ?? "Unknown";
        const confidence = data.confidence ?? "N/A";

        showResultPopup(
            `Label: ${label}\nConfidence: ${confidence}%`,
            clickX,
            clickY
        );
    } catch (error) {
        showResultPopup(`Connection failed: ${error.message}`, clickX, clickY);
    }
  });

  document.body.appendChild(analyzeButton);
}

// Create the floating result popup
function showResultPopup(message, x, y) {
    removeResultPopup();

    resultPopup = document.createElement("div");
    resultPopup.textContent = message;

    resultPopup.style.position = "absolute";
    resultPopup.style.left = `${x}px`;
    resultPopup.style.top = `${y + 35}px`;
    resultPopup.style.zIndex = "999999";
    resultPopup.style.maxWidth = "260px";
    resultPopup.style.padding = "10px 12px";
    resultPopup.style.borderRadius = "8px";
    resultPopup.style.background = "white";
    resultPopup.style.color = "#111";
    resultPopup.style.border = "1px solid #ccc";
    resultPopup.style.boxShadow = "0 4px 12px rgba(0,0,0,0.2)";
    resultPopup.style.fontSize = "13px";
    resultPopup.style.whiteSpace = "pre-wrap";
    resultPopup.style.lineHeight = "1.4";

    // close button
    const closeBtn = document.createElement("div");
    closeBtn.textContent = "×";
    closeBtn.style.position = "absolute";
    closeBtn.style.top = "4px";
    closeBtn.style.right = "8px";
    closeBtn.style.cursor = "pointer";
    closeBtn.style.fontWeight = "bold";
    closeBtn.style.fontSize = "14px";

    closeBtn.addEventListener("click", () => {
        removeResultPopup();
    });

    resultPopup.appendChild(closeBtn);
    document.body.appendChild(resultPopup);
}

// Detect text selection
document.addEventListener("mouseup", (event) => {
    setTimeout(() => {
        const selectedText = window.getSelection().toString().trim();

        if (selectedText.length > 0) {
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