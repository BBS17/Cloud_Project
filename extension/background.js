const API_URL = "http://127.0.0.1:8081/predict";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action !== "predict") return false;

    fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message.text }),
    })
        .then(async (response) => {
            let data;
            try {
                data = await response.json();
            } catch {
                // Response was not JSON (e.g. plain-text rate limit or proxy error)
                sendResponse({ error: `Server error (HTTP ${response.status})` });
                return;
            }
            if (!response.ok) {
                sendResponse({ error: data.detail ?? `Request failed (${response.status})` });
            } else {
                sendResponse({ result: data });
            }
        })
        .catch((error) => {
            sendResponse({ error: `Connection failed: ${error.message}` });
        });

    return true; // keep message channel open for async response
});
