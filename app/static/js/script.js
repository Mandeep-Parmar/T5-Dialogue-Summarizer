/**
 * Text Summarizer - Frontend Interaction Controller
 * Minimal, editorial SaaS productivity tool aesthetic
 * Compatible with existing FastAPI /summarize/ endpoint
 */
document.addEventListener("DOMContentLoaded", () => {
  // Core DOM Elements (Preserving existing IDs)
  const form = document.getElementById("summarization-form");
  const dialogueInput = document.getElementById("dialogue-input");
  const summaryText = document.getElementById("summary-text");
  const summaryOutput = document.querySelector(".summary-output");
  const submitButton = form ? form.querySelector("button[type='submit']") : null;
  const btnText = submitButton ? submitButton.querySelector(".btn-text") : null;

  // New Interactive Elements
  const charCount = document.getElementById("char-count");
  const wordCount = document.getElementById("word-count");
  const clearBtn = document.getElementById("clear-btn");
  const copyBtn = document.getElementById("copy-btn");
  const copyBtnText = document.getElementById("copy-btn-text");
  const errorBanner = document.getElementById("error-banner");
  const errorMessage = document.getElementById("error-message");
  const statusBadge = document.getElementById("summary-status-badge");

  let currentSummary = "";
  let copyTimeout = null;

  /**
   * Update character and word counters and manage submit button state
   */
  function updateInputStats() {
    const text = dialogueInput.value;
    const trimmed = text.trim();

    if (charCount) {
      charCount.textContent = text.length.toLocaleString();
    }

    if (wordCount) {
      const words = trimmed ? trimmed.split(/\s+/).length : 0;
      wordCount.textContent = words.toLocaleString();
    }

    // Toggle submit button state based on content
    if (submitButton && !submitButton.classList.contains("is-loading")) {
      submitButton.disabled = trimmed.length === 0;
    }

    // Toggle clear button state
    if (clearBtn) {
      clearBtn.disabled = text.length === 0;
    }
  }

  // Initial stats calculation
  updateInputStats();

  // Listen to input events on the textarea
  dialogueInput.addEventListener("input", updateInputStats);

  // Clear button action
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      dialogueInput.value = "";
      updateInputStats();
      dialogueInput.focus();
    });
  }

  // Keyboard shortcut: Cmd+Enter or Ctrl+Enter to submit
  dialogueInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      if (!submitButton.disabled) {
        e.preventDefault();
        form.requestSubmit();
      }
    }
  });

  /**
   * Show error banner
   */
  function showError(msg) {
    if (errorBanner && errorMessage) {
      errorMessage.textContent = msg;
      errorBanner.hidden = false;
    }
  }

  /**
   * Hide error banner
   */
  function hideError() {
    if (errorBanner) {
      errorBanner.hidden = true;
    }
  }

  /**
   * Handle Form Submission (Existing endpoint: POST /summarize/)
   */
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const dialogue = dialogueInput.value.trim();
    if (!dialogue) return;

    hideError();

    // Set UI to loading state
    submitButton.disabled = true;
    submitButton.classList.add("is-loading");
    if (btnText) btnText.textContent = "Summarizing...";

    dialogueInput.disabled = true;
    if (clearBtn) clearBtn.disabled = true;
    if (copyBtn) copyBtn.disabled = true;

    summaryText.classList.remove("placeholder-text");
    summaryText.classList.add("is-loading-text");
    summaryText.textContent = "Analyzing dialogue and generating summary...";

    if (statusBadge) statusBadge.hidden = true;

    try {
      const response = await fetch("/summarize/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ dialogue })
      });

      if (!response.ok) {
        let errorDetail = `Server error: ${response.status}`;
        try {
          const errorJson = await response.json();
          if (errorJson.detail) errorDetail = errorJson.detail;
        } catch (_) {}
        throw new Error(errorDetail);
      }

      const data = await response.json();
      currentSummary = (data.summary || "").trim();

      if (currentSummary) {
        summaryText.classList.remove("placeholder-text", "is-loading-text");
        summaryText.textContent = currentSummary;

        if (summaryOutput) summaryOutput.classList.add("has-content");
        if (statusBadge) statusBadge.hidden = false;
        if (copyBtn) copyBtn.disabled = false;
      } else {
        throw new Error("No summary returned by the model.");
      }
    } catch (err) {
      summaryText.classList.remove("is-loading-text");
      summaryText.classList.add("placeholder-text");
      summaryText.textContent = "Your summary will appear here.";

      if (summaryOutput) summaryOutput.classList.remove("has-content");
      if (statusBadge) statusBadge.hidden = true;
      if (copyBtn) copyBtn.disabled = true;

      showError(err.message || "Failed to generate summary. Please try again.");
    } finally {
      submitButton.classList.remove("is-loading");
      if (btnText) btnText.textContent = "Summarize";

      dialogueInput.disabled = false;
      updateInputStats();
      dialogueInput.focus();
    }
  });

  /**
   * Copy to Clipboard functionality
   */
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      if (!currentSummary) return;

      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(currentSummary);
        } else {
          // Fallback for older browsers
          const tempArea = document.createElement("textarea");
          tempArea.value = currentSummary;
          tempArea.style.position = "fixed";
          tempArea.style.opacity = "0";
          document.body.appendChild(tempArea);
          tempArea.focus();
          tempArea.select();
          document.execCommand("copy");
          document.body.removeChild(tempArea);
        }

        // Visual feedback
        copyBtn.classList.add("is-copied");
        if (copyBtnText) copyBtnText.textContent = "Copied!";

        clearTimeout(copyTimeout);
        copyTimeout = setTimeout(() => {
          copyBtn.classList.remove("is-copied");
          if (copyBtnText) copyBtnText.textContent = "Copy";
        }, 2000);
      } catch (err) {
        showError("Could not copy to clipboard.");
      }
    });
  }
});
