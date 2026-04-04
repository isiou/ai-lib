const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const loadingIndicator = document.getElementById("loading-indicator");

let messages = [];

function appendMessage(role, content) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message");
  if (role === "user") {
    messageDiv.classList.add("user-message");
  } else {
    messageDiv.classList.add("bot-message");
  }
  messageDiv.textContent = content;
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  // Add user message to UI
  appendMessage("user", text);
  messages.push({ role: "user", content: text });
  userInput.value = "";

  // Show loading
  loadingIndicator.style.display = "block";
  chatContainer.appendChild(loadingIndicator);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages: messages }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Hide loading indicator immediately when stream starts
    loadingIndicator.style.display = "none";
    document.body.appendChild(loadingIndicator);

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let assistantMessage = "";

    // Create empty message div for the bot
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", "bot-message");
    chatContainer.appendChild(messageDiv);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      assistantMessage += decoder.decode(value, { stream: true });

      // Format to hide <think> tags dynamically
      let displayMessage = assistantMessage;
      displayMessage = displayMessage.replace(
        /<think>[\s\S]*?<\/think>\n*/g,
        "",
      );
      displayMessage = displayMessage.replace(
        /<think>[\s\S]*$/,
        "💭 思考中...",
      );

      messageDiv.textContent = displayMessage;
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Final cleanup for history array
    const finalCleanedMessage = assistantMessage
      .replace(/<think>[\s\S]*?<\/think>\n*/g, "")
      .replace(/<think>[\s\S]*/g, "")
      .trim();
    messages.push({ role: "assistant", content: finalCleanedMessage });
  } catch (error) {
    console.error("Error:", error);
    loadingIndicator.style.display = "none";
    document.body.appendChild(loadingIndicator);
    appendMessage("assistant", "抱歉，系统发生错误，请稍后再试。");
  }
}

sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    sendMessage();
  }
});
