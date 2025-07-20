async function uploadPDF() {
  const fileInput = document.getElementById("pdfInput");
  const file = fileInput.files[0];
  if (!file) return alert("Please select a PDF file.");

  const formData = new FormData();
  formData.append("pdf", file);

  const res = await fetch("/upload", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  alert(data.message);
}

async function askQuestion() {
  const input = document.getElementById("questionInput");
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  addMessage("user", question);
  addMessage("bot", "<span class='loading'>Thinking...</span>");

  const res = await fetch("/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ question })
  });

  const data = await res.json();
  // Remove loading message
  removeLoading();
  addMessage("bot", data.answer);
}

function addMessage(sender, text) {
  const chatBox = document.getElementById("chatOutput");

  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = text;

  msgDiv.appendChild(bubble);
  chatBox.appendChild(msgDiv);

  chatBox.scrollTop = chatBox.scrollHeight; // Auto scroll to bottom
}

function removeLoading() {
  const loading = document.querySelector(".bot .loading");
  if (loading) loading.parentElement.parentElement.remove();
}

// Send on Enter
document.getElementById("questionInput").addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    askQuestion();
  }
});
