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
  const question = document.getElementById("questionInput").value;
  const res = await fetch("/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ question })
  });

  const data = await res.json();
  const chatDiv = document.getElementById("chatOutput");
  chatDiv.innerHTML += `<p><strong>You:</strong> ${question}</p>`;
  chatDiv.innerHTML += `<p><strong>Bot:</strong> ${data.answer}</p>`;
}
