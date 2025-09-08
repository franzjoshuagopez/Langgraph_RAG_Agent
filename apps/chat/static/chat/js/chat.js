const form = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");

function addMessage(role, text, tempId = null) {
    const msg = document.createElement("div");
    msg.classList.add("message", role);
    msg.textContent = text;

    if (tempId) {
        msg.setAttribute("id", tempId)
    }

    chatBox.appendChild(msg);

    chatBox.scrollTop = chatBox.scrollHeight;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const userInput = document.getElementById("user-input").value.trim();
    
    if (!userInput) return;

    addMessage("user", userInput);

    document.getElementById("user-input").value = "";

    const typingId = "typing-indicator";
    addMessage("ai", "FranzAI is thinking...", typingId);

    try {

        const response = await fetch("/chat/send_message/", {
            method: "POST",
            headers: {"Content-Type" : "application/json"},
            body: JSON.stringify({message: userInput})
        });
        const data = await response.json();

        const typingBubble = document.getElementById(typingId);
        if (typingBubble) typingBubble.remove();

        addMessage("ai", data.reply);


    } catch (err) {
        chatBox.innerHTML += `<div><strong>Error:</strong> ${err}</div>`;
    }

});