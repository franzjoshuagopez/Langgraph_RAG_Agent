const form = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const userInput = document.getElementById("user-input").value;
    chatBox.innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;
    document.getElementById("user-input").value = "";

    try {

        const response = await fetch("/chat/send_message/", {
            method: "POST",
            headers: {"Content-Type" : "application/json"},
            body: JSON.stringify({message: userInput})
        });
        const data = await response.json();
        chatBox.innerHTML += `<div><strong>FranzAI:</strong> ${data.reply}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (err) {
        chatBox.innerHTML += `<div><strong>Error:</strong> ${err}</div>`;
    }

});