function sendText() {
    const question = document.getElementById("question").value;

    if (!question.trim()) {
        alert("Please enter a question.");
        return;
    } else {
        const messageContainer = document.getElementById("message-container");
        const loaderDiv = document.createElement("div");
        const questionDiv = document.createElement("div");

        questionDiv.classList.add("message", "sender-message", "animate__animated", "animate__fadeInRight");
        questionDiv.textContent = question;
        questionDiv.addEventListener("animationend", () => {
            questionDiv.classList.remove("animate__animated", "animate__fadeInRight");
        });

        messageContainer.appendChild(questionDiv);
        document.getElementById('send-button').disabled = true;
        document.getElementById("question").value = "";

        

        loaderDiv.classList.add("loader");
        loaderDiv.id = "loader";
        messageContainer.appendChild(loaderDiv);

        fetch("/ask-question", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: question }) 
        })
        .then(response => response.json())
        .then(data => {

            const loader = document.getElementById("loader");
            if (loader) {
                loader.remove();
            }
        
            const answerDiv = document.createElement("div");
            answerDiv.classList.add("message", "receiver-message", "animate__animated", "animate__fadeInLeft");
            messageContainer.appendChild(answerDiv);
            messageContainer.style.borderRadius = "30px";
            messageContainer.scrollTop = messageContainer.scrollHeight;
        
            const words = data.answer.split(" "); 
            let displayedText = "";
        
            words.forEach((word, index) => {
                setTimeout(() => {
                    displayedText += (index === 0 ? "" : " ") + word;
                    answerDiv.textContent = displayedText; 

                    // Add the icon instead of "HELLO"
                    const iconSpan = document.createElement('span');
                    iconSpan.innerHTML = '<i class="fa-solid fa-pen"></i>'; // Example icon (smile)
                    if (index != words.length - 1) {
                        // Append the icon
                        answerDiv.appendChild(iconSpan);
                    }
                    messageContainer.scrollTop = messageContainer.scrollHeight;
                }, index * 100); 
            });
            setTimeout(() => {
                document.getElementById('send-button').disabled = false;
            }, words.length * 300); 
        })
        .catch(error => {
            console.error("Error:", error);

            const loader = document.getElementById("loader");
            if (loader) {
                loader.remove();
            }

            alert("An error occurred. Please try again.");
            document.getElementById('send-button').disabled = false;
        });
    }
}
