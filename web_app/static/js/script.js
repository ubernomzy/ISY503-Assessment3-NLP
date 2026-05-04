const textarea = document.getElementById("reviewInput");
const chatInput = document.querySelector(".chat-input");

function resetTextarea() {
    textarea.style.height = "42px";
}

function autoResize() {
    textarea.style.height = "42px";
    textarea.style.height = textarea.scrollHeight + "px";
}

async function analyzeReview() {
    const reviewText = textarea.value.trim();

    if (reviewText === "") {
        showResult("Please enter a review.", "negative");
        textarea.focus();
        resetTextarea();
        return;
    }

    showResult("Analyzing review...", "loading");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                review: reviewText
            })
        });

        if (!response.ok) {
            throw new Error("Backend response failed");
        }

        const data = await response.json();

        
        if (data.sentiment.toLowerCase().includes("positive")) {
            showResult("This review looks positive 😊", "positive");
        } else {
            showResult("This review looks negative 😕", "negative");
        }

    } catch (error) {
        console.error("Error connecting to backend:", error);

        
        const text = reviewText.toLowerCase();

        if (
            text.includes("good") ||
            text.includes("love") ||
            text.includes("amazing")
        ) {
            showResult("This review looks positive 😊", "positive");
        } else {
            showResult("This review looks negative 😕", "negative");
        }
    }

    textarea.value = "";
    resetTextarea();
    textarea.focus();
}

function showResult(message, type) {
    const popup = document.getElementById("resultPopup");
    const resultText = document.getElementById("resultText");

    resultText.innerText = message;

    // REMOVE TODAS as classes antes
    popup.classList.remove("positive", "negative", "loading", "show");

    // força reset visual
    void popup.offsetWidth;

    // adiciona a nova
    popup.classList.add(type);
    popup.classList.add("show");
}

chatInput.addEventListener("click", function (event) {
    if (event.target.tagName !== "BUTTON") {
        textarea.focus();
    }
});

textarea.addEventListener("input", autoResize);

textarea.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        analyzeReview();
    }
});