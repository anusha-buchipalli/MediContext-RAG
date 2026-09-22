// ==========================================================
// GET HTML ELEMENTS
// ==========================================================

const chatForm = document.getElementById("chat-form");

const questionInput = document.getElementById("question");

const answerBox = document.getElementById("answer");

const answerContainer =
    document.getElementById("answer-container");

const loadingContainer =
    document.getElementById("loading");

const submitButton =
    document.getElementById("submit-button");


// ==========================================================
// FORM SUBMISSION
// ==========================================================

chatForm.addEventListener(
    "submit",
    async function (event) {

        // Prevent page refresh
        event.preventDefault();


        // Get user's question
        const question =
            questionInput.value.trim();


        // --------------------------------------------------
        // Validate question
        // --------------------------------------------------

        if (!question) {

            answerContainer.classList.remove("hidden");

            answerBox.textContent =
                "Please enter a medical question.";

            return;
        }


        // --------------------------------------------------
        // Show loading state
        // --------------------------------------------------

        answerContainer.classList.add("hidden");

        loadingContainer.classList.remove("hidden");

        questionInput.disabled = true;

        submitButton.disabled = true;

        submitButton.textContent = "Processing...";


        try {

            // --------------------------------------------------
            // Send question to Flask
            // --------------------------------------------------

            const response = await fetch(
                "/ask",
                {

                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        question: question
                    })

                }
            );


            // --------------------------------------------------
            // Read Flask response
            // --------------------------------------------------

            const data = await response.json();


            // Hide loading
            loadingContainer.classList.add("hidden");


            // --------------------------------------------------
            // Handle server error
            // --------------------------------------------------

            if (!response.ok) {

                answerContainer.classList.remove(
                    "hidden"
                );

                answerBox.textContent =
                    data.error ||
                    "Something went wrong while processing your question.";

                return;
            }


            // --------------------------------------------------
            // Display chatbot answer
            // --------------------------------------------------

            answerContainer.classList.remove(
                "hidden"
            );

            answerBox.textContent =
                data.answer;


        } catch (error) {

            console.error(
                "Chatbot error:",
                error
            );


            loadingContainer.classList.add(
                "hidden"
            );


            answerContainer.classList.remove(
                "hidden"
            );


            answerBox.textContent =
                "Unable to connect to the chatbot server. Please try again.";


        } finally {

            // --------------------------------------------------
            // Restore input
            // --------------------------------------------------

            questionInput.disabled = false;

            submitButton.disabled = false;

            submitButton.textContent =
                "Ask Question";


            // Put cursor back in question box
            questionInput.focus();

        }

    }
);


// ==========================================================
// ENTER KEY SUPPORT
// ==========================================================

questionInput.addEventListener(
    "keydown",
    function (event) {

        /*
        Enter          -> submit question
        Shift + Enter  -> new line
        */

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            chatForm.requestSubmit();

        }

    }
);