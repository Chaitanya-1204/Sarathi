document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const urlSection = document.getElementById('url-section');
    const chatSection = document.getElementById('chat-section');
    const repoUrlInput = document.getElementById('repo-url');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    const statusMessage = document.getElementById('status-message');
    const loaderIcon = document.getElementById('loader-icon');
    const analyzeIcon = document.getElementById('analyze-icon');

    const chatContainer = document.getElementById('chat-container');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');

    // --- State Management ---
    let chatHistory = [];

    // --- Helper Functions ---

    /**
     * Converts a simplified markdown string to HTML.
     * @param {string} text - The markdown text.
     * @returns {string} - The corresponding HTML.
     */
    function markdownToHtml(text) {
        // Handle headings (e.g., ### Heading)
        text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        
        // Handle bold text (**text**)
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Handle inline code (`code`)
        text = text.replace(/`(.*?)`/g, '<code>$1</code>');

        // Handle lists (1., 2., or * )
        text = text.replace(/^\s*\n?1\. (.*$)/gim, '<ol><li>$1</li></ol>');
        text = text.replace(/^\s*\n?\* (.*$)/gim, '<ul><li>$1</li></ul>');
        // Combine adjacent list items
        text = text.replace(/<\/ol>\s*<ol>/g, '');
        text = text.replace(/<\/ul>\s*<ul>/g, '');

        // Handle paragraphs (newlines)
        text = text.split('\n').map(p => p.trim() ? `<p>${p}</p>` : '').join('');
        // Clean up empty paragraphs that might result from other replacements
        text = text.replace(/<p><(h[23]|ul|ol)>/g, '<$1>');
        text = text.replace(/<\/(h[23]|ul|ol)><\/p>/g, '</$1>');


        return text;
    }


    /**
     * Adds a message to the chat container.
     * @param {string} sender - 'user' or 'ai'.
     * @param {string} text - The message content.
     */
    function addMessage(sender, text) {
        // Remove typing indicator if it exists
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }

        const messageElement = document.createElement('div');
        messageElement.className = `chat-message flex items-start gap-3 ${sender === 'user' ? 'justify-end' : ''}`;

        const avatar = document.createElement('div');
        avatar.className = `w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center ${sender === 'user' ? 'bg-indigo-500' : 'bg-gray-600'}`;
        avatar.textContent = sender === 'user' ? 'You' : 'AI';
        avatar.classList.add('text-xs', 'font-bold');

        const messageBubble = document.createElement('div');
        messageBubble.className = `message-bubble max-w-2xl p-4 rounded-xl ${sender === 'user' ? 'bg-indigo-600 rounded-br-none' : 'bg-gray-700 rounded-bl-none'}`;
        
        // Use the new markdown parser for AI messages
        if (sender === 'ai') {
            messageBubble.innerHTML = markdownToHtml(text);
        } else {
            messageBubble.textContent = text;
        }


        if (sender === 'user') {
            messageElement.appendChild(messageBubble);
            messageElement.appendChild(avatar);
        } else {
            messageElement.appendChild(avatar);
            messageElement.appendChild(messageBubble);
        }

        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight; // Auto-scroll to bottom
    }

    /**
     * Shows a typing indicator for the AI.
     */
    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.className = 'chat-message flex items-start gap-3';
        indicator.innerHTML = `
            <div class="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-gray-600 text-xs font-bold">AI</div>
            <div class="max-w-xl p-4 rounded-xl bg-gray-700 rounded-bl-none flex items-center space-x-2">
                <div class="typing-dot"></div>
                <div class="typing-dot" style="animation-delay: 0.2s;"></div>
                <div class="typing-dot" style="animation-delay: 0.4s;"></div>
            </div>
        `;
        chatContainer.appendChild(indicator);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    /**
     * Sets the state of the analyze button (loading or idle).
     * @param {boolean} isLoading - True if loading, false otherwise.
     */
    function setAnalyzeButtonState(isLoading) {
        analyzeBtn.disabled = isLoading;
        if (isLoading) {
            analyzeBtnText.textContent = 'Analyzing...';
            loaderIcon.classList.remove('hidden');
            analyzeIcon.classList.add('hidden');
        } else {
            analyzeBtnText.textContent = 'Analyze';
            loaderIcon.classList.add('hidden');
            analyzeIcon.classList.remove('hidden');
        }
    }

    // --- Event Handlers ---

    /**
     * Handles the click event for the "Analyze" button.
     */
    async function handleAnalyzeClick() {
        const repoUrl = repoUrlInput.value.trim();
        if (!repoUrl) {
            statusMessage.textContent = 'Please enter a valid GitHub repository URL.';
            statusMessage.className = 'text-red-400 mt-4 text-sm';
            return;
        }

        setAnalyzeButtonState(true);
        statusMessage.textContent = 'Cloning repository and building knowledge base... This may take a few minutes.';
        statusMessage.className = 'text-blue-400 mt-4 text-sm';

        try {
            const response = await fetch('/preprocess', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ repo_url: repoUrl }),
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'An unknown error occurred.');
            }

            // Transition to chat interface
            urlSection.classList.add('hidden');
            chatSection.classList.remove('hidden');
            addMessage('ai', result.message || 'Ready to chat!');

        } catch (error) {
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.className = 'text-red-400 mt-4 text-sm';
            setAnalyzeButtonState(false);
        }
    }

    /**
     * Handles the submission of the chat form.
     */
    async function handleChatSubmit(event) {
        event.preventDefault();
        const messageText = chatInput.value.trim();
        if (!messageText) return;

        // Add user message to UI and history
        addMessage('user', messageText);
        chatHistory.push({ role: 'user', content: messageText });
        chatInput.value = '';
        showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: messageText, history: chatHistory }),
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Failed to get a response.');
            }

            // Add AI response to UI and history
            addMessage('ai', result.answer);
            chatHistory.push({ role: 'ai', content: result.answer });

        } catch (error) {
            addMessage('ai', `Sorry, an error occurred: ${error.message}`);
        }
    }

    // --- Attach Event Listeners ---
    analyzeBtn.addEventListener('click', handleAnalyzeClick);
    chatForm.addEventListener('submit', handleChatSubmit);
});
