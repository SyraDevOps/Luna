// Elementos DOM
const messagesContainer = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const thinkingContent = document.getElementById('thinking-content');
const modelSelect = document.getElementById('model-select');

// Estado
let isProcessing = false;
const chatHistory = [];

// Função para adicionar mensagem ao chat
function addMessage(content, isUser = false, timestamp = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = isUser ? 'Você' : 'LunaGPT';
    messageDiv.appendChild(header);
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Processar markdown
    const formattedContent = formatMarkdown(content);
    messageContent.innerHTML = formattedContent;
    messageDiv.appendChild(messageContent);
    
    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'timestamp';
    const now = timestamp || new Date();
    timestampDiv.textContent = `${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}`;
    messageDiv.appendChild(timestampDiv);
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Adicionar à história
    chatHistory.push({
        role: isUser ? 'user' : 'assistant',
        content: content,
        timestamp: now
    });
    
    // Salvar no localStorage
    localStorage.setItem('lunaChatHistory', JSON.stringify(chatHistory));
}

// Função para formatar markdown
function formatMarkdown(text) {
    // Negrito
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Itálico
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Código inline
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Blocos de código
    text = text.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>');
    
    // Parágrafos
    text = text.replace(/\n\n/g, '<br><br>');
    
    return text;
}

// Função para mostrar indicador de digitação
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        typingDiv.appendChild(dot);
    }
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Função para remover indicador de digitação
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Função para enviar mensagem
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isProcessing) return;
    
    isProcessing = true;
    sendButton.disabled = true;
    
    // Limpar input
    userInput.value = '';
    
    // Adicionar mensagem do usuário
    addMessage(message, true);
    
    // Mostrar indicador de digitação
    showTypingIndicator();
    
    // Atualizar área de pensamento
    thinkingContent.textContent = 'Processando sua mensagem...';
    
    try {
        // Enviar requisição para o backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                model: modelSelect.value,
                history: chatHistory.slice(-10) // Últimas 10 mensagens
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Remover indicador de digitação
            removeTypingIndicator();
            
            // Adicionar resposta
            addMessage(data.response);
            
            // Atualizar área de pensamento
            thinkingContent.textContent = data.thinking || 'Raciocínio não disponível';
        } else {
            // Lidar com erro
            removeTypingIndicator();
            addMessage(`Erro: ${data.error || 'Algo deu errado'}`, false);
            thinkingContent.textContent = `Erro: ${data.error}\n${data.details || ''}`;
        }
    } catch (error) {
        console.error('Erro ao comunicar com servidor:', error);
        removeTypingIndicator();
        addMessage('Desculpe, ocorreu um erro na comunicação com o servidor. Por favor, tente novamente.', false);
        thinkingContent.textContent = `Erro de comunicação: ${error.message}`;
    } finally {
        isProcessing = false;
        sendButton.disabled = false;
    }
}

// Carregar histórico do chat do localStorage
function loadChatHistory() {
    const savedHistory = localStorage.getItem('lunaChatHistory');
    if (savedHistory) {
        const history = JSON.parse(savedHistory);
        
        // Limitar a exibição a 50 mensagens para performance
        const limitedHistory = history.slice(-50);
        
        // Limpar mensagens existentes
        messagesContainer.innerHTML = '';
        
        // Recriar mensagens
        limitedHistory.forEach(msg => {
            const timestamp = new Date(msg.timestamp);
            addMessage(msg.content, msg.role === 'user', timestamp);
        });
    }
}

// Event Listeners
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Carregar histórico ao iniciar
document.addEventListener('DOMContentLoaded', () => {
    loadChatHistory();
});