import { useState, useRef, useEffect } from 'react';
import type { KeyboardEvent } from 'react';
import './index.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
}

function ChatView() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: '你好！我是厦小嘉，厦门大学嘉庚学院图书馆的专属智能助手。有什么我可以帮你的吗？',
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const sendMessage = async () => {
    const text = inputValue.trim();
    if (!text || isLoading) return;

    const newMessages: Message[] = [...messages, { role: 'user', content: text }];
    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages.filter((m) => m.role === 'user' || m.role === 'assistant'),
        }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');
      const decoder = new TextDecoder('utf-8');

      let assistantMessage = '';

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '', isStreaming: true },
      ]);
      setIsLoading(false);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        assistantMessage += decoder.decode(value, { stream: true });

        const displayMessage = assistantMessage
          .replace(/<think>[\s\S]*?<\/think>\n*/g, '')
          .replace(/<think>[\s\S]*$/, '💭 思考中...');

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: displayMessage,
            isStreaming: true,
          };
          return updated;
        });
      }

      const finalCleanedMessage = assistantMessage
        .replace(/<think>[\s\S]*?<\/think>\n*/g, '')
        .replace(/<think>[\s\S]*/g, '')
        .trim();

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: 'assistant',
          content: finalCleanedMessage,
          isStreaming: false,
        };
        return updated;
      });
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '抱歉，系统发生错误，请稍后再试。' },
      ]);
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div className="tab-content">
      <div className="chat-container" ref={chatContainerRef}>
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.role === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {msg.content}
          </div>
        ))}
        {isLoading && <div className="loading">厦小嘉正在思考...</div>}
      </div>
      <div className="input-container">
        <input
          type="text"
          className="user-input"
          placeholder="请输入你的问题..."
          autoComplete="off"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading || messages.some((m) => m.isStreaming)}
        />
        <button
          className="send-button"
          onClick={sendMessage}
          disabled={isLoading || messages.some((m) => m.isStreaming) || !inputValue.trim()}
        >
          发送
        </button>
      </div>
    </div>
  );
}

function KnowledgeView() {
  const [kbText, setKbText] = useState('');
  const [status, setStatus] = useState({ type: '', message: '' });
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    const text = kbText.trim();
    if (!text) {
      setStatus({ type: 'error', message: '请输入知识库内容！' });
      return;
    }

    setIsSaving(true);
    setStatus({ type: 'loading', message: '正在保存至向量库...' });

    try {
      const response = await fetch('/api/knowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const result = await response.json();
      setStatus({ type: 'success', message: result.message || '保存成功！' });
      setKbText('');

      setTimeout(() => {
        setStatus({ type: '', message: '' });
      }, 3000);
    } catch (error) {
      console.error('Error saving knowledge:', error);
      setStatus({ type: 'error', message: '保存失败，请稍后再试。' });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="tab-content">
      <div className="kb-container">
        <h2>添加知识库片段</h2>
        <p>在此处输入你要让模型记住的文本信息。这些信息会即时录入向量库中：</p>
        <textarea
          className="kb-input"
          placeholder="输入知识库文本内容，例如：图书馆本周六上午闭馆..."
          value={kbText}
          onChange={(e) => setKbText(e.target.value)}
          disabled={isSaving}
        ></textarea>
        {status.message && (
          <div className={`kb-status ${status.type}`}>{status.message}</div>
        )}
        <button className="save-kb-button" onClick={handleSave} disabled={isSaving}>
          {isSaving ? '保存中...' : '保存到知识库'}
        </button>
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'knowledge'>('chat');

  return (
    <>
      <div className="tabs">
        <div
          className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          智能对话
        </div>
        <div
          className={`tab ${activeTab === 'knowledge' ? 'active' : ''}`}
          onClick={() => setActiveTab('knowledge')}
        >
          知识库录入
        </div>
      </div>
      {activeTab === 'chat' ? <ChatView /> : <KnowledgeView />}
    </>
  );
}

export default App;
