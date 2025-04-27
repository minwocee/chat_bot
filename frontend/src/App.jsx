import { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);

  const chatEndRef = useRef(null); // 채팅 자동 스크롤용 ref

  // 자동 스크롤
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const askBot = async () => {
    if (!query.trim()) return;

    const userMessage = { sender: "user", text: query };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setQuery("");

    try {
      const res = await axios.post("http://localhost:8000/ask", {
        query: userMessage.text,
      });
      const botMessage = { sender: "bot", text: res.data.answer };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      const errorMsg = {
        sender: "bot",
        text: "❗ 서버 응답 오류. 다시 시도해주세요.",
      };
      setMessages((prev) => [...prev, errorMsg]);
    }

    setLoading(false);
  };

  return (
    <section className="container">
      <div className="container_box">
        <div className="title">
          <img src="./logo.png" alt="logo" />
          <h1> 컴공도우미봇</h1>
        </div>

        <p className="description">
          컴공 신입생을 위한 스마트 질문 도우미입니다. 궁금한 내용을 입력하고
          대화해보세요!
        </p>

        <div className="chat-history">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <div className="sender-label">
                {msg.sender === "user" ? "나" : "도우미봇"}
              </div>
              <div className="bubble">{msg.text}</div>
            </div>
          ))}

          {loading && (
            <div className="message bot">
              <div className="sender-label">도우미봇</div>
              <div className="bubble">
                답변 생성 중...
                <span className="loader"></span>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <textarea
          placeholder="질문을 입력하세요 (예: 자료구조란?)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              askBot();
            }
          }}
        ></textarea>

        <button onClick={askBot} disabled={loading}>
          {loading ? "답변 생성 중..." : "질문하기"}
        </button>
      </div>
    </section>
  );
}

export default App;
