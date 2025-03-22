import {useState} from "react";

const ChatMessageBox = ({chatHistory,setChatHistory, generateBotReplies}) => {
    const [message, setMessage] = useState('');

    const handleSendMessage = (e) => {
        e.preventDefault();
        const userMessage = message.trim();
        if (!userMessage) {
            return ;
        }
        setChatHistory((history)=> [...history,{role:"user" , text: userMessage}]);
        setTimeout(()=>{
            setChatHistory((history) => [...history,{role:"bot" , text: "Analyzing...."}]);

            generateBotReplies([...chatHistory,{role:"user" , text: userMessage}]);
        },600)
        setMessage("");
    }
    return (
        <div className="p-3 border-t flex items-center bg-white shadow-md rounded-b-lg">
          <input
            type="text"
            placeholder="Type a message..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-full text-black focus:outline-none focus:ring-2 focus:ring-green-400 bg-gray-100"
          />
          <button
            className="ml-3 px-5 py-2.5 rounded-full shadow-md transition-all bg-gradient-to-r from-green-400 to-blue-500 text-white hover:from-green-500 hover:to-blue-600 hover:scale-105"
            onClick={(e) => handleSendMessage(e)}
          >
            Send
          </button>
        </div>

    )
}
export default ChatMessageBox;