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
        <div className="p-2 border-t flex items-center bg-white">
            <input
                type="text"
                placeholder="Type a message..."
                value={message}
                onChange={(e) => {setMessage(e.target.value)}}
                className="flex-1 p-2 border rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-green-400"
            />
            <button className="ml-2 bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600"
                    onClick={(e) => handleSendMessage(e)}>
                Send
            </button>
        </div>
    )
}
export default ChatMessageBox;