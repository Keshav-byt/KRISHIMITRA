import ChatMessageBox from "./ChatMessageBox.jsx";
import {useEffect, useRef, useState} from "react";
import Messages from "./Messages.jsx";

const Chat = ({handleChatDisplay}) => {
    const [chatHistory, setChatHistory] = useState([{role: "model" , text: "Hi! How can i help?"}]);
    const chatBodyRef = useRef();

    const generateBotReplies = async (history) => {
        const updateHistory= (text) => {
            setChatHistory(prev=> [...prev.filter(msg => msg.text !== "Analyzing...." ), {role:"model" , text}]);
        }

        const formattedHistory = history.map(({ role, text }) => ({ role, parts: [{ text }] }));

        const request = {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ contents: formattedHistory }),
        };

        try {
            const response = await fetch(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyCWxueKaAmxBwbqE19PS0tOF5bcxEyvzcQ",
                request
            );

            if (!response.ok) {
                throw new Error(`Failed to fetch data: ${response.statusText}`);
            }

            const data = await response.json();
            console.log(data);

            const apiResponse = data?.candidates[0]?.content?.parts[0]?.text.replace(/\*\*(.*?)\*\*/g, "$1").trim();
            updateHistory(apiResponse);

        } catch (e) {
            console.log(e);
        }
    };

    useEffect(() => {
        chatBodyRef.current.scrollTo({top: chatBodyRef.current.scrollHeight, behavior: "smooth"});
    },[chatHistory])

    return (
        <div className="w-[22em] h-[32em] flex flex-col bg-white shadow-lg rounded-lg border ">

            <div className="h-[3em] text-black shadow-[0px_50px_100px_50px_rgba(0,_0,_0,_0.1)] flex justify-between items-center px-4 font-semibold rounded-t-lg">
                AI ChatBot
                <img src="https://www.svgrepo.com/show/506172/cross.svg"
                     className="w-5 h-5 cursor-pointer"
                     onClick={() => handleChatDisplay(false)} // Close on Click
                     alt="Close" />
            </div>

            {/* Chat Body */}
            <div ref={chatBodyRef} className="flex-1 overflow-y-auto p-3 space-y-2 bg-gray-200" >

                {chatHistory.map((chat,index) => (
                    <Messages chat={chat} key={index} />
                ))}

            </div>

            {/* Input Field */}
            <ChatMessageBox chatHistory={chatHistory}
                            setChatHistory={setChatHistory}
                            generateBotReplies={generateBotReplies} />
        </div>
    );
};

export default Chat;
