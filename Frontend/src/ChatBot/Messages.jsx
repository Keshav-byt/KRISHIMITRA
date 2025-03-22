const Messages = ({chat}) => {
    console.log(chat);
    return (
        <>
            {chat.role !== "user" ? (<div className="flex justify-start">
                <div className="bg-gradient-to-r from-violet-600 to-indigo-600  p-2 rounded-lg rounded-bl-none max-w-[80%]">
                    {chat.text}

                </div>
            </div>) : (<div className="flex justify-end">
                <div className="bg-blue-500 text-white p-2 rounded-lg rounded-br-none max-w-[80%]">
                    {chat.text}
                </div>
            </div>)}


            {/* User Message (Right Side) */}

        </>

    )
}
export default Messages;