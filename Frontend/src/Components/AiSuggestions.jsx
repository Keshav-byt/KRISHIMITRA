import {useEffect, useState} from "react";
import model from "../Utils/gemini.js";

const AiSuggestions=({isAnswered,result,minerals})=>{


    const [isClicked, setIsClicked] = useState(false);
    const [message, setMessage] = useState("");

    console.log(minerals)
    const getAiSuggestions=async ()=>{
        const prompt=`Act as a soil fertility improvement model these are the quantity and the particular minerals  ${JSON.stringify(minerals)} and this is the result of fertility ${result} give suggestion/feedback
         based on the quantity and the result of the soil in strictly 20-30 words`
        console.log(prompt)
        const res = await model.generateContent(prompt);
        setMessage(res.response.text())

    }
    useEffect(()=>{
        getAiSuggestions();
    },[isAnswered])

    return (
        <>
            <div
                onClick={() => setIsClicked(!isClicked)}
                className={`${!isClicked? "":"hidden"} ${isAnswered ? "translate-x-0 " : "translate-x-48"}  transition-transform ease-in-out duration-300 cursor-pointer absolute bottom-4 right-0 h-12 w-48 text-white bg-gradient-to-br from-[#FF5733] via-[#8E44AD] to-[#3498DB]  rounded-l-3xl z-10  flex items-center justify-center`}>
                <h1 className={' font-bold'}>Get AI Suggestion
                </h1>
            </div>
            <div
                className={`${isClicked ? "translate-x-0" : "translate-x-100"} ${isAnswered ? " " : ""}  transition-transform ease-in-out duration-300 h-100 w-88 z-20 rounded-l-2xl bg-gradient-to-br from-[#FF5733] via-[#8E44AD] to-[#3498DB]  absolute bottom-2 right-0 p-1.5 pr-0`}>
                <div className=" h-full w-full bg-gradient-to-br  backdrop-blur-2xl from-zinc-800/80 via-zinc-900/80  to-zinc-950/80 rounded-l-xl p-4 flex-col flex items-center justify-center  text-center relative">
                    <h1 className={'text-xl text-white font-medium m-2'}>AI Suggestions To Improve Soil Fertility</h1>
                    <p  className={'text-gray-400 m-2 text-sm'}>{message}</p>
                    <button
                        onClick={() => setIsClicked(!isClicked)}
                        className={'absolute bottom-4 right-4 text-gray-600 font-bold cursor-pointer'}>Hide</button>
                </div>
            </div>

        </>

    )
}
export default AiSuggestions;