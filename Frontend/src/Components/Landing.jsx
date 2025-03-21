import Navbar from "./Navbar.jsx";
import SecondaryContainer from "./SecondaryContainer.jsx";
import Footer from "./Footer.jsx";
import Carousel from "./Carousel.jsx";
import Chat from "../ChatBot/Chat.jsx";
import {useState} from "react";

const Landing=()=>{
    const[botDisplay,setBotDisplay]=useState(false);
    const handleChatDisplay = (p) => {
        setBotDisplay(p);
    }
    return (
        <div className={'min-h-screen max-w-screen  text-white overflow-x-hidden'}>
            <Navbar/>
            <div className="fixed bottom-10 right-10 flex  z-50 " >
                {botDisplay ? <Chat handleChatDisplay={handleChatDisplay} /> :
                (<div className="bg-green-400 w-[3em] rounded-full cursor-pointer"
                      onClick={() => setBotDisplay(prev => !prev)}>
                    <img src="https://www.svgrepo.com/show/339963/chat-bot.svg"  className="p-2"/>
                </div>)}

            </div>

            <div className="w-full aspect-video pt-[20%] flex flex-col items-center  absolute text-white bg-gradient-to-t from-black">
                <h1 className=" text-2xl md:text-7xl font-bold font-montserrat">Krishiमित्र</h1>
                <p className="hidden md:inline-block py-6 text-lg w-1/2 text-center">Boost your harvest with our cutting-edge soil analysis model, designed to optimize fertility, paired with an advanced pest detection system to protect your crops. Precision farming starts here!</p>
            </div>

            <div className="-my-4 max-w-screen">
                <iframe
                    className="w-full aspect-video "
                    src={"https://www.youtube.com/embed/kQWAXuP0pik?&autoplay=1&mute=1&loop=1&controls=0&playlist=kQWAXuP0pik"}
                    title="video player"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; "
                    referrerPolicy="strict-origin-when-cross-origin" >
                </iframe>
            </div>
            <Carousel/>
            <SecondaryContainer/>
            <Footer/>

        </div>
    )
}
export default Landing