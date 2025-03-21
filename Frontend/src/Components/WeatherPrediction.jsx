import Navbar from "./Navbar.jsx";
import {Toaster} from "react-hot-toast";
import {useState} from "react";
import usePredictWeather from "../Hooks/usePredictWeather.js";


const WeatherPrediction=()=>{
    const [isAnswered, setIsAnswered] = useState(false);
    const [result, setResult] = useState(null);
    const[tempValue,setTempValue]=useState(undefined);
    const[humidity,setHumidity]=useState(undefined);
    const[windSpeed,setWindSpeed]=useState(undefined);

    const getWeather=usePredictWeather()

    const inputValue=[tempValue,humidity,windSpeed]

    const handleSubmit=async ()=>{
        console.log(inputValue)
        const res=await getWeather(inputValue);
        if(res){
            setIsAnswered(true)
            setResult(res)
        }

    }

    return (
        <div className={'h-screen w-full flex flex-col items-center justify-center bg-[url(https://images.pexels.com/photos/414659/pexels-photo-414659.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500)] bg-cover bg-center'}>
            <Navbar/>
            <Toaster/>
            <div className={'h-[70%] w-[53%] bg-gradient-to-br from-zinc-800/50 via-zinc-900/50 to-zinc-950/50 backdrop-blur-sm rounded-3xl border border-gray-500 p-10 flex flex-col items-center justify-between text-white'}>
                <h1 className={'text-3xl font-medium mb-3'}>Weather Prediction</h1>
                <div className={'flex items-center w-full h-[85%]'}>
                    <div className={'h-full w-[50%] p-2.5 flex flex-col items-center justify-between'}>
                        <div className={'overflow-y-auto w-full h-[80%]'}>
                            <div  className={'m-1.5 mb-3.5 text-sm'}>
                                <label>{`Enter value for Temperature `}</label>
                                <input
                                    type="number"
                                    placeholder={'Temperature'}
                                    value={tempValue}
                                    className={'w-[90%] border border-gray-200 rounded p-1 placeholder:text-sm'}
                                    onChange={(e) =>setTempValue(e.target.value)}
                                />
                            </div>
                            <div  className={'m-1.5 mb-3.5 text-sm'}>
                                <label>{`Enter value for  Humidity`}</label>
                                <input
                                    type="number"
                                    placeholder={'Humidity'}
                                    value={humidity}
                                    className={'w-[90%] border border-gray-200 rounded p-1 placeholder:text-sm'}
                                    onChange={(e) => setHumidity(e.target.value)}
                                />
                            </div>
                            <div  className={'m-1.5 mb-3.5 text-sm'}>
                                <label>{`Enter value for Wind Speed `}</label>
                                <input
                                    type="number"
                                    placeholder={'Wind Speed'}
                                    value={windSpeed}
                                    className={'w-[90%] border border-gray-200 rounded p-1 placeholder:text-sm'}
                                    onChange={(e) => setWindSpeed(e.target.value)}
                                />
                            </div>
                        </div>
                        <button
                            onClick={handleSubmit}
                            className={'px-8 py-2 border-2 border-white rounded-full font-medium my-2'}>
                            Predict Weather
                        </button>
                    </div>
                    <div className={'h-full w-[50%] border-l-2 border-gray-200 flex flex-col items-center justify-center text-center'}>
                        { !isAnswered ? (<>
                            <img src="https://cdni.iconscout.com/illustration/free/thumb/free-searching-data-illustration-download-in-svg-png-gif-file-formats--no-content-yet-for-web-empty-states-pack-design-development-illustrations-3385493.png" alt="" />
                            <h1 className={'px-12 text-lg font-medium text-gray-300'}>
                                Click on <span className={'font-bold'}>Predict Weather</span> To get the results
                            </h1>
                        </>):
                        <>
                            <img src="https://assets.streamlinehq.com/image/private/w_240,h_240,ar_1/f_auto/v1/icons/illustrations-multicolor/weather/weather/weather-forecast-reporter-6-3e4xsjlz30fwcoilq7w7xh.png?_a=DAJFJtWIZAAC" alt="" />
                            <h1 className={'px-12 text-lg font-medium text-white'}>
                                Predicted weather is <span className={'font-bold'}>{result}</span>
                            </h1>
                        </>}
                    </div>
                </div>
            </div>
        </div>
    )
}
export default WeatherPrediction