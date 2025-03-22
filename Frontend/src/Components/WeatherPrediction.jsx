import Navbar from "./Navbar.jsx";
import { Toaster } from "react-hot-toast";
import { useState } from "react";
import usePredictWeather from "../Hooks/usePredictWeather.js";

const WeatherPrediction = () => {
    const [isAnswered, setIsAnswered] = useState(false);
    const [result, setResult] = useState(null);
    const [soilColour, setSoilColour] = useState([]); // Default to empty string instead of undefined
    const [cityName, setCityName] = useState([]); // Default to empty string instead of undefined

    // Call the custom hook
    const getWeather = usePredictWeather();

    // Combine soilColour and cityName into an object or array to send to the hook
    const inputValue = { soilColour, cityName };  // Using an object instead of an array

    console.log(inputValue);

    const handleSubmit = async () => {
        console.log(inputValue );

        // Make sure to pass the values to the custom hook
        const inputValues = {soil_color:soilColour , city:cityName};
        const res = await getWeather(inputValues);
        console.log(res);
        if (res) {
            setIsAnswered(true);
            setResult(res); // Set the result
        }
    };

    return (
        <div className={'h-screen w-full flex flex-col items-center justify-center bg-[url(https://img.freepik.com/premium-photo/irrigation-plant-system-field-agriculture-plants_1257223-148232.jpg)] bg-cover bg-center'}>
            <Navbar />
            <Toaster />
            <div className={'h-[70%] w-[53%] bg-gradient-to-br from-zinc-800/50 via-zinc-900/50 to-zinc-950/50 backdrop-blur-sm rounded-3xl border border-gray-500 p-10 flex flex-col items-center justify-between text-white'}>
                <h1 className={'text-3xl font-medium mb-3'}>Irrigation Advice</h1>
                <div className={'flex items-center w-full h-[85%]'}>
                    <div className={'h-full w-[50%] p-2.5 flex flex-col items-center justify-between'}>
                        <div className={'overflow-y-auto w-full h-[80%]'}>
                            <div className={'m-1.5 mb-3.5 text-sm'}>
                                <label>{`Enter colour of soil `}</label>
                                <input
                                    type="text"
                                    placeholder={'Soil Colour'}
                                    value={soilColour}
                                    className={'w-[90%] border border-gray-200 rounded p-1 placeholder:text-sm'}
                                    onChange={(e) => setSoilColour(e.target.value)} // Update soilColour state
                                />
                            </div>

                            <div className={'m-1.5 mb-3.5 text-sm'}>
                                <label>{`Enter city name `}</label>
                                <input
                                    type="text"
                                    placeholder={'City'}
                                    value={cityName}
                                    className={'w-[90%] border border-gray-200 rounded p-1 placeholder:text-sm'}
                                    onChange={(e) => setCityName(e.target.value)} // Update cityName state
                                />
                            </div>
                        </div>
                        <button
                            onClick={handleSubmit}
                            className={'px-8 py-2 border-2 border-white rounded-full font-medium my-2'}>
                            Give Me Advice
                        </button>
                    </div>
                    <div className={'h-full w-[50%] border-l-2 border-gray-200 flex flex-col items-center justify-center text-center'}>
                        { !isAnswered ? (
                            <>
                                <img src="https://cdni.iconscout.com/illustration/free/thumb/free-searching-data-illustration-download-in-svg-png-gif-file-formats--no-content-yet-for-web-empty-states-pack-design-development-illustrations-3385493.png" alt="" />
                                <h1 className={'px-12 text-lg font-medium text-gray-300'}>
                                    Click on <span className={'font-bold'}>Give Me Advice</span> To get the suggestions
                                </h1>
                            </>
                        ) : (
                            <>
                                {/* Display the result once it is answered */}
                                <h1 className={'px-12 text-lg font-medium text-white'}>
                                    Temperature: <span className={'font-bold'}>{result?.temperature}</span>
                                </h1>
                                <h1 className={'px-12 text-lg font-medium text-white'}>
                                    Humidity:<span className={'font-bold'}>{result?.humidity}</span>
                                </h1>
                                <h1 className={'px-12 text-lg font-medium text-white'}>
                                    Crop suggestion: <span className={'font-bold'}>{result?.recommended_crop}</span>
                                </h1>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WeatherPrediction;
