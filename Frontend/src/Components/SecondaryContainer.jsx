import {useNavigate} from "react-router-dom";

const SecondaryContainer=()=>{
    const navigate = useNavigate();
    return (
        <div
            className={'w-full px-6 md:px-24 md:py-16 bg-gradient-to-br from-green-950 via-green-800 to-green-700    text-white flex flex-col items-center justify-between text-center'}>
            <h1 className={'font-bold text-5xl text-center my-4'}>Get Started</h1>
            <div className={'flex items-center justify-between w-[95%] gap-5 my-6'}>
                <div
                    className=" flex flex-col items-center justify-between p-6 h-96 w-[45%] bg-gradient-to-br from-lime-700/10 via-lime-600/10 to-lime-500/10 backdrop-blur border border-green-600 rounded-xl">
                    <h1 className={'text-2xl font-bold'}>Soil Analysis</h1>
                    <img className={'h-[65%] '}
                         src="https://cdni.iconscout.com/illustration/premium/thumb/soil-analysis-illustration-download-in-svg-png-gif-file-formats--composition-ground-compost-check-various-themes-pack-miscellaneous-illustrations-7342817.png?f=webp"
                         alt="soil analysis illustration"
                    />
                    <button
                        onClick={() => navigate('/soil')}
                        className={'bg-green-600 px-7 py-3 rounded-full text-white font-bold cursor-pointer'}>Analyze
                    </button>
                </div>
                <div
                    className=" flex flex-col items-center justify-between p-6 h-96 w-[45%] bg-gradient-to-br from-lime-700/10 via-lime-600/10 to-lime-500/10 backdrop-blur border border-green-600 rounded-xl">
                    <h1 className={'text-2xl font-bold'}>Weather Prediction</h1>
                    <img className={'h-[65%] '}
                         src=
                             "https://i.ibb.co/rGcd01xz/weather-forecast-icon-clouds-and-sunshine-vector-removebg-preview.png"
                         alt="soil analysis illustration"
                    />
                    <button
                        onClick={() => navigate('/weather')}
                        className={'bg-green-600 px-7 py-3 rounded-full text-white font-bold cursor-pointer'}>Predict
                    </button>
                </div>

                <div
                    className=" flex flex-col items-center justify-between p-6 h-96 w-[45%] bg-gradient-to-br from-lime-700/10 via-lime-600/10 to-lime-500/10 backdrop-blur border border-green-600 rounded-xl">
                    <h1 className={'text-2xl font-bold'}>Pest Detection</h1>
                    <img className={'h-[100%] w-[100%]'}
                         src=
                             "https://i.ibb.co/996YkvLV/Untitled-design.png"
                         alt="soil analysis illustration"
                    />
                    <button
                        onClick={() => navigate('/pest')}
                        className={'bg-green-600 px-7 py-3 rounded-full text-white font-bold cursor-pointer'}>Detect
                    </button>
                </div>
            </div>
        </div>
    )
}
export default SecondaryContainer;