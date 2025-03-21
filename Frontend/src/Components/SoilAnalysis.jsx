import Navbar from "./Navbar.jsx";
import {useState} from "react";
import useGetSoilAnalysis from "../Hooks/useGetSoilAnalysis.js";
import toast, {Toaster} from "react-hot-toast";
import AiSuggestions from "./AiSuggestions.jsx";

const SoilAnalysis = () => {
    const [isAnswered, setIsAnswered] = useState(false);
    const [result, setResult] = useState(null);

    const getSoilAnalysis = useGetSoilAnalysis();
    const soilMinerals = [
        ["Nitrogen", "N"],
        ["Phosphorus", "P"],
        ["Potassium", "K"],
        ["pH", "pH"],
        ["Electrical Conductivity", "EC"],
        ["Organic Carbon", "OC"],
        ["Sulfur", "S"],
        ["Zinc", "Zn"],
        ["Iron", "Fe"],
        ["Copper", "Cu"],
        ["Manganese", "Mn"],
        ["Boron", "B"]
    ];

    const [mineralValue, setMineralValue] = useState(
        soilMinerals.reduce((acc, mineral) => {
            acc[mineral[1]] = undefined;
            return acc;
        }, {})
    );

    const handleChange = (value, mineral) => {
        setMineralValue((prev) => ({
            ...prev,
            [mineral]: value
        }));
    };

    const handleSubmit = async () => {
        console.log(mineralValue);
        const response = await getSoilAnalysis(mineralValue);
        if (response) {
            setIsAnswered(true);
            setResult(response);
        }
    };

    return (
        <div className={' overflow-hidden relative h-screen w-full flex flex-col items-center justify-center bg-[url("https://t3.ftcdn.net/jpg/02/57/58/20/360_F_257582025_LUf6zGRPA0x0OGaLFS1UJIgkRKrrZhAk.jpg")] bg-cover bg-center'}>
            <Navbar />
            <Toaster/>
            <AiSuggestions isAnswered={isAnswered} result={result} minerals={mineralValue} />
            <div className={'h-[70%] w-[53%] bg-gradient-to-br from-zinc-800/50 via-zinc-900/50 to-zinc-950/50 backdrop-blur-sm rounded-3xl border border-gray-500 p-10 flex flex-col items-center justify-between text-white'}>
                <h1 className={'text-3xl font-medium mb-3'}>Soil Analysis</h1>
                <div className={'flex items-center w-full h-[85%]'}>
                    <div className={'h-full w-[50%] p-2.5 flex flex-col items-center justify-between'}>
                        <div className={'overflow-y-auto w-full h-[80%]'}>
                            {soilMinerals.map((m) => (
                                <div key={m[1]} className={'m-1.5 mb-3.5 text-sm'}>
                                    <label>{`Enter value for ${m[0]} `}</label>
                                    <input
                                        type="number"
                                        placeholder={m[1]}
                                        value={mineralValue[m[1]]}
                                        className={'w-[90%] border border-gray-500 rounded p-1 placeholder:text-sm'}
                                        onChange={(e) => handleChange(e.target.value, m[1])}
                                    />
                                </div>
                            ))}
                        </div>
                        <button
                            onClick={handleSubmit}
                            className={'px-8 py-2 border-2 border-white rounded-full font-medium my-2'}>
                            Get Fertility
                        </button>
                    </div>
                    <div className={'h-full w-[50%] border-l-2 border-gray-500 flex flex-col items-center justify-center text-center'}>
                        {!isAnswered ? (
                            <>
                                <img src="https://cdni.iconscout.com/illustration/free/thumb/free-searching-data-illustration-download-in-svg-png-gif-file-formats--no-content-yet-for-web-empty-states-pack-design-development-illustrations-3385493.png" alt="" />
                                <h1 className={'px-12 text-lg font-medium text-gray-400'}>
                                    Click on <span className={'font-bold'}>Get Fertility</span> To get the results
                                </h1>
                            </>
                        ) : (
                            <>
                                {result === "Low" && <>
                                    <img
                                        className={'h-[65%] rounded-full border-2 border-gray-500'}
                                        src="https://i.ibb.co/chS3jHFr/DALL-E-2025-03-14-23-13-19-A-modern-stylized-digital-illustration-representing-low-soil-fertility-Th.webp"
                                        alt="DALL-E-2025-03-14-23-13-19-A-modern-stylized-digital-illustration-representing-low-soil-fertility-Th"
                                        border="0"
                                    />
                                    <h1 className={' text-xl m-2 font-medium'}>Soil Fertility is :
                                        <span className={'font-bold text-red-700'}> Low</span>
                                    </h1>
                                </>
                                }
                                {result === "High" &&
                                    <>
                                        <img
                                            className={'h-[65%] rounded-full border-2 border-gray-500'}
                                            src="https://i.ibb.co/jPFVPC13/DALL-E-2025-03-14-23-13-13-A-modern-stylized-digital-illustration-representing-high-soil-fertility-T.webp"
                                            alt="DALL-E-2025-03-14-23-13-13-A-modern-stylized-digital-illustration-representing-high-soil-fertility-T"
                                            border="0"
                                        />
                                        <h1 className={' text-xl m-2 font-medium'}>Soil Fertility is :
                                            <span className={'font-bold text-green-600'}> High</span>
                                        </h1>
                                    </>}
                                {result === "missing" &&
                                    <>
                                        <img
                                            className={'h-[65%] '}
                                            src="https://cdni.iconscout.com/illustration/premium/thumb/no-data-found-illustration-download-in-svg-png-gif-file-formats--missing-error-business-pack-illustrations-8019228.png?f=webp"
                                            alt="DALL-E-2025-03-14-23-13-13-A-modern-stylized-digital-illustration-representing-high-soil-fertility-T"
                                            border="0"
                                        />
                                        <h1 className={' text-xl m-2 font-medium'}>
                                            Soil data is missing. Please enter all values.
                                        </h1>
                                    </>}
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
export default SoilAnalysis