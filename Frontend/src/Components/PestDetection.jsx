import Navbar from "./Navbar.jsx";
import usePestDetection from "../Hooks/usePestDetection.js";

const PestDetection = () => {
    const { selectedFile, preview, responseData, loading, handleChange, handleUpload } = usePestDetection();

    return (
        <div className={'overflow-hidden relative h-screen w-full flex flex-col items-center justify-center bg-[url("https://extension.psu.edu/media/catalog/product/c/8/c821d56ca2f2fd3276c65fc06e9476a3.jpeg?quality=80&bg-color=255,255,255&fit=bounds&height=&width=&canvas=:")] bg-cover bg-center'}>
            <Navbar />
            <div className={'h-[70%] w-[53%] bg-gradient-to-br from-zinc-800/50 via-zinc-900/50 to-zinc-950/50 backdrop-blur-sm rounded-3xl border border-gray-500 p-10 flex flex-col items-center justify-between text-white'}>
                <h1 className={'text-3xl font-medium mb-3'}>Pest Detection</h1>
                <div className={'flex items-center w-full h-[85%]'}>
                    {/* Left Side - Upload Section */}
                    <div className={'h-full w-[50%] p-2.5 flex flex-col items-center justify-between'}>
                        <div className="border border-gray-500 bg-zinc-700/30 rounded-2xl w-[80%] h-[80%] flex flex-col items-center justify-center">
                            <h1 className={`${!selectedFile ? "opacity-100" : "opacity-0"} mb-4`}>Enter Image</h1>

                            {!selectedFile ? (
                                <>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        id="fileInput"
                                        onChange={handleChange}
                                        className="hidden"
                                    />
                                    <label
                                        htmlFor="fileInput"
                                        className="cursor-pointer border-2 text-white px-5 py-2 rounded-4xl text-sm hover:bg-white hover:text-black transition duration-200"
                                    >
                                        Upload Image
                                    </label>
                                </>
                            ) : (
                                <img className="w-32 h-32 object-cover rounded-lg mt-3" src={preview} alt="Preview" />
                            )}
                        </div>

                        <button
                            onClick={handleUpload}
                            className={'px-8 py-2 border-2 border-white rounded-full font-medium my-2 hover:bg-white hover:text-black'}
                            disabled={loading}
                        >
                            {loading ? "Detecting..." : "Detect Pest"}
                        </button>
                    </div>

                    {/* Right Side - Response Section */}
                    <div className={'h-full w-[50%] border-l-2 border-gray-500 flex flex-col items-center justify-center text-center'}>
                        {loading && <p className="text-white">Processing image...</p>}
                        {responseData && !loading && (
                            <div>
                                <h2 className="text-lg font-semibold">Detection Result:</h2>
                                {responseData.error ? (
                                    <p className="text-red-400">{responseData.error}</p>
                                ) : (
                                    <div className="mt-4">
                                        <p className="text-lg">
                                            {responseData.class_name ? (
                                                <>Detected: <span className="font-bold">{responseData.class_name}</span></>
                                            ) : (
                                                <>Detected class ID: {responseData.class_id}</>
                                            )}
                                        </p>
                                    
                                    </div>
                                )}
                            </div>
                        )}
                        {!responseData && !loading && (
                            <p className="text-gray-400">Upload an image to detect pests</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PestDetection;