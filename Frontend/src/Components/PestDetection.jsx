import Navbar from "./Navbar.jsx";
import { useState } from "react";

const PestDetection = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);

    // Handle file selection
    const handleChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);  // Store file for upload
            setPreview(URL.createObjectURL(file));  // Generate preview
        }
    };

    // Handle file upload
    const handleUpload = async () => {
        if (!selectedFile) {
            alert("Please select an image first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("http://localhost:5000/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            console.log("Upload Response:", data);
            alert("Image uploaded successfully!");
        } catch (error) {
            console.error("Upload error:", error);
            alert("Failed to upload image.");
        }
    };

    return (
        <div className={'overflow-hidden relative h-screen w-full flex flex-col items-center justify-center bg-[url("https://extension.psu.edu/media/catalog/product/c/8/c821d56ca2f2fd3276c65fc06e9476a3.jpeg?quality=80&bg-color=255,255,255&fit=bounds&height=&width=&canvas=:")] bg-cover bg-center'}>
            <Navbar />
            <div className={'h-[70%] w-[53%] bg-gradient-to-br from-zinc-800/50 via-zinc-900/50 to-zinc-950/50 backdrop-blur-sm rounded-3xl border border-gray-500 p-10 flex flex-col items-center justify-between text-white'}>
                <h1 className={'text-3xl font-medium mb-3'}>Pest Detection</h1>
                <div className={'flex items-center w-full h-[85%]'}>
                    <div
                        className={'h-full w-[50%] p-2.5 flex flex-col items-center justify-between'}>
                        <div
                            className="border border-gray-500 bg-zinc-700/30 rounded-2xl w-[80%] h-[80%] flex flex-col items-center justify-center">
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
                                <img
                                    className="w-32 h-32 object-cover rounded-lg mt-3"
                                    src={preview}
                                    alt="Preview"
                                />
                            )}
                        </div>


                        <button
                            onClick={handleUpload}
                            className={'px-8 py-2 border-2 border-white rounded-full font-medium my-2 hover:bg-white hover:text-black'}>
                            Detect Pest
                        </button>
                    </div>

                    <div
                        className={'h-full w-[50%] border-l-2 border-gray-500 flex flex-col items-center justify-center text-center'}>
                        {/* Result Section (Optional) */}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PestDetection;
