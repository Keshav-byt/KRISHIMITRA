import { useState } from "react";

const usePestDetection = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [responseData, setResponseData] = useState(null);
    const [loading, setLoading] = useState(false);
    
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

    // Handle file selection
    const handleChange = (event) => {
        const file = event.target.files[0];

        if (!file) return;

        // Validate file type
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file.");
            return;
        }

        // Validate file size
        if (file.size > MAX_FILE_SIZE) {
            alert("File size exceeds 5MB. Please choose a smaller file.");
            return;
        }

        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
        setResponseData(null); // Clear previous response
    };

    // Handle file upload
    const handleUpload = async () => {
        if (!selectedFile) {
            alert("Please select an image first.");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);
        
        try {
            setLoading(true);
            const response = await fetch("http://localhost:5000/pest-detection", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log("Response data:", data);
            setResponseData(data);
        } catch (error) {
            console.error("Upload error:", error);
            alert(`Failed to upload image: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    return { selectedFile, preview, responseData, loading, handleChange, handleUpload };
};

export default usePestDetection;