import govtSchemes from '../Utils/GovtSchemes.json';
import Navbar from "./Navbar.jsx";

const GovtSchemes = () => {
    return (
        <div className="relative min-h-screen overflow-hidden flex flex-col items-center justify-center px-6">
            <Navbar/>

            <div className="absolute inset-0 bg-slate-100 -z-10"></div>

            {/* Content Section */}
            <div className="relative z-10 w-full flex flex-col items-center">
                {/* Title */}
                <div className="text-center w-screen h-[80%]">
                    <a href="https://pmaymis.gov.in/PMAYMIS2_2024/PmayDefault.aspx"
                       rel="noopener noreferrer">
                        <img
                            src="https://pmay-urban.gov.in/uploads/sliders/web/6731a0f95941e-PMAY-U_2.0_2.jpg"
                            className="mb-10 cursor-pointer"
                            alt="PMAY Banner"
                        />
                    </a>
                </div>


                {/* Schemes Grid */}
                <div className="w-[95%] flex justify-center">
                    <div className="grid grid-cols-4 gap-8 max-w-8xl">
                        {govtSchemes.map((scheme, index) => (
                            <div key={index}
                                 className="bg-white shadow-lg rounded-lg p-4 flex flex-col items-center  transition-transform transform hover:scale-105">
                            <img
                                    src={scheme.image_link || "https://via.placeholder.com/150"}
                                    alt={scheme.title}
                                    className="w-full h-40 object-cover rounded-lg top-0"
                                />
                                <h2 className="text-l g font-bold mt-3 mb-2">{scheme.scheme_name}</h2>
                                <p className="text-gray-600 text-sm">{scheme.description}</p>
                                <a
                                    href={scheme.official_website}
                                    className="text-blue-500 rounded-xl mt-2 inline-block border-2 p-2"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    Learn More
                                </a>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GovtSchemes;
