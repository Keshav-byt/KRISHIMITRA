import govtSchemes from '../Utils/GovtSchemes.json';
import Navbar from "./Navbar.jsx";

const GovtSchemes = () => {
    return (
        <div className="relative min-h-screen overflow-hidden flex flex-col items-center justify-center px-6 ">
            <Navbar/>

            {/* Background */}
            <div className="absolute inset-0 bg-slate-200 -z-10 "></div>

            {/* Content Section */}
            <div className="relative z-10 flex flex-col items-center w-[85%] m-auto">
                {/* Banner */}
                <div className="text-center ">
                    <a href="https://pmaymis.gov.in/PMAYMIS2_2024/PmayDefault.aspx" rel="noopener noreferrer">
                        <img
                            src="https://pmay-urban.gov.in/uploads/sliders/web/6731a0f95941e-PMAY-U_2.0_2.jpg"
                            className="mb-10 cursor-pointer w-full rounded-lg shadow-md"
                            alt="PMAY Banner"
                        />
                    </a>
                </div>

                {/* Schemes Grid */}
                <div className="w-full max-w-7xl flex justify-center">
                    <div className="grid grid-cols-3 gap-8 w-full px-4">
                        
                        {govtSchemes.map((scheme, index) => (
                            <div
                            key={index}
                            className="relative w-full min-h-[250px] rounded-lg overflow-hidden group shadow-md "
                            style={{
                                backgroundImage: `url(${scheme?.image_link || "https://via.placeholder.com/150"})`,
                                backgroundSize: "cover",
                                backgroundPosition: "center",
                                backgroundRepeat: "no-repeat"
                            }}
                            >
                                {/* Dark Overlay */}
                                {/* Dark Overlay (Reduce Opacity) */}
                                <div className="absolute inset-0 bg-gradient-to-br from-black transition-opacity group-hover:backdrop-blur-[5px]"></div>

                            
                                {/* Scheme Name (Fade Out on Hover) */}
                                <h2 className="absolute inset-0 flex items-center w-[70%] ml-2 justify-center text-white text-lg font-bold transition-opacity group-hover:opacity-0">
                                    {scheme.scheme_name}
                                </h2>
                            
                                {/* Hidden Content (Fade In on Hover) */}
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-4 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300 ease-in-out">
                                    <p className="text-sm mb-2">{scheme.description}</p>
                                    <a
                                        href={scheme.official_website}
                                        className="bg-blue-500 px-4 py-2 rounded-lg shadow-md hover:bg-blue-600 transition-colors"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        Learn More
                                    </a>
                                </div>
                            </div>
                        
                        ))}

                    </div>
                </div>
            </div>
        </div>
    );
};

export default GovtSchemes;
