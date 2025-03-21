
import  { useState } from 'react';

const Carousel = () => {
    const [currentIndex, setCurrentIndex] = useState(0);

    const slides = [
        {
            id: 1,
            title: "Our mission",
            content: "Krishi Mitra boosts farm yields by 20-30% while reducing costs and resource waste. It enhances sustainability by cutting water use by 50% and pesticide use by 30%. By providing AI-driven insights to small farmers, it strengthens food security and pest detection. Scalable and future-ready, krishimitra paves the way for a tech-driven, sustainable agriculture system.",
            backgroundImage:'https://images.pexels.com/photos/11733071/pexels-photo-11733071.jpeg'

        },
        {
            id: 2,
            title: "Our Approach",
            content: "Our AI-powered web platform provides real-time insights for smarter farming. It detects pests, predicts yields, and optimizes planting and irrigation. With a user-friendly, offline-accessible interface, it supports farmers in all regions. By reducing water and pesticide waste, it promotes sustainable agriculture. Scalable and cost-effective, it benefits farms of all sizes. This approach maximizes productivity while ensuring environmental sustainability.",
            backgroundImage: 'https://t4.ftcdn.net/jpg/02/03/53/83/360_F_203538316_fwVfdJorJuZQQum1F47ghOipEF87yMgT.jpg'
        },
        {
            id: 3,
            title: "Implementation strategy",
            content: "Our implementation strategy begins with developing and testing the AI-powered web platform, ensuring accuracy and usability through pilot programs with farmers.Continuous feedback-driven improvements refine AI models and enhance platform features. As adoption grows, we scale the solution to diverse regions, adapting it to different crops and farming conditions. This structured approach ensures accessibility, efficiency, and long-term impact in transforming agriculture with AI.",
            backgroundImage: 'https://t4.ftcdn.net/jpg/02/10/87/61/360_F_210876109_BckfP4EoATsce9Orj9lceyEHkTRWn70p.jpg'
        },

    ];

    const goToPrevious = () => {
        const newIndex = (currentIndex - 1 + slides.length) % slides.length;
        setCurrentIndex(newIndex);
    };

    const goToNext = () => {
        const newIndex = (currentIndex + 1) % slides.length;
        setCurrentIndex(newIndex);
    };


    const prevSlide = slides[(currentIndex - 1 + slides.length) % slides.length];
    const currentSlide = slides[currentIndex];
    const nextSlide = slides[(currentIndex + 1) % slides.length];

    return (
        <div className={`w-full overflow-hidden bg-black relative z-10 h-120 text-white `}>

            <div
                className="absolute inset-0 bg-cover bg-center transition-opacity duration-500 h-full w-full opacity-55"
                style={{ backgroundImage: `url(${currentSlide.backgroundImage})` }}
            ></div>
            <div className="relative h-full flex items-center justify-center">

                <div className="flex w-full h-full items-center justify-center px-12 overflow-hidden ">
                    {/*left*/}
                    <div className="hidden md:flex w-280 h-84 items-center justify-center transform -translate-x-165">
                        <div className="bg-white bg-opacity-90 p-12 rounded shadow-lg w-full h-full flex flex-col items-center justify-center text-center">
                            <h2 className="text-4xl font-medium text-zinc-800 mb-4">{prevSlide.title}</h2>
                            <p className="text-gray-600 text-lg">{prevSlide.content}</p>
                        </div>
                    </div>

                    {/*middle*/}
                    <div className="absolute w-full md:w-260 h-84 flex items-center justify-center z-10 px-4">
                        <div className="bg-white bg-opacity-90 p-12 rounded shadow-xl w-full h-full flex flex-col items-center justify-center text-center ">
                            <h2 className="text-4xl font-medium text-zinc-800 mb-4">{currentSlide.title}</h2>
                            <p className="text-gray-600 text-lg">{currentSlide.content}</p>
                        </div>
                    </div>

                    {/*right*/}
                    <div className="hidden md:flex w-280 h-84 items-center justify-center transform translate-x-165">
                        <div className="bg-white bg-opacity-90 p-12 rounded shadow-lg w-full h-full flex flex-col items-center justify-center text-center">
                            <h2 className="text-4xl font-medium text-zinc-800 mb-4">{nextSlide.title}</h2>
                            <p className="text-gray-600 text-lg">{nextSlide.content}</p>
                        </div>
                    </div>
                </div>


                <button
                    onClick={goToPrevious}
                    aria-label="Previous slide"
                    className="absolute left-8 text-white bg-blue-700/50 hover:bg-blue-700 w-10 h-10  rounded-full  transition-all"
                >
                    &larr;
                </button>
                <button
                    onClick={goToNext}
                    aria-label="Next slide"
                    className="absolute right-8 text-white bg-blue-700/50 hover:bg-blue-700 w-10 h-10  rounded-full  transition-all"
                >
                    &rarr;
                </button>


                <div className="absolute bottom-4 flex space-x-2 justify-center items-center">
                    {slides.map((_, index) => (
                        <button
                            key={index}
                            onClick={() => setCurrentIndex(index)}
                            className={` rounded-full ${index === currentIndex ? 'bg-white w-3 h-3' : 'bg-gray-400 w-1.5 h-1.5'}`}
                            aria-label={`Go to slide ${index + 1}`}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Carousel;