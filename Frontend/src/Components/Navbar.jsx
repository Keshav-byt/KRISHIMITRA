import {useNavigate} from "react-router-dom";

const Navbar=()=>{
    const navigate = useNavigate();
    return (
        <div className={'bg-gradient-to-b from-black text-white w-full z-40 fixed top-0 py-3.5 px-4 md:px-16 flex items-center justify-between'}>
            <div
                onClick={()=>navigate('/')}
                className={'flex items-center gap-2 cursor-pointer'}>
                <img
                    className={'w-12 h-12 md:w-16 md:h-16'}
                    src="https://i.ibb.co/V0zgCvRg/Krishi-Mitra.png" alt="KrishiMitra Logo"/>
                <h1 className={'text-xl md:text-2xl font-medium'}>KrishiMitra</h1>
            </div>
            <div className={'flex items-center space-x-6 md:space-x-12'}>
                <div className="group ">
                    <h1 onClick={() => navigate('/soil')}
                        className={'text-sm font-medium cursor-pointer transition-all duration-300 ease-in-out group-hover:text-green-500 shadow'}>
                        Soil Analysis
                    </h1>
                </div>
                <div className="group ">
                    <h1 onClick={() => navigate('/weather')}
                        className={'text-sm font-medium cursor-pointer transition-all duration-300 ease-in-out group-hover:text-green-500 shadow'}>
                        Irrigation Advice
                    </h1>
                </div>
                <div className="group ">
                    <h1 onClick={() => navigate('/pest')}
                        className={'text-sm font-medium cursor-pointer transition-all duration-300 ease-in-out group-hover:text-green-500 shadow'}>
                        Pest Detection
                    </h1>
                </div>
                <div className="group ">
                    <h1 onClick={() => navigate('/govt-schemes')}
                        className={'text-sm font-medium cursor-pointer transition-all duration-300 ease-in-out group-hover:text-green-500 shadow'}>
                        Govt. Schemes
                    </h1>
                </div>
            </div>
        </div>
    )
}
export default Navbar;