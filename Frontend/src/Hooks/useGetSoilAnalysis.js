import {toast} from 'react-hot-toast'

const useGetSoilAnalysis=()=>{
    const API_BASE_URL = 'http://localhost:5000';
    const getSoilAnalysis = async (mineralValue) => {
        try{
            const response=await fetch (`${API_BASE_URL}/soil-analysis`,{
                method: "POST",
                headers:{
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(mineralValue)
            })
            if (!response.ok) {
                const errorData = await response.json();
                const errorMessage=errorData.error || 'Soil analysis failed. Please try again.'
                // toast.error(errorMessage)
                throw new Error(errorMessage);
            }
            const data=await response.json();
            // console.log(data)

            if(data && data.fertility_status){
                console.log(data.fertility_status)
                return data.fertility_status
            }
            else{
                console.log('Soil Fertility: Classification data is missing')
                toast.error('Soil Fertility: Classification data is missing')
                return "missing"
            }
        }
        catch(error){
            console.error('Soil analysis error:', error);
            toast.error(''+ error)
        }
    }
    return getSoilAnalysis;
}
export default useGetSoilAnalysis;