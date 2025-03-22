import {toast} from 'react-hot-toast'
const usePredictWeather=()=>{
    const API_BASE_URL = 'http://localhost:5000';
    const getWeather=async (inputValues)=>{
        console.log(inputValues)
        try{
            // if (inputValues.length !== 2) {
            //     const errorMessage = 'Expected both soilColour and cityName inputs';
            //     throw new Error(errorMessage);
            // }
            const response=await fetch(`${API_BASE_URL}/predict-crop`,{
                method: 'POST',
                headers:{
                    'Content-Type': 'application/json'

                },
                body:JSON.stringify(inputValues)
            })
            if(!response.ok){
                const errorData=await response.json();
                throw new Error(errorData.error || 'Weather prediction failed. Please try again.');

            }
            const data=await response.json();
            // console.log(data)
            return data
        }
        catch (error){
            toast(error.message || 'An unexpected error occurred.')
            console.error('Weather prediction error:', error);
        }
    }
    return getWeather;

}
export default usePredictWeather;