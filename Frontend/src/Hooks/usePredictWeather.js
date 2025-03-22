import {toast} from 'react-hot-toast'
const usePredictWeather=()=>{
    const API_BASE_URL = 'http://localhost:5000';
    const getWeather=async (inputValues)=>{
        try{
            if(inputValues.length!==3){
                const errorMessage=`Expected 3 input fields, but found ${inputValues.length}`
                throw Error(errorMessage)

            }
            const response=await fetch(`${API_BASE_URL}/weather-prediction`,{
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
            return ` ${data.predicted_temperature.toFixed(2)} °C`
        }
        catch (error){
            toast(error.message || 'An unexpected error occurred.')
            console.error('Weather prediction error:', error);
        }
    }
    return getWeather;

}
export default usePredictWeather;