import  { GoogleGenerativeAI } from "@google/generative-ai";


const genAI = new GoogleGenerativeAI("AIzaSyAfx35VF3CKdHOAyEuQxP7RJwLKwdXLXX4");
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });


// const testRun=async ()=>{
//     const prompt="hello gemini "
//     const result = await model.generateContent(prompt);
//     console.log(result.response.text());
// }
// testRun();
// console.log(import.meta.env)

export default model

