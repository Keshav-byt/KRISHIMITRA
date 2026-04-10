
import './App.css'
import Landing from "./Components/Landing.jsx";
import {createBrowserRouter, RouterProvider} from "react-router-dom";
import SoilAnalysis from "./Components/SoilAnalysis.jsx";
import GovtSchemes from "./Components/GovtSchemes.jsx";

function App() {

    const appRouter=createBrowserRouter([
        {
            path: "/",
            element: <Landing/>
        },
        {
            path:"/soil",
            element: <SoilAnalysis/>
        },
        {
            path:"/govt-schemes",
            element: <GovtSchemes/>
        }
    ])
  return (
    <div>
        <RouterProvider router={appRouter}>
        </RouterProvider>
    </div>
  )
}

export default App;
