import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { IN_BROWSER } from "./api";
import { warmUp } from "./engine/client";
import "./index.css";
import { AppProvider } from "./state";

if (IN_BROWSER) warmUp(); // start loading the engine while the page renders

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppProvider>
        <App />
      </AppProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
