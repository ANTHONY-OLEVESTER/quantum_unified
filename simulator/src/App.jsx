import React, { useState } from "react";
import StoryMode from "./story/StoryMode.jsx";
import SimulationMode from "./simulation/SimulationMode.jsx";

export default function App() {
  const [mode, setMode] = useState("story");

  if (mode === "story") {
    return <StoryMode onEnterSimulation={() => setMode("simulation")} />;
  }

  return <SimulationMode onBackToStory={() => setMode("story")} />;
}
