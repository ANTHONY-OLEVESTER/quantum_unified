import React, { useEffect } from "react";
import Storyboard from "./Storyboard.jsx";

export default function StoryMode({ onEnterSimulation }) {
  useEffect(() => {
    document.title = "Curvature–Information • Story Mode";
  }, []);

  return (
    <div className="min-h-screen bg-black text-white">
      <Storyboard onEnterSimulation={onEnterSimulation} />
    </div>
  );
}
