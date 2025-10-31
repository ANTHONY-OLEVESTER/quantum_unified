import React, { useEffect } from "react";
import Storyboard from "./Storyboard.jsx";

export default function StoryMode({ onEnterSimulation }) {
  useEffect(() => {
    document.title = "Curvature–Information • Story Mode";
  }, []);

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="fixed top-4 right-4 z-[1100] flex gap-2">
        <button
          className="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-white text-sm border border-white/20"
          onClick={onEnterSimulation}
        >
          Open Simulation Mode
        </button>
      </div>
      <Storyboard onEnterSimulation={onEnterSimulation} />
    </div>
  );
}
