import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import CurvedLoop from './components/CurvedLoop';
import FileUpload from './components/FileUpload';
import ResultsPage from './components/ResultsPage';
import GraphPage from './components/GraphPage';
import HypothesisPage from './components/HypothesisPage';
import SimulationPlanPage from './components/SimulationPlanPage';
import DatasetPage from './components/DatasetPage';
import CodePage from './components/CodePage';
import ErrorPage from './components/ErrorPage';
import ReportGeneratorPage from './components/ReportGeneratorPage';

function HomePage() {
  return (
    <div className="min-h-screen w-full relative flex items-center justify-center" style={{ backgroundColor: '#00A86B' }}>
      {/* CurvedLoop - Full screen background */}
      <div className="absolute w-full h-full flex flex-col items-center justify-center py-20" style={{ paddingTop: '180px' }}>
        {/* Top CurvedLoop - Faint, Opposite Direction */}
        <div className="w-full flex-shrink-0" style={{ height: '33.33vh', opacity: 0.1, marginBottom: '-15vh' }}>
          <CurvedLoop 
            marqueeText="Science Loop ✦"
            speed={3}
            curveAmount={0}
            direction="left"
            interactive={true}
          />
        </div>
        
        {/* Middle CurvedLoop */}
        <div className="w-full flex-shrink-0" style={{ height: '33.33vh' }}>
          <CurvedLoop 
            marqueeText="Science Loop ✦"
            speed={3}
            curveAmount={0}
            direction="right"
            interactive={true}
          />
        </div>
        
        {/* Bottom CurvedLoop - Faint, Opposite Direction */}
        <div className="w-full flex-shrink-0" style={{ height: '33.33vh', opacity: 0.1, marginTop: '-15vh' }}>
          <CurvedLoop 
            marqueeText="Loop ✦ Science"
            speed={3}
            curveAmount={0}
            direction="left"
            interactive={true}
            initialOffset={-200}
          />
        </div>
      </div>
      
      {/* File Upload - Overlay on top, right edge */}
      <div className="absolute right-0 top-0 bottom-0 flex items-center justify-center z-20 pointer-events-none" style={{ paddingTop: '10vh', paddingBottom: '10vh' }}>
        <div className="bg-white shadow-lg p-8 pointer-events-auto flex flex-col items-center justify-center overflow-hidden rounded-l-3xl" style={{ width: '45vw', height: '70vh', minHeight: '70vh' }}>
          <FileUpload />
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/hypothesis" element={<HypothesisPage />} />
          <Route path="/simulation-plan" element={<SimulationPlanPage />} />
          <Route path="/datasets" element={<DatasetPage />} />
          <Route path="/code" element={<CodePage />} />
          <Route path="/error" element={<ErrorPage />} />
          <Route path="/report" element={<ReportGeneratorPage />} />
        </Routes>
      </Router>
  );
}

export default App
