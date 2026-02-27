import React, { useState, useEffect, useRef } from 'react';
import Confetti from 'react-confetti';
import { reviewCode, translateCode } from '../services/api';
import CodeBlock from '../components/CodeBlock';

/**
 * AnimatedNumber Component
 * Provides a smooth count-up animation for performance scores.
 */
const AnimatedNumber = ({ target }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = parseInt(target) || 0;
    if (end <= 0) {
      setCount(0);
      return;
    }
    const duration = 1500;
    const incrementTime = Math.max(duration / end, 10);
    
    const timer = setInterval(() => {
      start += 1;
      setCount(start);
      if (start >= end) clearInterval(timer);
    }, incrementTime);
    
    return () => clearInterval(timer);
  }, [target]);

  return <span>{count}</span>;
};

export default function Editor() {
  // --- STATE ---
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [readingFile, setReadingFile] = useState(false);
  const [error, setError] = useState(null);
  const [response, setResponse] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('');
  const [translatedCode, setTranslatedCode] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [isShaking, setIsShaking] = useState(false);
  const [windowSize, setWindowSize] = useState({ width: window.innerWidth, height: window.innerHeight });

  // --- WINDOW RESIZE ---
  useEffect(() => {
    const handleResize = () => setWindowSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // --- STYLING LOGIC ---
  const getGlassyBorderStyle = (gradientColors, glowColor) => ({
    position: 'relative',
    borderRadius: '16px',
    padding: '2px', // Thin, elegant border stroke
    background: `linear-gradient(135deg, ${gradientColors})`,
    boxShadow: `0 10px 40px -15px ${glowColor}`,
    animation: 'borderGradientShift 10s ease infinite alternate',
    width: '100%',
    transition: 'all 0.3s ease'
  });

  const getGlassyContentStyle = () => ({
    backgroundColor: '#0a0a0f', // Dark, uniform core for clarity
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    borderRadius: '14px',
    padding: '30px',
    height: '100%',
    display: 'flex',
    flexDirection: 'column'
  });

  // --- HANDLERS ---
  const handleFile = (e) => {
    const file = e.target.files[0];
    if (file) {
      setReadingFile(true);
      const reader = new FileReader();
      reader.onload = (event) => {
        setText(event.target.result);
        setReadingFile(false);
      };
      reader.readAsText(file);
    }
  };

  const handleRefine = async () => {
    if (!text.trim()) {
      setError('Please provide code input.');
      return;
    }

    setLoading(true);
    setError(null);
    setResponse(null);
    setTargetLanguage('');
    setTranslatedCode('');

    try {
      const result = await reviewCode(text);
      setResponse(result);

      if (result.refined_score >= 95) {
        setShowConfetti(true);
        setTimeout(() => setShowConfetti(false), 5000);
      }
      
      if (result.original_score < 50 || result.bugs?.toLowerCase().includes('critical')) {
        setIsShaking(true);
        setTimeout(() => setIsShaking(false), 500);
      }
    } catch (err) {
      setError('Communication Failure: Backend is unreachable.');
    } finally {
      setLoading(false);
    }
  };

  const handleTranslate = async (e) => {
    const selectedLang = e.target.value;
    setTargetLanguage(selectedLang); 
    if (!selectedLang) {
      setTranslatedCode('');
      return;
    }

    setIsTranslating(true);
    try {
      const result = await translateCode(response.optimized_code, selectedLang);
      setTranslatedCode(result.translated_code);
    } catch (err) {
      setTranslatedCode('// Error occurred during translation.');
    } finally {
      setIsTranslating(false);
    }
  };

  const isActive = loading || response;

  return (
    <div className={isShaking ? 'shake-effect' : ''} style={{ 
      minHeight: '100vh', 
      paddingTop: '3rem', 
      paddingBottom: '5rem',
      backgroundColor: '#050508',
      backgroundImage: `
        radial-gradient(circle at 10% 10%, rgba(67, 56, 202, 0.2) 0%, transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(126, 34, 206, 0.15) 0%, transparent 40%)
      `,
      backgroundAttachment: 'fixed',
      color: '#f8fafc',
      fontFamily: 'Inter, sans-serif'
    }}>
      
      <style>{`
        @keyframes borderGradientShift { 0% { filter: hue-rotate(0deg); } 100% { filter: hue-rotate(20deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .shake-effect { animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both; }
        @keyframes shake { 10%, 90% { transform: translate3d(-1px, 0, 0); } 20%, 80% { transform: translate3d(2px, 0, 0); } }
        .code-block-container::-webkit-scrollbar { width: 8px; }
        .code-block-container::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
        textarea::placeholder { color: #4b5563; }
      `}</style>

      {showConfetti && <Confetti width={windowSize.width} height={windowSize.height} recycle={false} numberOfPieces={500} />}

      <div style={{ maxWidth: '1200px', width: 'calc(100% - 2cm)', margin: '0 auto' }}>
        
        <header style={{ marginBottom: '40px', textAlign: 'center' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: '800', letterSpacing: '-0.04em', background: 'linear-gradient(to bottom, #fff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            CodeRefine <span style={{ color: '#6366f1' }}>Workspace</span>
          </h1>
          <p style={{ color: '#64748b', marginTop: '8px' }}>Vibrant AI Code Optimization Engine</p>
        </header>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
          
          {/* --- 1. INPUT BLOCK (Blue Theme) --- */}
          <section style={getGlassyBorderStyle('#3b82f6, #06b6d4', 'rgba(59, 130, 246, 0.2)')}>
            <div style={getGlassyContentStyle()}>
              <h2 style={{ fontSize: '1.2rem', color: '#38bdf8', marginBottom: '15px', fontWeight: '700' }}>Input Logic</h2>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="// Paste code logic here (Up to 400+ lines)..."
                disabled={loading}
                style={{ 
                  width: '100%', minHeight: '400px', padding: '20px', 
                  fontFamily: '"Fira Code", monospace', fontSize: '14px', borderRadius: '10px', 
                  border: '1px solid rgba(255, 255, 255, 0.05)', backgroundColor: 'rgba(0, 0, 0, 0.3)',
                  color: '#f8fafc', resize: 'vertical', outline: 'none', lineHeight: '1.6'
                }}
              />
              <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <input type="file" onChange={handleFile} style={{ fontSize: '0.85rem', color: '#64748b' }} />
                <button 
                  onClick={handleRefine}
                  disabled={loading || readingFile}
                  style={{ 
                    padding: '12px 36px', borderRadius: '8px', background: '#fff', color: '#000', 
                    fontWeight: '700', border: 'none', cursor: 'pointer', transition: '0.2s'
                  }}
                >
                  {loading ? 'Analyzing...' : 'Refine Code'}
                </button>
              </div>
              {error && <div style={{ marginTop: '15px', color: '#f87171', fontSize: '0.9rem' }}>{error}</div>}
            </div>
          </section>

          {/* --- RESULTS (Vertical Stack) --- */}
          {isActive && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '32px', animation: 'fadeIn 0.7s ease forwards' }}>
              
              {/* 2. SCORE BLOCK (Purple Theme) */}
              <div style={getGlassyBorderStyle('#a855f7, #ec4899', 'rgba(168, 85, 247, 0.15)')}>
                <div style={{ ...getGlassyContentStyle(), flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center', padding: '40px' }}>
                   <div style={{ textAlign: 'center' }}>
                      <p style={{ color: '#94a3b8', textTransform: 'uppercase', fontSize: '0.7rem', marginBottom: '8px' }}>Original Score</p>
                      <div style={{ fontSize: '3rem', fontWeight: '800', color: '#f87171' }}>
                        <AnimatedNumber target={response?.original_score} />
                      </div>
                   </div>
                   <div style={{ fontSize: '2rem', color: '#1e293b' }}>➔</div>
                   <div style={{ textAlign: 'center' }}>
                      <p style={{ color: '#94a3b8', textTransform: 'uppercase', fontSize: '0.7rem', marginBottom: '8px' }}>Refined Score</p>
                      <div style={{ fontSize: '3rem', fontWeight: '800', color: '#10b981' }}>
                        <AnimatedNumber target={response?.refined_score} />
                      </div>
                   </div>
                </div>
              </div>

              {/* 3. METRICS BLOCK (Teal Theme) */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                  <div style={getGlassyBorderStyle('#14b8a6, #10b981', 'rgba(20, 184, 166, 0.1)')}>
                    <div style={getGlassyContentStyle()}>
                      <h3 style={{ fontSize: '0.85rem', color: '#2dd4bf', fontWeight: '700', marginBottom: '10px' }}>TIME COMPLEXITY</h3>
                      <p style={{ fontSize: '1.2rem', margin: 0 }}>{response?.time_complexity || 'N/A'}</p>
                    </div>
                  </div>
                  <div style={getGlassyBorderStyle('#14b8a6, #10b981', 'rgba(20, 184, 166, 0.1)')}>
                    <div style={getGlassyContentStyle()}>
                      <h3 style={{ fontSize: '0.85rem', color: '#10b981', fontWeight: '700', marginBottom: '10px' }}>SPACE COMPLEXITY</h3>
                      <p style={{ fontSize: '1.2rem', margin: 0 }}>{response?.space_complexity || 'N/A'}</p>
                    </div>
                  </div>
              </div>

              {/* 4. BUG BLOCK (Red/Orange Theme) */}
              {response?.bugs && (
                <div style={getGlassyBorderStyle('#ef4444, #fb923c', 'rgba(239, 68, 68, 0.15)')}>
                  <div style={getGlassyContentStyle()}>
                    <h3 style={{ fontSize: '0.85rem', color: '#f87171', fontWeight: '700', marginBottom: '8px' }}>DIAGNOSTICS</h3>
                    <p style={{ margin: 0, fontSize: '0.95rem', color: '#e2e8f0', lineHeight: '1.6' }}>{response?.bugs}</p>
                  </div>
                </div>
              )}

              {/* 5. OPTIMIZED BLOCK (GOLD SHINING - Side by Side Comparison) */}
              <div style={getGlassyBorderStyle('#f59e0b, #fbbf24', 'rgba(245, 158, 11, 0.25)')}>
                <div style={getGlassyContentStyle()}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <h3 style={{ fontSize: '1.2rem', color: '#fbbf24', fontWeight: '800' }}>✨ Optimized Architecture</h3>
                    <select 
                      value={targetLanguage} 
                      onChange={handleTranslate}
                      style={{ 
                        background: '#1c1917', color: '#fbbf24', border: '1px solid #78350f', 
                        borderRadius: '8px', padding: '8px 16px', fontSize: '0.9rem', fontWeight: '700', cursor: 'pointer'
                      }}
                    >
                      <option value="">Translate to Language...</option>
                      <option value="Java">Java</option>
                      <option value="Go">Go</option>
                      <option value="Python">Python</option>
                      <option value="Rust">Rust</option>
                      <option value="C++">C++</option>
                    </select>
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                    <div style={{ border: '1px solid rgba(255,255,255,0.05)', borderRadius: '10px', overflow: 'hidden', background: '#000' }}>
                      <div style={{ padding: '10px 15px', background: 'rgba(255,255,255,0.02)', color: '#64748b', fontSize: '0.75rem', fontWeight: '700', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>REFINED CODE</div>
                      <div className="code-block-container" style={{ maxHeight: '600px', overflowY: 'auto' }}>
                         <CodeBlock code={response?.optimized_code} language="javascript" />
                      </div>
                    </div>
                    <div style={{ border: '1px solid rgba(255,255,255,0.05)', borderRadius: '10px', overflow: 'hidden', background: '#000' }}>
                      <div style={{ padding: '10px 15px', background: 'rgba(255,255,255,0.02)', color: '#fbbf24', fontSize: '0.75rem', fontWeight: '700', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        {isTranslating ? 'SYNCING...' : targetLanguage ? targetLanguage.toUpperCase() : 'TRANSLATED CODE'}
                      </div>
                      <div className="code-block-container" style={{ maxHeight: '600px', overflowY: 'auto' }}>
                         <CodeBlock code={translatedCode || '// Select a target language above'} language={targetLanguage?.toLowerCase() || 'text'} />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 6. STRATEGY BLOCK (Sky Blue Theme) */}
              <div style={getGlassyBorderStyle('#0ea5e9, #a78bfa', 'rgba(14, 165, 233, 0.1)')}>
                <div style={getGlassyContentStyle()}>
                  <h3 style={{ fontSize: '0.9rem', color: '#7dd3fc', fontWeight: '700', marginBottom: '10px' }}>LOGIC EXPLANATION</h3>
                  <p style={{ margin: 0, fontSize: '1rem', lineHeight: '1.7', color: '#cbd5e1' }}>{response?.explanation}</p>
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}