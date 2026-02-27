import React from 'react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <>
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

          .saas-container {
            font-family: 'Poppins', sans-serif;
            background-color: #0b0f19; /* Deep premium navy */
            color: #ffffff;
            min-height: 100vh;
            overflow: hidden;
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
          }

          /* Ambient Glowing Background Backgrounds */
          .glow-blob-1 {
            position: absolute;
            top: -10%; left: -10%;
            width: 50vw; height: 50vw;
            background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, rgba(0,0,0,0) 70%);
            z-index: 0;
            pointer-events: none;
          }
          .glow-blob-2 {
            position: absolute;
            bottom: -10%; right: -10%;
            width: 50vw; height: 50vw;
            background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, rgba(0,0,0,0) 70%);
            z-index: 0;
            pointer-events: none;
          }

          /* Typography */
          .hero-title {
            font-size: clamp(3rem, 6vw, 5rem);
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1.5rem;
            z-index: 1;
            position: relative;
          }
          
          .gradient-text {
            background: linear-gradient(135deg, #60a5fa 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }

          .hero-subtitle {
            font-size: 1.25rem;
            color: #94a3b8;
            max-width: 600px;
            margin: 0 auto 3rem auto;
            line-height: 1.6;
            font-weight: 300;
            z-index: 1;
            position: relative;
          }

          /* Interactive Premium Button */
          .cta-button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            padding: 1rem 2.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
            z-index: 1;
            position: relative;
            border: 1px solid rgba(255,255,255,0.1);
          }
          
          .cta-button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.6);
            color: white;
          }

          /* Glassmorphism Cards */
          .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            width: 100%;
            max-width: 1100px;
            padding: 0 2rem;
            z-index: 1;
            position: relative;
            margin-bottom: 5rem;
          }

          .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 2rem;
            text-align: left;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
          }

          .glass-card:hover {
            transform: translateY(-10px);
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
          }

          .card-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
          }

          .card-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            color: #f1f5f9;
          }

          .card-desc {
            color: #94a3b8;
            font-size: 0.95rem;
            line-height: 1.6;
          }
        `}
      </style>

      <div className="saas-container">
        {/* Ambient Background Globs */}
        <div className="glow-blob-1"></div>
        <div className="glow-blob-2"></div>

        {/* --- HERO SECTION --- */}
        <section style={{ textAlign: 'center', marginTop: '8rem', marginBottom: '5rem', padding: '0 2rem' }}>
          <h1 className="hero-title">
            CodeRefine <br />
            <span className="gradient-text">Zero Compromises.</span>
          </h1>
          <p className="hero-subtitle">
            The ultimate AI-powered workspace to review, optimize, score, and seamlessly translate your codebase in real-time.
          </p>
          <Link to="/editor" className="cta-button">
            Start Refining for Free
          </Link>
        </section>

        {/* --- GLASSMORPHISM FEATURES GRID --- */}
        <section className="features-grid">
          
          <div className="glass-card">
            <div className="card-icon">🎯</div>
            <h3 className="card-title">AI Code Scoring</h3>
            <p className="card-desc">
              Get objective "Before & After" metrics out of 100 to instantly visualize your code quality and logical improvements.
            </p>
          </div>

          <div className="glass-card">
            <div className="card-icon">🔄</div>
            <h3 className="card-title">Polyglot Translation</h3>
            <p className="card-desc">
              Instantly port your optimized algorithms between Python, C++, Java, JS, Rust, and Go with our side-by-side split screen.
            </p>
          </div>

          <div className="glass-card">
            <div className="card-icon">⚡</div>
            <h3 className="card-title">Big-O Analysis</h3>
            <p className="card-desc">
              Automatically calculate and compare Time and Space complexity so your applications scale flawlessly.
            </p>
          </div>

          <div className="glass-card">
            <div className="card-icon">🛡️</div>
            <h3 className="card-title">Deep Vulnerability Scans</h3>
            <p className="card-desc">
              Detect hidden memory leaks, logic errors, and security flaws before they ever make it into your production environment.
            </p>
          </div>

        </section>

        {/* --- MINIMAL FOOTER --- */}
        <footer style={{ marginTop: 'auto', padding: '2rem', borderTop: '1px solid rgba(255,255,255,0.05)', width: '100%', textAlign: 'center', zIndex: 1 }}>
          <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
            © {new Date().getFullYear()} CodeRefine. Designed for the builders of tomorrow.
          </p>
        </footer>
      </div>
    </>
  );
}