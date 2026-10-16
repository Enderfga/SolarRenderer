import { useEffect, useRef, useState } from 'react';
import { asset, media } from '../lib/assets';
import './Hero.css';

type HeroItem = { video: string; poster: string };
type Props = { tiles: HeroItem[]; title: string; paperUrl?: string; datasetUrl: string; codeUrl?: string | null; weightsUrl: string };

function Icon({ kind }: { kind: 'paper' | 'dataset' | 'code' | 'board' | 'weights' }) {
  if (kind === 'weights') return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" /><path d="M4 7.5 12 12l8-4.5M12 12v9" /></svg>;
  if (kind === 'paper') return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M9 13h6M9 17h6" /></svg>;
  if (kind === 'dataset') return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></svg>;
  if (kind === 'board') return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>;
  return <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .7a11.5 11.5 0 0 0-3.64 22.4c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.37-3.9-1.37-.52-1.34-1.28-1.7-1.28-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.57-.3-5.27-1.29-5.27-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.16 1.18a10.93 10.93 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.71 5.38-5.29 5.67.42.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z" /></svg>;
}

export default function Hero({ tiles, title, paperUrl, datasetUrl, codeUrl, weightsUrl }: Props) {
  const stage = useRef<HTMLElement>(null);
  const players = useRef<(HTMLVideoElement | null)[]>([]);
  const [paused, setPaused] = useState(false);
  const [inView, setInView] = useState(true);
  const [activated, setActivated] = useState(false);

  useEffect(() => {
    const pref = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => { if (pref.matches) setPaused(true); };
    update(); pref.addEventListener('change', update);
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { threshold: 0.05 });
    if (stage.current) io.observe(stage.current);
    return () => { io.disconnect(); pref.removeEventListener('change', update); };
  }, []);

  useEffect(() => {
    if (inView && !paused) setActivated(true);
    players.current.forEach(p => { if (!p || !p.getAttribute('src')) return; if (!paused && inView) p.play().catch(() => {}); else p.pause(); });
  }, [paused, inView, activated]);

  // three rows of nine; every tile is a render, looping, each row drawn twice for the endless slide
  const rows = [0, 1, 2].map(r => tiles.filter((_, i) => i % 3 === r).slice(0, 9));
  let n = 0;
  const tile = (t: HeroItem, key: string) => {
    const k = n++;
    return <span className="hm-tile" key={key}>
      <video ref={el => { players.current[k] = el; }} src={activated ? media(t.video) : undefined} poster={media(t.poster)} preload="none" muted loop playsInline
        onCanPlay={e => { if (!paused && inView) e.currentTarget.play().catch(() => {}); }} />
    </span>;
  };

  return <section ref={stage} className={`hero${paused ? ' hero-paused' : ''}`} aria-labelledby="hero-title">
    <div className="hm" aria-hidden="true">
      {rows.map((row, r) => <div className="hm-row" key={r}>{row.map((t, i) => tile(t, `a${i}`))}{row.map((t, i) => tile(t, `b${i}`))}</div>)}
    </div>
    <div className="hero-shade" aria-hidden="true" />
    <div className="hero-copy">
      <span className="hero-eyebrow"><i />Any structure in. Any world out.</span>
      <h1 id="hero-title" className="hero-logo"><img src={asset('brand/solar-ring.png')} alt="SolarRenderer" /></h1>
      <p className="hero-sub">{title.replace(/^SolarRenderer\s*[:：—-]\s*/i, '')}</p>
      <p className="hero-team">The agent builds it. We render it.</p>
      <div className="hero-btns">
        {paperUrl ? <a className="hbtn hbtn-glass" href={paperUrl} target="_blank" rel="noreferrer"><Icon kind="paper" />Paper</a>
          : <span className="hbtn hbtn-glass hbtn-off"><Icon kind="paper" />Paper <small>soon</small></span>}
        {codeUrl ? <a className="hbtn hbtn-glass" href={codeUrl} target="_blank" rel="noreferrer"><Icon kind="code" />Code</a>
          : <span className="hbtn hbtn-glass hbtn-off"><Icon kind="code" />Code <small>soon</small></span>}
        <a className="hbtn hbtn-glass" href={weightsUrl} target="_blank" rel="noreferrer"><Icon kind="weights" />Weights</a>
        <a className="hbtn hbtn-glass" href={datasetUrl} target="_blank" rel="noreferrer"><Icon kind="dataset" />Data</a>
      </div>
    </div>
    <div className="hero-foot">
      <button type="button" className="hero-pause" aria-label={paused ? 'Play background motion' : 'Pause background motion'} aria-pressed={paused} onClick={() => setPaused(v => !v)}>
        {paused ? <svg viewBox="0 0 20 20" aria-hidden="true"><path d="m7 4 9 6-9 6z" /></svg> : <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 4h3v12H6zm5 0h3v12h-3z" /></svg>}
      </button>
    </div>
  </section>;
}
