import { useEffect, useRef, useState } from 'react';
import './StepFigure.css';

type Step = { title: string; text: string };
type Props = { src: string; width: number; height: number; steps: Step[]; label: string; dwell?: number };

/** A paper figure built up step by step: the figure (same drawing as the paper) on the left, the steps on the right.
 *  Plays while on screen, one step every `dwell` ms, then holds the full figure and starts again. */
export default function StepFigure({ src, width, height, steps, label, dwell = 2800 }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const frame = useRef<HTMLIFrameElement>(null);
  const [scale, setScale] = useState(0);
  const [step, setStep] = useState(1);          // 1..n; n + 1 = the whole figure, held before the loop restarts
  const [playing, setPlaying] = useState(true);
  const [full, setFull] = useState(false);
  const [visible, setVisible] = useState(false);
  const [ready, setReady] = useState(false);
  const n = steps.length;

  useEffect(() => {
    const el = box.current; if (!el) return;
    const ro = new ResizeObserver(([e]) => setScale(e.contentRect.width / width));
    ro.observe(el);
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.35 });
    io.observe(el);
    return () => { ro.disconnect(); io.disconnect(); };
  }, [width]);

  // the frame may finish loading before this component hydrates, and then its load event is gone: look for it
  useEffect(() => {
    let tries = 0;
    const id = setInterval(() => {
      const w = frame.current?.contentWindow as (Window & { show?: (k: number) => void }) | null;
      if (w?.show && w.document.readyState === 'complete') { setReady(true); clearInterval(id); }
      else if (++tries > 100) clearInterval(id);
    }, 100);
    return () => clearInterval(id);
  }, []);

  // drive the figure inside the frame
  useEffect(() => {
    const w = frame.current?.contentWindow as (Window & { show?: (k: number) => void }) | null;
    if (ready && w?.show) w.show(full || step > n ? 0 : step);
  }, [ready, step, full, n]);

  useEffect(() => {
    if (!playing || full || !visible) return;
    const t = setTimeout(() => setStep(s => (s > n ? 1 : s + 1)), step > n ? dwell * 1.4 : dwell);
    return () => clearTimeout(t);
  }, [playing, full, visible, step, n, dwell]);

  const go = (k: number) => { setFull(false); setStep(k); setPlaying(false); };
  const shown = full ? n + 1 : step;
  const running = playing && visible && step <= n;
  return <div className="sf">
    <div className="sf-tools">
      <button onClick={() => setFull(f => !f)} aria-pressed={full}>{full ? 'Step by step' : 'Full figure'}</button>
      {!full && <button onClick={() => { if (step > n) setStep(1); setPlaying(p => !p); }}>{playing ? 'Pause' : 'Play'}</button>}
    </div>
    <div className="sf-body">
      <div className="sf-stage">
        <div className="sf-fig" ref={box} style={{ height: scale ? height * scale : undefined, aspectRatio: scale ? undefined : `${width} / ${height}` }}>
          <iframe ref={frame} src={src} title={label} width={width} height={height} loading="lazy" scrolling="no" tabIndex={-1}
            onLoad={() => setReady(true)} style={{ transform: `scale(${scale})` }} />
        </div>
        {!full && <div className="sf-progress"><i key={`${step}-${running}`} className={running ? 'run' : ''}
          style={{ width: `${((running ? step - 1 : Math.min(step, n)) / n) * 100}%`, ['--to' as string]: `${(step / n) * 100}%`,
            ['--dur' as string]: `${dwell}ms` }} /></div>}
      </div>
      <ol className="sf-steps" style={{ ['--n' as string]: n }}>
        {steps.map((s, i) => <li key={s.title}>
          <button className={shown === i + 1 ? 'cur' : shown > i + 1 ? 'done' : ''} onClick={() => go(i + 1)}>
            <span className="sf-n">{i + 1}</span>
            <span><b>{s.title}</b><small>{s.text}</small></span>
          </button>
        </li>)}
      </ol>
    </div>
  </div>;
}
