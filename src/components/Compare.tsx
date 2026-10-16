import { useEffect, useRef, useState } from 'react';
import { media } from '../lib/assets';
import './Compare.css';

type Pair = { id: string; work: string; n: string; sec: number; structure: string };
type Props = { works: [string, string][]; pairs: Pair[] };

/** Their structure, their published output, and ours from the same structure: three clips in step. */
function Row({ p, name }: { p: Pair; name: string }) {
  const root = useRef<HTMLDivElement>(null);
  const vids = useRef<(HTMLVideoElement | null)[]>([]);
  const [live, setLive] = useState(false);
  useEffect(() => {
    const el = root.current; if (!el) return;
    const io = new IntersectionObserver(([e]) => setLive(e.isIntersecting), { threshold: 0.3 });
    io.observe(el); return () => io.disconnect();
  }, []);
  useEffect(() => {
    const [lead, ...rest] = vids.current.filter(Boolean) as HTMLVideoElement[];
    if (!lead) return;
    [lead, ...rest].forEach(v => { live ? v.play().catch(() => {}) : v.pause(); });
    const sync = () => rest.forEach(v => { if (Math.abs(v.currentTime - lead.currentTime) > 0.1) v.currentTime = lead.currentTime; });
    lead.addEventListener('timeupdate', sync); return () => lead.removeEventListener('timeupdate', sync);
  }, [live]);
  const col = (k: string, label: string, i: number, ours = false) => <figure className={ours ? 'cmp-ours' : ''}>
    <video ref={el => { vids.current[i] = el; }} src={live ? media(`media/cmp/cmp-${p.id.replace(/^cmp-/, '')}-${k}.mp4`) : undefined}
      poster={media(`media/cmp/cmp-${p.id.replace(/^cmp-/, '')}-${k}.jpg`)} muted loop playsInline preload="none" />
    <figcaption>{label}</figcaption>
  </figure>;
  return <div ref={root} className="cmp-row">
    {col('cond', `Input · ${p.structure}`, 0)}
    {col('theirs', name, 1)}
    {col('ours', 'SolarRenderer · 4 steps', 2, true)}
  </div>;
}

export default function Compare({ works, pairs }: Props) {
  const [work, setWork] = useState(works[0][0]);
  const names = Object.fromEntries(works);
  const shown = pairs.filter(p => p.work === work);
  return <div className="cmp">
    <div className="cmp-works" role="tablist" aria-label="Work">
      {works.map(([k, n]) => <button key={k} role="tab" aria-selected={k === work} onClick={() => setWork(k)}>
        {n} <b>{pairs.filter(p => p.work === k).length}</b></button>)}
    </div>
    <div className="cmp-list">{shown.map(p => <Row key={p.id} p={p} name={names[p.work]} />)}</div>
  </div>;
}
