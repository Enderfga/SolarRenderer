import { useEffect, useMemo, useRef, useState } from 'react';
import { media } from '../lib/assets';
import type { Case } from './StructureGallery';
import './Worlds.css';

const pretty = (s: string) => s.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, c => c.toUpperCase());

/** Pick a structure type, then one of its samples: the input clip and every style rendered from it, playing in step. */
export default function Worlds({ cases }: { cases: Case[] }) {
  // structure type -> its samples (one input clip each), richest sample first; seen types first, then unseen
  const types = useMemo(() => {
    const by = new Map<string, Map<string, Case[]>>();
    cases.forEach(c => {
      const t = by.get(c.condition) ?? new Map<string, Case[]>();
      t.set(c.segment, [...(t.get(c.segment) ?? []), c]);
      by.set(c.condition, t);
    });
    return [...by.entries()]
      .map(([condition, segs]) => ({ condition, seen: [...segs.values()][0][0].seen,
        samples: [...segs.values()].filter(s => s.length > 1).sort((a, b) => b.length - a.length) }))
      .filter(t => t.samples.length)
      .sort((a, b) => Number(b.condition === 'scene grey box') - Number(a.condition === 'scene grey box')
        || Number(b.seen) - Number(a.seen) || b.samples.length - a.samples.length);
  }, [cases]);
  const [ti, setTi] = useState(0);
  const [si, setSi] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const vids = useRef<(HTMLVideoElement | null)[]>([]);
  const [live, setLive] = useState(false);
  const type = types[ti];
  const looks = type.samples[Math.min(si, type.samples.length - 1)];
  useEffect(() => {
    const el = root.current; if (!el) return;
    const io = new IntersectionObserver(([e]) => setLive(e.isIntersecting), { threshold: 0.2 });
    io.observe(el); return () => io.disconnect();
  }, []);
  useEffect(() => {
    const [lead, ...rest] = vids.current.filter(Boolean) as HTMLVideoElement[];
    if (!lead) return;
    [lead, ...rest].forEach(v => { v.currentTime = 0; live ? v.play().catch(() => {}) : v.pause(); });
    const sync = () => rest.forEach(v => { if (Math.abs(v.currentTime - lead.currentTime) > 0.1) v.currentTime = lead.currentTime; });
    lead.addEventListener('timeupdate', sync); return () => lead.removeEventListener('timeupdate', sync);
  }, [ti, si, live]);
  vids.current = [];
  const key = looks[0].segment;
  const clip = (src: string, poster: string, label: string, k: number, input = false) =>
    <figure key={`${key}-${k}`} className={input ? 'w-input' : ''}>
      <video ref={el => { vids.current[k] = el; }} src={live ? media(src) : undefined} poster={media(poster)} muted loop playsInline preload="none" />
      <figcaption>{label}</figcaption>
    </figure>;
  return <div ref={root} className="worlds">
    <div className="w-tabs" role="tablist" aria-label="Structure type">
      {types.map((t, k) => <button key={t.condition} role="tab" aria-selected={k === ti} onClick={() => { setTi(k); setSi(0); }}>
        {pretty(t.condition)}{!t.seen && <em>unseen</em>}</button>)}
    </div>
    {type.samples.length > 1 && <div className="w-samples" role="tablist" aria-label="Sample">
      {type.samples.map((s, k) => <button key={s[0].segment} role="tab" aria-selected={k === si} onClick={() => setSi(k)}>Sample {k + 1}</button>)}
    </div>}
    <div className={`w-grid w-n${looks.length + 1}`}>
      {clip(`media/gallery/${looks[0].id}/cond.mp4`, `media/gallery/${looks[0].id}/cond.webp`, 'Input', 0, true)}
      {looks.map((c, k) => clip(`media/gallery/${c.id}/render.mp4`, `media/gallery/${c.id}/render.webp`, pretty(c.look), k + 1))}
    </div>
  </div>;
}
