import { useEffect, useMemo, useRef, useState } from 'react';
import { media } from '../lib/assets';
import './StructureGallery.css';

export type Case = { id: string; segment: string; look: string; condition: string; origin: string; seen: boolean; tags: string[] };

const pretty = (s: string) => s.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, c => c.toUpperCase());
// readable names of the looks whose ids run words together
const LOOK: Record<string, string> = {"aaa": "AAA game", "alienhive": "Alien hive", "autumngold": "Autumn gold", "bathtubtoy": "Bath toy", "blackpearl": "Black Pearl", "cleanfuture": "Clean future", "comicink": "Comic ink", "comictoon": "Comic toon", "cyberninja": "Cyber ninja", "desertday": "Desert day", "duskruins": "Dusk ruins", "edoukiyoe": "Edo ukiyo-e", "eldenring": "Elden Ring", "ghostship": "Ghost ship", "glowforest": "Glowing forest", "goldenhour": "Golden hour", "hokusaiwoodblock": "Hokusai woodblock", "ironsuit": "Iron suit", "kpop": "K-pop", "livefantasy": "Live-action fantasy", "lupinanime": "Lupin anime", "madmax": "Mad Max", "mobydickoilpainting": "Moby-Dick oil painting", "modernla": "Modern LA", "moscowsnow": "Moscow snow", "neontokyo": "Neon Tokyo", "nightrain": "Night rain", "nightsnow": "Night snow", "nightstorm": "Night storm", "oilpainting": "Oil painting", "onepieceanime": "One Piece anime", "papercut": "Paper cut", "parisheist": "Paris heist", "pixelart": "Pixel art", "postapoc": "Post-apocalyptic", "rainnight": "Rainy night", "scifi": "Sci-fi", "shadowpuppet": "Shadow puppet", "skypirate": "Sky pirate", "snowdusk": "Snowy dusk", "spacemarine": "Space marine", "spaceopera": "Space opera", "stormlight": "Storm light", "techstage": "Tech stage", "tombraider": "Tomb Raider", "toonroyale": "Toon royale", "toonviking": "Toon viking", "tron": "Tron", "ukiyoe": "Ukiyo-e", "voxelcraft": "Voxel craft", "westerntown": "Western town"};
const lookLabel = (s: string) => LOOK[s] ?? pretty(s);
// the case shown first for its structure type
const FIRST = new Set(['mat_stagedance__techstage']);

/** One case: the render with the structure laid over it up to a draggable seam; both clips play in step. */
export function Wipe({ c, autoplay = false }: { c: Case; autoplay?: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const render = useRef<HTMLVideoElement>(null);
  const cond = useRef<HTMLVideoElement>(null);
  const [x, setX] = useState(0.5);
  const [live, setLive] = useState(autoplay);
  useEffect(() => {
    const el = box.current; if (!el) return;
    const io = new IntersectionObserver(([e]) => setLive(e.isIntersecting), { threshold: 0.25 });
    io.observe(el); return () => io.disconnect();
  }, []);
  useEffect(() => {
    const r = render.current, s = cond.current; if (!r || !s) return;
    if (live) { r.play().catch(() => {}); s.play().catch(() => {}); } else { r.pause(); s.pause(); }
    const sync = () => { if (Math.abs(s.currentTime - r.currentTime) > 0.08) s.currentTime = r.currentTime; };
    r.addEventListener('timeupdate', sync); return () => r.removeEventListener('timeupdate', sync);
  }, [live]);
  const move = (clientX: number) => {
    const b = box.current?.getBoundingClientRect(); if (!b) return;
    setX(Math.min(1, Math.max(0, (clientX - b.left) / b.width)));
  };
  const base = `media/gallery/${c.id}`;
  return <div ref={box} className="wipe" style={{ ['--x' as string]: `${x * 100}%` }}
    onPointerMove={e => { if (e.buttons || e.pointerType === 'mouse') move(e.clientX); }}
    onPointerDown={e => move(e.clientX)}>
    <video ref={render} src={live ? media(`${base}/render.mp4`) : undefined} poster={media(`${base}/render.webp`)} muted loop playsInline preload="none" />
    <video ref={cond} className="wipe-cond" src={live ? media(`${base}/cond.mp4`) : undefined} poster={media(`${base}/cond.webp`)} muted loop playsInline preload="none" />
    <i className="wipe-seam" aria-hidden="true" />
    <span className="wipe-tag wipe-tag-l" style={{ opacity: x < 0.2 ? 0 : 1 }}>Input</span><span className="wipe-tag wipe-tag-r" style={{ opacity: x > 0.75 ? 0 : 1 }}>SolarRenderer</span>
  </div>;
}

export default function StructureGallery({ cases }: { cases: Case[] }) {
  const types = useMemo(() => {
    const m = new Map<string, { seen: boolean; n: number }>();
    cases.forEach(c => { const t = m.get(c.condition); m.set(c.condition, { seen: (t?.seen ?? false) || c.seen, n: (t?.n ?? 0) + 1 }); });   // a type counts as seen if any of its cases is
    return [...m.entries()].sort((a, b) => Number(b[1].seen) - Number(a[1].seen) || b[1].n - a[1].n);
  }, [cases]);
  // "All" deals the cases out one structure type at a time, unseen types first, so the first rows show the range
  const dealt = useMemo(() => {
    const by = new Map<string, Case[]>();
    cases.forEach(c => by.set(c.condition, [...(by.get(c.condition) ?? []), c]));
    by.forEach(list => list.sort((a, b) => Number(FIRST.has(b.id)) - Number(FIRST.has(a.id))));
    const seenType = (t: string) => by.get(t)!.some(c => c.seen);
    const order = [...by.keys()].sort((a, b) => Number(seenType(a)) - Number(seenType(b)));
    const out: Case[] = [];
    for (let k = 0; out.length < cases.length; k++) order.forEach(t => { const c = by.get(t)![k]; if (c) out.push(c); });
    return out;
  }, [cases]);
  const [type, setType] = useState('all');
  const [count, setCount] = useState(12);
  const shown = dealt.filter(c => type === 'all' || (type === 'unseen' ? !c.seen : c.condition === type));
  return <div className="sg">
    <div className="sg-filters" role="tablist" aria-label="Structure type">
      <button role="tab" aria-selected={type === 'all'} onClick={() => { setType('all'); setCount(12); }}>All <b>{cases.length}</b></button>
      <button role="tab" aria-selected={type === 'unseen'} className="sg-unseen" onClick={() => { setType('unseen'); setCount(12); }}>Unseen in training</button>
      <span className="sg-sep" aria-hidden="true" />
      {types.map(([t, v]) => <button key={t} role="tab" aria-selected={type === t} className={v.seen ? '' : 'sg-new'}
        onClick={() => { setType(t); setCount(12); }}>{pretty(t)} <b>{v.n}</b></button>)}
    </div>
    <div className="sg-grid">
      {shown.slice(0, count).map(c => <figure key={c.id} className="sg-card">
        <Wipe c={c} />
        <figcaption><span>{pretty(c.condition)}{!c.seen && <em>unseen</em>}</span><small>{lookLabel(c.look)}</small></figcaption>
      </figure>)}
    </div>
    {count < shown.length && <button className="sg-more" onClick={() => setCount(n => n + 12)}>Show more <small>{shown.length - count} left</small></button>}
  </div>;
}
