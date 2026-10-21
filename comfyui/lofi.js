// lofi.js - a three.js game played as a rough early build: loaded before the game's own
// script (web_game.py, lofi on), window.__lofi() then takes the detail out of everything it draws - each
// texture becomes one pixel of its average colour, surfaces flat-shaded and matte, no shadows, environment or fog - so
// the renderer has the most to add. The game itself is untouched: its logic, sizes and colours stay as they were.
(() => {
  // three.js announces every Scene and WebGLRenderer it makes to a devtools hook, if one exists: keep them
  const scenes = [], renderers = [];
  const hook = new EventTarget();
  hook.addEventListener('observe', e => {
    const o = e.detail;
    if (o && o.isScene) scenes.push(o);
    else if (o && o.isWebGLRenderer) {
      renderers.push(o);
      const render = o.render.bind(o);
      o.render = (scene, camera) => {  // whatever scene is drawn (passes, off-screen scenes too) gets swept
        if (window.__lofiOn && scene && scene.isScene && !scenes.includes(scene)) scenes.push(scene);
        return render(scene, camera);
      };
    }
  });
  window.__THREE_DEVTOOLS__ = hook;
  const MAPS = ['map', 'normalMap', 'bumpMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap', 'envMap',
                'lightMap', 'displacementMap', 'alphaMap', 'clearcoatMap', 'clearcoatNormalMap', 'sheenColorMap',
                'specularMap', 'specularIntensityMap', 'iridescenceMap', 'transmissionMap', 'thicknessMap'];
  const probe = document.createElement('canvas'); probe.width = probe.height = 1;
  const pctx = probe.getContext('2d', { willReadFrequently: true });
  function mean(tex) {  // a texture's average colour (sRGB 0..1), so an untextured surface keeps its hue
    try {
      const img = tex && tex.image;
      if (!img) return null;
      if (img.data && img.width) {
        const d = img.data, n = img.width * img.height, step = Math.max(1, Math.floor(n / 4096)), ch = d.length / n;
        let r = 0, g = 0, b = 0, k = 0;
        for (let i = 0; i < n; i += step) { r += d[i * ch]; g += d[i * ch + 1]; b += d[i * ch + 2]; k++; }
        const s = d instanceof Uint8Array || d instanceof Uint8ClampedArray ? 255 : 1;
        return [r / k / s, g / k / s, b / k / s];
      }
      pctx.clearRect(0, 0, 1, 1); pctx.drawImage(img, 0, 0, 1, 1);
      const p = pctx.getImageData(0, 0, 1, 1).data;
      return [p[0] / 255, p[1] / 255, p[2] / 255];
    } catch (err) { return null; }
  }
  // a texture is kept (games read its offset, repeat, ...) but its picture becomes one pixel of its own average
  // colour: the surface keeps its hue and loses every detail; normal maps flatten to their mean normal
  const blanked = new WeakSet();
  function blank(tex) {
    if (!tex || blanked.has(tex) || tex.isCubeTexture || tex.isRenderTargetTexture || tex.isVideoTexture) return false;
    const c = mean(tex);
    if (!c) return false;
    blanked.add(tex);
    const px = document.createElement('canvas'); px.width = px.height = 1;
    const g = px.getContext('2d');
    g.fillStyle = `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
    g.fillRect(0, 0, 1, 1);
    tex.image = px;
    if ('mipmaps' in tex) tex.mipmaps = [];
    tex.dispose();  // the card's copy was allocated at the old size (texStorage): drop it, it is made anew from 1x1
    tex.needsUpdate = true;
    return true;
  }
  // every sweep: a game may hand a material its textures later (made lazily, loaded async)
  function flatten(m) {
    if (!m || m.isShaderMaterial || m.isRawShaderMaterial || m.isPointsMaterial || m.isSpriteMaterial || m.isLineBasicMaterial) return;
    let changed = false;
    for (const k of MAPS) {
      const t = m[k];
      if (!t) continue;
      if (k === 'envMap' || t.isCubeTexture) { m[k] = null; changed = true; } else blank(t);
    }
    if ('flatShading' in m && !m.flatShading) { m.flatShading = true; changed = true; }
    if ('roughness' in m && m.roughness !== 1) { m.roughness = 1; changed = true; }
    if ('metalness' in m && m.metalness !== 0) { m.metalness = 0; changed = true; }
    for (const k of ['clearcoat', 'sheen', 'transmission', 'iridescence']) if (typeof m[k] === 'number' && m[k]) { m[k] = 0; changed = true; }
    if (changed) m.needsUpdate = true;
  }
  function sweep() {
    for (const r of renderers) r.shadowMap.enabled = false;
    for (const s of scenes) {
      s.environment = null; s.fog = null;
      s.traverse(o => {
        if (o.isMesh || o.isSkinnedMesh || o.isInstancedMesh) {
          o.castShadow = o.receiveShadow = false;
          (Array.isArray(o.material) ? o.material : [o.material]).forEach(flatten);
        }
      });
    }
  }
  window.__lofi = () => { window.__lofiOn = true; sweep(); setInterval(sweep, 300); return [scenes.length, renderers.length]; };
})();
