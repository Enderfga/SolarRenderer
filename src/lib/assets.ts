export function asset(path: string): string {
  if (/^(https?:|data:)/.test(path)) return path;
  return `${import.meta.env.BASE_URL.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
}
export function media(path: string): string {
  const origin = import.meta.env.PUBLIC_MEDIA_BASE;
  // PUBLIC_MEDIA_FLAT: every media file under one directory, its path folded into its name (preview hosting)
  if (import.meta.env.PUBLIC_MEDIA_FLAT && path.startsWith('media/')) return asset('srp-' + path.slice(6).replace(/\//g, '--'));
  if (origin && path.startsWith('media/')) return `${origin.replace(/\/$/, '')}/${path.slice(6)}`;
  return asset(path);
}
