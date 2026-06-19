// Post-build pass: rewrite every root-absolute URL in the built HTML to a
// page-relative one, so the site works when opened directly from disk (file://)
// with no server. Astro emits absolute paths (`/_astro/…`, `/js/…`) and our
// internal links are absolute (`/concepts/…`); file:// has no server root and no
// directory-index resolution, so both must become explicit relative paths to
// concrete files. Runs after `astro build` (see the `build` npm script).
//
// Handled: href/src attributes, srcset candidates, and `url(/…)` inside inlined
// <style>. Page routes (no file extension) map to their `…/index.html`; assets
// (with an extension) map straight through. External (`//`, `http`, `data:`,
// `mailto:`) and pure-hash URLs are left untouched.

import { readdir, readFile, writeFile } from "node:fs/promises";
import { join, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { posix } from "node:path";

const DIST = fileURLToPath(new URL("../dist/", import.meta.url));

async function* walkHtml(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) yield* walkHtml(full);
    else if (entry.name.endsWith(".html")) yield full;
  }
}

/** Map a root-absolute URL to a path relative to the current page's directory. */
function toRelative(absUrl, fromDir) {
  let hash = "";
  const hashAt = absUrl.indexOf("#");
  let pathPart = absUrl;
  if (hashAt !== -1) {
    hash = absUrl.slice(hashAt);
    pathPart = absUrl.slice(0, hashAt);
  }

  let target;
  if (pathPart === "/" || pathPart === "") {
    target = "index.html";
  } else {
    const stripped = pathPart.replace(/^\/+/, "").replace(/\/+$/, "");
    const last = stripped.split("/").pop() ?? "";
    // A filename with an extension is an asset; otherwise it's a page route.
    target = last.includes(".") ? stripped : `${stripped}/index.html`;
  }

  let rel = posix.relative(fromDir, target);
  if (rel === "") rel = target.split("/").pop() ?? target;
  if (!rel.startsWith(".")) rel = `./${rel}`;
  return rel + hash;
}

const isExternal = (u) =>
  u.startsWith("//") || /^[a-z][a-z0-9+.-]*:/i.test(u) || u.startsWith("#");

let files = 0;
let rewrites = 0;

for await (const file of walkHtml(DIST)) {
  const relFile = relative(DIST, file).split(/[\\/]/).join("/");
  const fromDir = posix.dirname(relFile) === "." ? "" : posix.dirname(relFile);
  let html = await readFile(file, "utf8");

  // href="/…" and src="/…"
  html = html.replace(/(\bhref|\bsrc)=("|')(\/[^"']*)\2/g, (m, attr, q, url) => {
    if (isExternal(url)) return m;
    rewrites++;
    return `${attr}=${q}${toRelative(url, fromDir)}${q}`;
  });

  // srcset="/a 1x, /b 2x"
  html = html.replace(/\bsrcset=("|')([^"']*)\1/g, (m, q, val) => {
    const out = val
      .split(",")
      .map((cand) => {
        const seg = cand.trim();
        if (!seg) return seg;
        const sp = seg.split(/\s+/);
        if (sp[0].startsWith("/") && !isExternal(sp[0])) {
          sp[0] = toRelative(sp[0], fromDir);
          rewrites++;
        }
        return sp.join(" ");
      })
      .filter(Boolean)
      .join(", ");
    return `srcset=${q}${out}${q}`;
  });

  // url(/…) inside inlined <style>
  html = html.replace(/url\((['"]?)(\/[^)'"]+)\1\)/g, (m, q, url) => {
    if (isExternal(url)) return m;
    rewrites++;
    return `url(${q}${toRelative(url, fromDir)}${q})`;
  });

  await writeFile(file, html);
  files++;
}

console.log(`[relativize] rewrote ${rewrites} absolute URL(s) across ${files} HTML file(s) for file:// use`);
