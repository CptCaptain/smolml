// @ts-check
import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import mdx from "@astrojs/mdx";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// The smolml learning compendium.
//
// HARD CONSTRAINT (user requirement): the built site must be fully usable by
// opening the HTML files DIRECTLY from disk (file://), with no server. That
// rules out ES-module island hydration (browsers block `type="module"` from a
// file:// origin), so there are no framework islands here — every interactive
// viz is a plain classic script (`public/js/compendium.js`) that mounts vanilla
// widgets by `data-widget` marker. Stylesheets are inlined (no external CSS
// 404s), and a post-build pass (`scripts/relativize.mjs`, chained in the
// `build` npm script) rewrites every absolute asset/link URL to a relative one
// so file:// resolves them. KaTeX math is wired via the markdown processor.
export default defineConfig({
  site: "https://smolml.local",
  build: {
    // Inline all CSS so there are zero external-stylesheet requests on file://.
    inlineStylesheets: "always",
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  integrations: [mdx()],
});
