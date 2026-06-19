// @ts-check
import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import mdx from "@astrojs/mdx";
import preact from "@astrojs/preact";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// The smolml learning compendium. Static build; `npm run build` is the PR gate.
// The Node/Astro toolchain is confined to docs/learning/ (ADR 0005).
//
// KaTeX math is wired through the Astro 6 `markdown.processor` API; MDX inherits
// the same processor, so equations render identically in .md and .mdx.
export default defineConfig({
  site: "https://smolml.local",
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  integrations: [preact(), mdx()],
});
