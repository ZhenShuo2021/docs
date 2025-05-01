const presets = require("./config/presets");
const plugins = require("./config/plugins");
const themeConfig = require("./config/themeConfig");

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: "ZSL 的文檔庫",
  // tagline: "Reading a lot, writing what I read.",
  url: "https://zsl0621.cc",
  baseUrl: "/",
  organizationName: "ZhenShuo2021", // Usually your GitHub org/user name.
  projectName: "docs", // Usually your repo name.
  themes: ["docusaurus-theme-github-codeblock", "@docusaurus/theme-mermaid"],
  markdown: { mermaid: true },
  i18n: { defaultLocale: "zh-TW", locales: ["zh-TW"] },

  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css",
      type: "text/css",
      integrity: "sha384-5TcZemv2l/9On385z///+d7MSYlvIEw9FuZTIdZ14vJLqWphw7e7ZPuOiCHJcFCP",
      crossorigin: "anonymous",
      media: "print",
      onload: "this.media='all'"
    },
  ],

  // https://docusaurus.io/blog/releases/3.6#docusaurus-faster
  future: {
    experimental_faster: {
      swcJsLoader: true,
      swcJsMinimizer: true,
      // swcHtmlMinimizer incurs error pairing for html tags while building, especially for titles <h>
      // swcHtmlMinimizer: true,
      lightningCssMinimizer: true,
      rspackBundler: true,
      mdxCrossCompilerCache: true,
    },
  },

  // Basically, you don't need to change anything here
  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",
  favicon: "img/favicon.ico",
  trailingSlash: false,
  presets,
  plugins,
  themeConfig,
};
