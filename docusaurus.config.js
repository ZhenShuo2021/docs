// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const katex = require("rehype-katex");
const math = require("remark-math");
const path = require("path");

const unified = require("unified");
const remarkParse = require("remark-parse");
const stringify = require("rehype-stringify");
const remark2rehype = require("remark-rehype");

// import { rehypeExtendedTable } from 'rehype-extended-table';

import { themes as prismThemes } from 'prism-react-renderer';

// require("dotenv").config();

function unwrapCategory(items) {
  const newItems = [];

  items.forEach((item) => {
    const isDoc = item.type === "doc";
    const isCategory = item.type === "category";
    const hasOnlyOneDocItem = isCategory && item.items.length === 1 && item.items[0].type === "doc";
    if (isDoc) {
      newItems.push(item);
      return;
    }

    if (hasOnlyOneDocItem) {
      newItems.push(item.items[0]);
      return;
    }

    item.items = unwrapCategory(item.items);
    newItems.push(item);
  });

  return newItems;
}

export default {
  markdown: {
    mdx1Compat: {
      comments: true,
      admonitions: true,
      headingIds: true,
    },
  },
};

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: "Docs",
  tagline: "個人用文檔庫",
  url: "https://docs.zsl0621.cc", //process.env.URL,
  baseUrl: "/", //process.env.BASE_URL,
  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",
  favicon: "img/favicon.ico",
  trailingSlash: false,
  organizationName: "ZhenShuo2021", // Usually your GitHub org/user name.
  projectName: "zsl0621@Docs", // Usually your repo name.
  i18n: {
    defaultLocale: 'zh-TW',
    locales: ['zh-TW'],
  },
  plugins: [
    require.resolve("docusaurus-plugin-image-zoom"),
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'memo',
        path: 'docs/memo',
        routeBasePath: 'memo',
        sidebarPath: require.resolve('./sidebars-memo.js'),
        remarkPlugins: [math],
        rehypePlugins: [
          // [rehypeExtendedTable, {}],
          [katex, {
            strict: false
          }]
        ],
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
        breadcrumbs: true,
      },
    ],
  ],
  markdown: {
    mermaid: true,
  },
  themes: ['docusaurus-theme-github-codeblock', '@docusaurus/theme-mermaid'],
  themeConfig: {
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 5,
    },
    metadata: [{
        name: 'robots',
        content: 'max-image-preview:large'
      },
      {
        name: 'og:type',
        content: 'article'
      // },
      // {
      //   name: 'fb:app_id',
      //   content: '173025689387886'
      }
    ],
    docs: {
      sidebar: {
        hideable: true,
      },
    },
    algolia: {
      appId: 'DSU91EEXY7',
      apiKey: 'd6be4daef0ab4a655727096c8a3a6000',
      indexName: 'zsl0621',
      contextualSearch: true,
    },
    zoom: {
      selector: '.markdown :not(em,a) > img',
      config: {
        background: {
          light: 'rgb(255, 255, 255)',
          dark: 'rgb(50, 50, 50)'
        }
      }
    },
    prism: {
      additionalLanguages: ["aspnet", "bash", "css", "csharp", "cshtml", "diff", "git", "java", "javascript", "json", "markup-templating", "powershell", "php", "python", "sql", "toml", "typescript"],
      theme: prismThemes.github,
      darkTheme: prismThemes.vsDark,
    },
    navbar: {
      title: "zsl0621@Docs",
      logo: {
        alt: "Site Logo",
        src: "img/favicon.svg",
      },
      items: [
        {
          type: "doc",
          docId: "intro/intro",  // 更新路徑
          position: "left",
          label: "文件庫📚",
        },
        {
          type: "doc",
          docId: "intro/memo",
          docsPluginId: "memo", // 指定使用 memo plugin
          position: "left",
          label: "備忘錄📝",
        },
        {
          href: 'https://blog.zsl0621.cc/',
          position: 'right',
          className: 'header-blog-link',
          'aria-label': 'Personal blog',
        },
        {
          href: 'https://github.com/ZhenShuo2021',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
        // {
        //   to: "/blog",
        //   label: "部落格",
        //   position: "left",
        // },
        // {
        //   to: "pathname:///slides",
        //   label: "投影片",
        //   position: "right",
        // }
      ],
    },
    footer: {
      style: "dark",
      copyright: `Copyright © ${new Date().getFullYear()}. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      "@docusaurus/preset-classic",
      {
        googleAnalytics: {
          trackingID: 'G-QB2VKFSQ0J',
          anonymizeIP: true,
        },
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl: "https://github.com/ZhenShuo2021/docs/edit/main",
          path: 'docs/docs',  // 更新文件庫路徑
          routeBasePath: 'docs',
          remarkPlugins: [math],
          rehypePlugins: [
            // [rehypeExtendedTable, {}],
            [katex, {
              strict: false
            }]
          ],
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          breadcrumbs: true,
          /*
          sidebarItemsGenerator: async function ({
            defaultSidebarItemsGenerator,
            ...args
          }) {
            const sidebarItems = await defaultSidebarItemsGenerator(args);
            return unwrapCategory(sidebarItems);
          },
          */
        },
        // blog: {
        //   blogSidebarCount: 0,
        //   showReadingTime: true,
        //   // Please change this to your repo.
        //   editUrl: "https://github.com/ouch1978/ouch1978.github.io/edit/main",
        //   remarkPlugins: [math],
        //   rehypePlugins: [
        //     [rehypeExtendedTable, {}],
        //     [katex, {
        //       strict: false
        //     }]
        //   ],
        // },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],
  stylesheets: [{
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css",
      type: "text/css",
      integrity: "sha384-Um5gpz1odJg5Z4HAmzPtgZKdTBHZdw8S29IecapCSB31ligYPhHQZMIlWLYQGVoc",
      crossorigin: "anonymous",
    },
    {
      rel: "preconnect",
      href: "https://fonts.googleapis.com",
    },
    {
      rel: "preconnect",
      href: "https://fonts.gstatic.com",
      crossorigin: "anonymous",
    },
    {
      rel: "stylesheet",
      href: "https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700;900&display=swap",
    },
  ],
  // scripts: [
  //   {
  //     src: "https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-8120664310721472",
  //     async: true,
  //     crossorigin: "anonymous",
  //   }
  // ],
};
