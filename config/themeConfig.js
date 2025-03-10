const path = require('path');
import { themes as prismThemes } from 'prism-react-renderer';

module.exports = {
  docs: {
    sidebar: {
      hideable: true,
    },
  },
  tableOfContents: {
    minHeadingLevel: 2,
    maxHeadingLevel: 5,
  },
  prism: {
    additionalLanguages: ["aspnet", "bash", "css", "csharp", "cshtml", "diff", "git", "java", "javascript", "json", "markup-templating", "powershell", "php", "python", "sql", "toml", "typescript", "lua"],
    theme: prismThemes.github,
    darkTheme: prismThemes.vsDark,
  },
  navbar: {
    // title: "zsl0621@Docs",
    logo: {
      alt: "zsl0621@Docs",
      src: "img/logo_circle.webp",
    },
    items: [
      {
        type: 'doc',
        docsPluginId: 'pluginForGit',
        docId: 'git-hello-page',
        sidebarId: 'gitSidebar',
        label: "Git",
        position: 'right',
      },
      {
        type: 'doc',
        docsPluginId: 'pluginForPython',
        docId: 'python-hello-page',
        sidebarId: 'pythonSidebar',
        label: "Python",
        position: 'right',
      },
      // {
      //   type: 'docSidebar',
      //   position: 'left',
      //   sidebarId: 'docsSidebar',
      //   label: "文檔庫",
      // },
      {
        type: 'doc',
        docsPluginId: 'pluginForMemo',
        docId: 'about',
        sidebarId: 'memoSidebar',
        label: "筆記",
        position: 'right',
      },
      // {
      //   type: 'dropdown',
      //   label: '更多',
      //   position: 'right',
      //   items: [
      //     {
      //       label: '筆記',
      //       to: '/memo/about',
      //       // href: 'https://www.zsl0621.cc/',
      //     },
      //     {
      //       label: 'Github',
      //       href: 'https://github.com/ZhenShuo2021/',
      //     },
      //   ],
      // },


      {
        label: '導航',
        to: '/portfolio',
        position: 'right',
      },
      {
        href: 'https://github.com/ZhenShuo2021',
        position: 'right',
        className: 'header-github-link',
        'aria-label': 'GitHub repository',
      },
    ],
  },
  footer: {
    copyright: `
    © ${new Date().getFullYear()} ZhenShuo2021 (zsl0621.cc). Built with Docusaurus.<br>
    All rights reserved. 轉載或引用請註明來源。
  `
    // Or a simple CC
    // <a href="https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="noopener noreferrer">
    // CC BY-NC 4.0</a>  授權條款<br>
    ,
  },
  docs: {
    sidebar: {
      hideable: true,
      autoCollapseCategories: true,
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
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.resolve.alias = {
        ...webpackConfig.resolve.alias,
        '@docs': path.resolve(__dirname, 'docs'),
      };
      return webpackConfig;
    },
  },
};
