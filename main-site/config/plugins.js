import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import gitWarningPlugin from '../src/remark/git-warning';

module.exports = [
  require.resolve("docusaurus-plugin-image-zoom"),
  [
    '@docusaurus/plugin-content-docs',
    {
      id: 'pluginForGit',
      path: 'docs/git',
      routeBasePath: 'git',
      sidebarPath: require.resolve('../config/sidebars-git.js'),
      remarkPlugins: [remarkMath, gitWarningPlugin],
      rehypePlugins: [[rehypeKatex, { strict: false }]],
      showLastUpdateAuthor: true,
      showLastUpdateTime: true,
      breadcrumbs: false,
      editUrl: "https://github.com/ZhenShuo2021/docs/edit/main",
    },
  ],
  [
    '@docusaurus/plugin-content-docs',
    {
      id: 'pluginForPython',
      path: 'docs/python',
      routeBasePath: 'python',
      sidebarPath: require.resolve('../config/sidebars-python.js'),
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: false }]],
      showLastUpdateAuthor: true,
      showLastUpdateTime: true,
      breadcrumbs: false,
      editUrl: "https://github.com/ZhenShuo2021/docs/edit/main",
    },
  ],
  [
    '@docusaurus/plugin-content-docs',
    {
      id: 'pluginForMemo',
      path: 'docs/memo',
      routeBasePath: 'memo',
      sidebarPath: require.resolve('../config/sidebars-memo.js'),
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: false }]],
      showLastUpdateAuthor: true,
      showLastUpdateTime: true,
      breadcrumbs: false,
      editUrl: "https://github.com/ZhenShuo2021/docs/edit/main",
    },
  ],
  [
    '@docusaurus/plugin-content-docs',
    {
      id: 'pluginForLinuxCommand',
      path: 'docs/linux-command',
      routeBasePath: 'linux-command',
      sidebarPath: require.resolve('../config/sidebars-linux-command.js'),
      remarkPlugins: [remarkMath],
      rehypePlugins: [[rehypeKatex, { strict: false }]],
      showLastUpdateAuthor: true,
      showLastUpdateTime: true,
      breadcrumbs: false,
      editUrl: "https://github.com/ZhenShuo2021/docs/edit/main",
    },
  ],
  [
    '@docusaurus/plugin-content-blog',
    {
      id: 'default',
      path: 'blog',
      routeBasePath: 'blog',
      blogSidebarCount: 0,
      blogSidebarTitle: "最新文章",
      blogDescription: "部落格，分享我對各種技術議題的觀點與開發實作紀錄",
      postsPerPage: 10,
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
      showLastUpdateTime: false,
    },
  ],
  [
    '@docusaurus/plugin-google-gtag',
    {
      trackingID: 'G-QB2VKFSQ0J',
      anonymizeIP: true,
    },
  ],
];
