import type { DefaultTheme } from 'vitepress'
const currentYear = new Date().getFullYear()

const themeConfig: DefaultTheme.Config = {
  // logo: "/img/logo_circle.png",
  siteTitle: 'Git 零到一百',
  sidebar: {
    '/basic/': { base: '/basic/', items: sidebarBasic() },
    '/core/': { base: '/core/', items: sidebarCore() },
    '/pro/': { base: '/pro/', items: sidebarPro() },
    '/help/': { base: '/help/', items: sidebarHelp() },
  },
  editLink: {
    pattern: 'https://github.com/ZhenShuo2021/docs/edit/main/ripgit/docs/:path',
    text: '編輯此頁'
  },
  footer: {
    message: '本站內容採用 <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.zh-Hant" target="_blank" rel="noopener noreferrer">CC BY-NC 4.0</a> 授權，歡迎非商業轉載並註明出處。',
    copyright: `© 2024-${currentYear} ZhenShuo2021 (<a href="https://zsl0621.cc/" target="_blank" rel="noopener noreferrer">zsl0621.cc</a>)`
  },
  socialLinks: getSocialLinks(),
  lastUpdated: {
    text: '最後更新',
    // https://econ-sense.com/engineer/blog-2.html
    formatOptions: {
      forceLocale: true,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    },
  },
  nav: nav(),
  docFooter: { prev: '上一頁', next: '下一頁' },
  darkModeSwitchLabel: '深色模式',
  sidebarMenuLabel: '選單',
  returnToTopLabel: '回到頂端',
  outline: { label: '文章目錄', level: [2, 4] },
  externalLinkIcon: true,
}

function sidebarBasic(): DefaultTheme.SidebarItem[] {
  return [
    { text: '前言', link: 'intro' },
    {
      text: '序章',
      items: [
        { text: '安裝與設定', link: 'installation' },
        { text: '基礎知識', link: 'basic-knowledge' },
        { text: 'Vim 文字編輯器', link: 'vim' },
        { text: '關鍵字、符號和基本組成', link: 'keyword' },
        { text: '讀懂文檔', link: 'read-git-docs' },
      ],
    },
    {
      text: '新手上路',
      items: [
        { text: '一分鐘入門', link: 'one-minute' },
        { text: '基礎操作', link: 'step-above-basic' },
        { text: '分支操作', link: 'branch' },
        { text: '慣例式提交', link: 'conventional-commit' },
        { text: '打標籤', link: 'tags' },
      ],
    },
  ]
}

function sidebarCore(): DefaultTheme.SidebarItem[] {
  return [
    { text: '段落說明', link: 'intro' },
    {
      text: '修改提交歷史',
      items: [
        { text: '各種情境', link: 'edit-commits' },
        { text: '變基 Rebase 合併分支', link: 'rebase' },
        { text: '互動式變基', link: 'interactive-rebase' },
        { text: '引入提交 Cherry Pick', link: 'cherry-pick' },
      ],
    },
    {
      text: '遠端操作',
      items: [
        { text: '遠端儲存庫設定', link: 'remote-setup' },
        { text: '🔥 觀念和錯誤排除', link: 'remote-concept' },
        { text: '🔥 團隊協作最佳實踐', link: 'collaboration-best-practice' },
      ],
    },
    {
      text: '客製化 Git',
      items: [
        { text: '行前準備', link: 'git-bash-setup-in-windows' },
        { text: '開始設定', link: 'advanced-settings-and-aliases' },
      ],
    },
  ]
}

function sidebarPro(): DefaultTheme.SidebarItem[] {
  return [
    { text: '段落說明', link: 'intro' },
    {
      text: '高級技巧',
      // collapsed: false,
      items: [
        { text: 'Worktree 多工處理', link: 'git-worktree' },
        { text: 'Rebase Onto 詳解', link: 'rebase-onto' },
        { text: 'Sparse Checkout 加速克隆', link: 'reduce-size-with-sparse-checkout' },
        { text: 'Force if Includes 強制推送', link: 'force-if-includes' },
        { text: '多儲存庫管理', link: 'submodule-and-subtree' },
        { text: 'Git Bisect 找出錯誤提交', link: 'git-bisect' },
      ],
    },
    {
      text: 'Github 技巧',
      // collapsed: false,
      items: [
        { text: '搜尋技巧', link: 'github-search' },
        { text: 'README 嵌入影片', link: 'github-readme-video' },
        { text: 'Github Actions', link: 'github-actions' },
        { text: '本地執行 Github Action', link: 'run-github-actions-locally' },
        { text: '免費架設網站', link: 'github-pages' },
        { text: 'Git LFS 減少儲存庫容量', link: 'reduce-size-using-git-lfs' },
      ],
    },
  ]
}

function sidebarHelp(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '疑難排解',
      // collapsed: false,
      items: [
        { text: '正確的移除敏感訊息', link: 'removing-sensitive-data' },
        { text: '日常問題 - 本地', link: 'daily-issues-local' },
        { text: '日常問題 - 遠端', link: 'daily-issues-remote' },
        { text: 'GPG 無法簽名的錯誤', link: 'gpg-failed-to-sign-the-data' },
        { text: '很少用到的知識', link: 'uncommon-knowledge' },
        { text: '關於修改的動詞', link: 'verb-cheatsheet' },
      ],
    },
    {
      text: '批量修改',
      // collapsed: false,
      items: [
        { text: 'GPG 簽章', link: 'batch-sign' },
        { text: '批量修改提交訊息', link: 'batch-rename-commit-message' },
        { text: '同步提交時間與作者資訊', link: 'sync-git-commit-author-times' },
        { text: '移除圖片檔案', link: 'filter-repo-image-files' },
      ],
    },
  ]
}

function nav(): DefaultTheme.NavItem[] {
  return [
    {
      text: '勉強及格',
      link: '/basic/intro',
      activeMatch: '/basic/',
    },
    {
      text: '八十分',
      link: '/core/intro',
      activeMatch: '/core/',
    },
    {
      text: '超過一百分',
      link: '/pro/intro',
      activeMatch: '/pro/',
    },
    {
      text: '疑難排解',
      link: '/help/removing-sensitive-data',
      activeMatch: '/help/',
    },
  ]
}

function getSocialLinks() {
  return [
    {
      icon: 'github',
      link: 'https://github.com/ZhenShuo2021/docs',
      ariaLabel: 'ZhenShuo2021 GitHub',
    },
    // {
    //   icon: {
    //     svg: '<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><path d="M21 5L18.5 20"/><path d="M21 5L9 13.5"/><path d="M21 5L2 12.5"/><path d="M18.5 20L9 13.5"/><path d="M2 12.5L9 13.5"/><path d="M12 16L9 19M9 13.5L9 19"/></g></svg>',
    //   },
    //   link: 'https://t.me/',
    //   ariaLabel: 'Telegram',
    // },
    {
      icon: {
        svg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><path d="M11 5H5V19H19V13"/><path d="M13 11L20 4"/><path d="M21 3H15M21 3V9"/></g></svg>',
      },
      link: 'https://zsl0621.cc',
      ariaLabel: '主站',
    },
  ]
}

export { themeConfig }
