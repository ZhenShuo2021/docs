// const imageGit = require("@site/docs/git/10-preliminaries/data/areas_upscayl.webp").default;
import imageGit from "@site/docs/git/10-fundamental/10-preliminaries/data/areas_upscayl.webp";
import imageShell from "@site/src/navigation/defaultBackground/shell.webp";
import imageGrayScale from "@site/src/navigation/defaultBackground/grayscale.webp";
import imageProgramming from "@site/src/navigation/defaultBackground/program.webp";

// 新增方式
// 1. 新增 xxxMeta
// 2. 更新 sectionsMeta, 排序影響顯示順序
// 3. 更新 getNavigationContent

export const articlesMeta = [
  // 完整範例，最少要填入 link，其餘會自動尋找
  // {
  //   id: 0,
  //   title: "卡片標題",
  //   description: "卡片描述",
  //   link: "網址",
  //   tags: ["標籤1", "標籤2"],
  //   image: "https://picsum.photos/600/300?random=7",
  // },
  {
    link: "/memo/useful-tools/cross-platform-terminal-comparison",
    image: imageShell,
  },
  {
    description: "應該是繁體中文唯一一篇帶你寫自動補全的文章",
    link: "/memo/linux/custom-zsh-completion",
    image: imageShell,
  },
  {
    title: "0.03 秒啟動極速 Zsh",
    description: "一鍵安裝快到誇張的 zsh 設定檔！",
    link: "/memo/linux/fastest-zsh-dotfile",
    image: imageShell,
  },
  {
    description: "最快、最正確、最完整的 Numba 教學",
    link: "/python/numba-tutorial-1",
    image: imageProgramming,
  },
  {
    link: "/python/numba-performance-test",
    image: imageProgramming,
  },
  {
    link: "/python/best-python-project-manager",
    description:
      "你知道 Python 有超過 10 個專案管理工具嗎？網路教學文章總是劈哩啪啦說一堆指令，卻沒有回答我心中的問題：我該怎麼選擇？",
    image: imageProgramming,
  },
  { link: "/python/uv-project-manager-2", image: imageProgramming },
  {
    description:
      "冷知識：網路教學 rebase onto 的文章十篇有九篇是錯的，本文就是那篇正確的教學",
    link: "/git/rebase-onto",
    image: imageGit,
  },
  {
    link: "/git/reduce-size-with-sparse-checkout",
    image: imageGit,
  },
];

export const projectsMeta = [
  {
    title:
      "Deep Neural Network Based Active User Detection for Grant-Free Multiple Access",
    description: "基本上從頭到尾都由我獨立完成的 Journal 論文",
    link: "https://ieeexplore.ieee.org/document/10496268",
    tags: ["稀疏訊號處理"],
    image: imageGrayScale,
  },
  {
    title: "微圖坊下載器",
    description: "功能豐富的跨平台命令行下載器，Windows, macOS, Linux 都可運行",
    link: "https://github.com/ZhenShuo2021/V2PH-Downloader",
    image: imageGrayScale,
    tags: ["Python"],
  },
  {
    title: "我的超快 Zsh Dotfiles",
    description: "特色是啟動速度超快、支援一鍵安裝並且功能齊全",
    link: "https://github.com/ZhenShuo2021/dotfiles",
    image: imageGrayScale,
    tags: ["Zsh"],
  },
  {
    title: "我的部落格",
    description:
      "使用 Hugo Blowfish 建立的部落格，PageSpeed Insights 測試四項滿分",
    link: "https://www.zsl0621.cc/",
    tags: ["Hugo", "Blowfish"],
    image: imageGrayScale,
  },
  {
    title: "我的個人文檔",
    description: "使用 Docusaurus 建立的文檔，最大特點是內容正確",
    link: "/",
    image: imageGrayScale,
    tags: ["Docusaurus"],
  },
  {
    title: "巴哈姆特黑名單工具 + 清單",
    description:
      "彈幕總是有人在抱怨和暴雷，這個腳本可以匯入和匯出黑名單，還你乾淨彈幕體驗",
    link: "https://github.com/ZhenShuo2021/baha-blacklist",
    image: imageGrayScale,
    tags: ["Python"],
  },
  {
    link: "/memo/python/water-stations-map",
    description:
      "發現自己整整三年都跑去更遠的加水站後直接破防，太生氣了所以弄了一個地圖",
    image: imageGrayScale,
    tags: ["Python", "folium"],
  },
  {
    title: "LazyKit for Python",
    description: "與其說 kit 更像是 cheat sheet ，順便測試用 mkdocs 架設網站",
    link: "https://github.com/ZhenShuo2021/lazykit",
    image: imageGrayScale,
    tags: ["Python"],
  },
];

export const sectionsMeta = [
  {
    id: "projectsID",
    title: "🚀 專案",
    buttonText: "瀏覽專案",
    contentType: "專案",
  },
  {
    id: "articlesID",
    title: "✍️ 文章",
    buttonText: "查看文章",
    contentType: "文章",
  },
];
