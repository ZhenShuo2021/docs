import imageGit from "../../docs/docs/git/10-preliminaries/data/areas_upscayl.webp";
import imageShell from "../pages/portfolio/defaultBackground/shell.webp";
import imageProgramming from "../pages/portfolio/defaultBackground/program.webp";

export const articles = [
  // 完整範例，最少要填入 link，其餘會自動尋找
  // {
  //   id: 0,
  //   title: "卡片標題",
  //   description: "卡片描述",
  //   link: "網址",
  //   tags: ["標籤1", "標籤2"],
  //   image: "https://picsum.photos/600/300?random=7",
  //   // 取得docs中圖片的方式
  //   image: require("@site/docs/docs/git/10-preliminaries/data/branch.webp")
  //     .default,
  // },
  {
    link: "/docs/useful-tools/cross-platform-terminal-comparison",
    image: imageShell,
  },
  {
    description: "應該是繁體中文唯一一篇帶你寫自動補全的文章",
    link: "/docs/linux/custom-zsh-completion",
    image: imageShell,
  },
  {
    title: "0.04 秒啟動極速 Zsh",
    description: "一鍵安裝快到誇張的 zsh 設定檔！",
    link: "/docs/linux/fastest-zsh-dotfile",
    image: imageShell,
  },
  {
    link: "/docs/python/virtual-environment-management-comparison",
    image: imageProgramming,
  },
  { link: "/docs/python/python-uv-complete-guide", image: imageProgramming },
  {
    description: "最快、最正確、最完整的 Numba 教學",
    link: "/docs/python/numba-tutorial-accelerate-python-computing",
    image: imageProgramming,
  },
  {
    link: "/docs/python/numba-performance-benchmark-svml-signal-processing",
    image: imageProgramming,
  },
  {
    description:
      "網路教學 rebase onto 的文章十篇有九篇是錯的，本文最大的特色是內容正確",
    link: "/docs/git/advance/rebase-onto",
    image: imageGit,
  },
  {
    link: "/docs/git/advance/reduce-size-with-sparse-checkout",
    image: imageGit,
  },
];

export const projects = [
  {
    title:
      "Deep Neural Network Based Active User Detection for Grant-Free Multiple Access",
    description: "基本上從頭到尾都由我獨立完成的 Journal 論文",
    link: "https://ieeexplore.ieee.org/document/10496268",
    tags: ["稀疏訊號處理"],
    // image: "https://picsum.photos/600/300?random=2332",
  },
  {
    title: "微圖坊下載器",
    description: "功能豐富的跨平台命令行下載器，Windows, macOS, Linux 都可運行",
    link: "https://github.com/ZhenShuo2021/V2PH-Downloader",
    tags: ["Python"],
  },
  {
    title: "我的超快 Zsh Dotfiles",
    description:
      "特點是啟動速度超快、一鍵安裝還有正確，你沒看錯，網路文章錯誤多到可以把正確當賣點",
    link: "https://github.com/ZhenShuo2021/dotfiles",
    tags: ["Zsh"],
  },
  {
    title: "我的部落格",
    description:
      "使用 Hugo Blowfish 建立的部落格，PageSpeed Insights 測試四項滿分",
    link: "https://github.com/ZhenShuo2021/ZhenShuo2021.github.io",
    tags: ["Hugo", "Blowfish"],
    image:
      "https://www.zsl0621.cc/img/1111830258087415808_3-5-5x-downsample.webp",
  },
  {
    title: "我的個人文檔",
    description: "使用 Docusaurus 建立的文檔，最大特點是內容正確",
    link: "https://github.com/ZhenShuo2021/docs",
    tags: ["Docusaurus"],
  },
  {
    title: "巴哈姆特黑名單工具 + 黑名單合輯",
    description:
      "彈幕的臭嘴太多了，這個腳本可以匯入和匯出巴哈姆特黑名單，支援各種資料來源",
    link: "https://github.com/ZhenShuo2021/baha-blacklist",
    tags: ["Python"],
  },
  {
    link: "/memo/python/water-stations-map",
    description:
      "發現自己整整三年都跑去更遠的加水站後直接破防，太生氣了所以弄了一個地圖",
    image:
      "https://raw.githubusercontent.com/ZhenShuo2021/blog-script/refs/heads/main/python/hsinchu-water-station/data/water-station.jpg",
    tags: ["Python", "folium"],
  },
  {
    title: "Lazy kit for Python",
    description:
      "與其說是 kit 更不如說是 cheat sheet ，順便練習用 mkdocs 寫文檔",
    link: "https://github.com/ZhenShuo2021/lazykit",
    tags: ["Python"],
  },
];

