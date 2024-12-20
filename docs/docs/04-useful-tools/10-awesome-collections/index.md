---
title: 實用工具選輯
description: 選輯
sidebar_label: 選輯
tags:
  - 實用工具
keywords:
  - 實用工具
last_update:
  date: 2024-12-17T03:35:00+08:00
  author: zsl0621
first_publish:
  date: 2024-12-17T14:22:30+08:00
---

# 工具選輯

把我自己挑選過在同一領域內最好的工具選出來，因為一定會忘記所以存起來

## Python 實用套件

- pathvalidate: 處理檔案路徑的套件
- DownloadKit: 封裝 requests 的下載工具，內建自動副檔名偵測，自動重新命名  
設定範例

```py
dk = DownloadKit()
dk.set.log.print_nothing()
dk.set.roads(5)
dk.set.retry(5)
dk.set.interval(5)
dk.set.timeout(10)
dk.set.split(True)
dk.set.block_size("1m")
```

- python-fire: 自動生成 CLI 介面
- [HTTP 客戶端、網頁自動化、HTML 解析套件集合](https://blog.zsl0621.cc/posts/scrappers/)
- Ruff: 用 Rust 寫的開源 Python linter + formatter
- uv: 用 Rust 寫的開源 Python 虛擬環境+套件+Python 版本管理

## CLI 工具

- yt-dlp: 實用 CLI 影片下載工具
- gallery-dl: 實用 CLI 圖片下載工具

## 瀏覽器擴充功能

### 還我乾淨瀏覽

- uBlock Origin
- uBlacklist

<details>
<summary>uBlacklist 封鎖清單</summary>

備份也當作分享，可以直接匯入（裡面叫做還原）

```json
{
  "blacklist": "*://blog.csdn.net/*\n*://cloud.baidu.com/*\n*://readforbetter.com/*\n*://www.zhihu.com/*\n*://ithelp.ithome.com.tw/*\n*://wenku.csdn.net/*\n*://www.kaixinit.com/*\n*://www.imooc.com/*\n*://cloud.tencent.com/*\n*://www.delftstack.com/*\n*://codelove.tw/*\n*://jujupp.medium.com/*\n*://hugo-for-newbie.kejyun.com/*\n*://sean22492249.medium.com/*\n*://www.cs.pu.edu.tw/*\n\n*://gitbook.tw/*\n*://python.libhunt.com/*\n*://codimd.mcl.math.ncu.edu.tw/*\n*://skyyen999.gitbooks.io/*\n*://python.plainenglish.io/*\n*://python.iswbm.com/*\n*://medium.com/@heidi-coding/*\n*://medium.com/@chenfelix/*\n*://www.readfog.com/*\n*://m.php.cn/*\n*://www.sohu.com/*\n*://segmentfault.com/*\n*://juejin.cn/*\n*://ftp.tku.edu.tw/*\n*://www.rapidseedbox.com/*\n*://medium.com/@tonykuoyj*\n*://2formosa.blogspot.com/*\n\n*://huaweicloud.csdn.net/*\n*://www.threads.net/*\n*://www.businessweekly.com.tw/*\n*://s.csdnimg.cn/*\n@*://shopee.tw/*\n@*://term.ptt.cc/*\n@*://www.ptt.cc/*\n@*://www.cnblogs.com/*\n*://newsn.net/*",
  "blockColor": "default",
  "blockWholeSite": false,
  "dialogTheme": "default",
  "enablePathDepth": false,
  "hideBlockLinks": false,
  "hideControl": false,
  "highlightColors": [
    "#ddeeff"
  ],
  "linkColor": "default",
  "skipBlockDialog": false,
  "subscriptions": [
    {
      "name": "標準內容農場清單 danny0838",
      "url": "https://danny0838.github.io/content-farm-terminator/files/blocklist-ublacklist/content-farms.txt",
      "enabled": true
    },
    {
      "name": "類內容農場清單 danny0838",
      "url": "https://danny0838.github.io/content-farm-terminator/files/blocklist-ublacklist/nearly-content-farms.txt",
      "enabled": true
    },
    {
      "name": "擴充內容農場清單 danny0838",
      "url": "https://danny0838.github.io/content-farm-terminator/files/blocklist-ublacklist/extra-content-farms.txt",
      "enabled": true
    },
    {
      "name": "劣質複製農場清單 danny0838",
      "url": "https://danny0838.github.io/content-farm-terminator/files/blocklist-ublacklist/bad-cloners.txt",
      "enabled": true
    },
    {
      "name": "詐騙網站清單 danny0838",
      "url": "https://danny0838.github.io/content-farm-terminator/files/blocklist-ublacklist/scam-sites.txt",
      "enabled": true
    },
    {
      "name": "假新聞網站清單  danny0838",
      "url": "https://danny0838.github.io/content-farm-terminator/files/blocklist-ublacklist/fake-news.txt",
      "enabled": true
    },
    {
      "name": "eallion",
      "url": "https://git.io/ublacklist",
      "enabled": false
    }
  ],
  "syncAppearance": true,
  "syncBlocklist": true,
  "syncGeneral": true,
  "syncInterval": 15,
  "syncSubscriptions": true,
  "updateInterval": 120,
  "version": "8.9.2"
}
```

</details>

### 影片相關

- YouTube 繁體自動翻譯修正
- Screenshot YouTube
- Better YouTube Shorts
- Allow Right-Click
- Return YouTube Dislike
- YouTube NonStop
- Video Speed Controller

### 工具類型

- Cookie-Editor
- GoFullPage
- Joplin Web Clipper
- Picture-in-Picture Extension (by Google)
- PixivBatchDownloader

### 個人客製化

- Bonjourr · Minimalist Startpage

## 終端機相關

- [helix](https://github.com/helix-editor/helix): 開箱即用的文字編輯器，theme onedarker ([keymap](https://docs.helix-editor.com/keymap.html), [commands](https://docs.helix-editor.com/commands.html))
- Warp: 附帶各種功能、支援各種工具的套件
- [wezterm-config](https://github.com/KevinSilvester/wezterm-config): 開箱即用的 WezTerm 客製化，截至 6febb08 設定都很好。zsh 使用者要把 `config/domains.lua` 裡面的 fish 改成 zsh
- [yazi](https://github.com/sxyazi/yazi): 超快的命令行檔案檢視器，需要用支援[圖形輸出](https://yazi-rs.github.io/docs/image-preview/)的終端才可以顯示圖片，例如 WezTerm
- WinGet/Chocolatey: Windows 的終端機套件管理
- UniGetUI: Windows 上的圖形化套件管理器
- [delta](https://github.com/dandavison/delta): 可以像是 Vscode 一樣 highlight git diff
- [lazygit](https://github.com/jesseduffield/lazygit): Git TUI

## 自架 Self-Host

- Immich: 完全替代 Google 相簿
- PhotoPrism: 畫廊形式的相簿
- [filebrowser](https://github.com/filebrowser/filebrowser): 簡單易用輕量的個人雲端，包含帳號功能
- Stirling PDF: PDF 全能工具
- Lanraragi: 漫畫伺服器
- Stash: 影片伺服器

## 還沒有分類

- RustDesk: 用 Rust 寫的開源遠端工具
- [Files](https://github.com/files-community/Files): 改善 Windows 內建的檔案瀏覽器
- 