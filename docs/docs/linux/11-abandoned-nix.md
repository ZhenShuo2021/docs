---
title: Nix 套件管理：我的失敗經驗
tags:
  - Linux
  - Nix
keywords:
  - Linux
  - Nix
last_update:
  date: 2025-03-07T21:16:30+08:00
  author: zsl0621
first_publish:
  date: 2025-03-07T21:16:30+08:00
---

本文原本是在補充前一篇 Ubuntu 套件管理器時臨時起意加上的，寫完後覺得把他放成一篇文章好了。

裡面在罵的不是 Nix，而是介紹 Nix 的中文文章，介紹 Nix 的中文文章大部分都很爛。

## Nix

Nix 是非常特別的套件管理系統，而且不只是在 NixOS 本身能使用，macOS 也能將其作為套件管理器。筆者曾經嘗試使用但最後放棄了，這是我的研究結果，因為 Google Search 排序前面的文章糟糕透頂就自己寫一篇，連介紹特點都不會還寫什麼文章，純粹在浪費別人時間。

Nix 的特點主要有幾個

1. 純函數式套件管理
2. 原子性更新與回滾
3. 不會遇到版本衝突，多版本共存

初學者看到第一個特點會覺得很有趣，然後再去網路上查那些稀巴爛的垃圾文章就會打退堂鼓了，我不知道未來的搜尋結果會是怎樣，至少在我研究的當下絕大多數文章都在講廢話甚至是錯誤資訊。回到特點介紹，函數式的意思要回到國中數學，函數一個輸入只會有一種輸出，所以 Nix 的函數式代表他的配置文件只要你寫了配置文件加上其 lock 檔，就絕對只會有這個配置，從套件管理延伸到整個作業系統都是這個概念，原子性同樣也是建築在函數式的原理之上。

版本衝突和共存多版本共存的原理是 Nix 每個套件都會以 `hash-套件名稱` 作為 symlink，所以即使兩個套件的依賴版本完全衝突沒有交集，使用這種機制就可以取用各自的套件，於是永遠不會遇到套件衝突。

介紹完優點接下來要開始瘋狂輸出缺點了，第一個問題，他使用自己的程式語言寫設定檔 (/etc/nixos/configuration.nix)，新手超級無敵不友善，你能想像安裝套件前還要先學會語言嗎？雖然沒有到很困難但就是一個門檻。

第二個問題也是我放棄的原因，我想改用 Nix 的原因是想要使用舊版套件，這是 homebrew 不允許的，homebrew 使用舊版套件的方式越來越麻煩，這就是我在 macOS 上嘗試 Nix 的原因，然而根據[此篇討論](https://www.reddit.com/r/Nix/comments/1iqtwtj/comment/md36gu6/?context=3)，要下載舊版套件你首先要看一個[長到誇張的文檔](https://nixos.org/manual/nixpkgs/stable/#chap-overrides)，這個文檔甚至不是在說怎麼安裝舊版套件而是在說 Nix 語言，新手不問人是**幾乎不可能直接從文檔找到解決方法**。

第三個問題，文檔很糟糕，不是說文檔太少，而是文檔過多，這裡列出我找到的文檔：

- [NixOS Manual](https://nixos.org/manual/nixos/stable/)
- [NixOS & Flakes Book \| 主页](https://nixos-and-flakes.thiscute.world/zh/)
- [主页 \| NixOS 中文](https://nixos-cn.org/)
- [Welcome to nix.dev — nix.dev documentation](https://nix.dev/)
- [Nixpkgs Reference Manual](https://nixos.org/manual/nixpkgs/stable/)
- [Home Manager Manual](https://nix-community.github.io/home-manager)
- [Zero to Nix](https://zero-to-nix.com/)

最離奇的是即便有這麼多文檔，還是沒有任何一個文檔提到問題二的解法。

第四個問題，工具過多並且介紹不清楚，有舊版的工具還有新版的 nix flakes 等等全部混在一起，理論上直接使用新版就好，但是你就要花心力過濾舊版資訊。除了新舊版本問題，home-manager, Nixpkgs, nix profile, devbox 等等各種不同工具也是全部混在一起，非常考驗你的資訊整理能力。

第五個問題，報錯基本上沒有，這在[社群文檔](https://nixos-and-flakes.thiscute.world/zh/introduction/advantages-and-disadvantages)裡面就有提到了，而且我們都已經習慣以往的錯誤處理方式，在這裡報錯訊息全部都要自己手動通靈。我想跳槽的原因是套件版本問題，既然都要解決衝突，那我為什麼不手動裝 binary 就好了，用我熟悉的方式解決不是更快嗎。

最後一個問題，我才裝了幾個套件安裝時間就已經比 Homebrew 還久了，網路上有避免重複編譯還是設定快取的方式，但是經過前面五個缺點之後我已經累了，這是最後一根稻草，決定放棄 Nix 套件管理器。

補充一個優點，官網的移除教學非常乾淨，至少沒有留下奇怪的東西在我的電腦裡。

## 有用資源

最後附上一些有用資源，因為無用資源太多，絕大部分是中文文章無用。

### 文檔

- [NixOS Manual](https://nixos.org/manual/nixos/stable/)
- [NixOS & Flakes Book \| 主页](https://nixos-and-flakes.thiscute.world/zh/)
- [主页 \| NixOS 中文](https://nixos-cn.org/)
- [Welcome to nix.dev — nix.dev documentation](https://nix.dev/)
- [Nixpkgs Reference Manual](https://nixos.org/manual/nixpkgs/stable/)
- [Home Manager Manual](https://nix-community.github.io/home-manager)

### 模板

- [GitHub - ryan4yin/nix-darwin-kickstarter: macOS as Code! A beginner-friendly nix-darwin + home-manager + flakes startup configuration. 一份易于理解的 nix-darwin 初始配置模板，专为新手制作.](https://github.com/ryan4yin/nix-darwin-kickstarter)
- [GitHub - LnL7/nix-darwin: nix modules for darwin](https://github.com/LnL7/nix-darwin)
- [GitHub - DeterminateSystems/nix-installer: Install Nix and flakes with the fast and reliable Determinate Nix Installer, with over 7 million installs.](https://github.com/DeterminateSystems/nix-installer)

### 說明文章

- [NixOS: 选择与放弃 - rqdmap \| blog](https://rqdmap.top/posts/nixos/)
- [Nix 和 NixOS：你们安利方法错了 - Nayuki's Archive](https://nyk.ma/posts/nix-and-nixos/)
- [Recommendation: nix profile vs nix-env : r/Nix](https://www.reddit.com/r/Nix/comments/1buqjau/recommendation_nix_profile_vs_nixenv/)
