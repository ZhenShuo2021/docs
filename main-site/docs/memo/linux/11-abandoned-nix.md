---
title: Nix 套件管理器：我的失敗經驗
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

Nix 是非常特別的套件管理系統，而且不只是 NixOS 能用，macOS 也能將其作為套件管理器。筆者曾經嘗試使用但最後放棄了，這篇是我的使用心得和整理。

## Nix

Nix 的特點主要有幾個

1. 純函數式套件管理
2. 原子性更新與回滾
3. 不會遇到版本衝突，多版本共存
4. 社群很活躍，更新頻繁，是全世界最大的套件庫

和 系統級快照/Ansible/Docker 比較的文章請見[文檔FAQ](https://nixos-and-flakes.thiscute.world/zh/faq/)，能和這些完全不同的東西互相比較就可以看出他的獨特之處。

### 優點

函數式的意思要回到國中數學，函數一個輸入只會有一種輸出，所以 Nix 的函數式代表他的配置文件只要你寫了配置文件加上其 lock 檔，就絕對只會有這個配置，從套件管理延伸到整個作業系統都是這個概念，原子性同樣也是建築在函數式的原理之上。

不存在版本衝突以及共存多版本共存的原理是，Nix 每個套件都會以 `hash-套件名稱` 作為 symlink，所以即使兩個套件的依賴版本完全衝突沒有交集，使用這種機制就可以取用各自的套件，於是永遠不會遇到套件衝突。

只要安裝快速、衝突容易解決就已經是非常好用的套件管理器了，更何況 Nix 連衝突都不會遇到，更新回滾都是原子性，有問題重設就好，乍看之下好像非常完美，所以接下來要介紹缺點了。

### 缺點

第一個問題，他使用自己的程式語言寫設定檔 (/etc/nixos/configuration.nix)，新手非常不友善，你能想像安裝套件前還要先學會新的語言嗎？雖然沒有到很困難但就是一個門檻，當然你也可以不使用設定檔直接安裝，但是這樣就完全浪費 Nix 的特點了。

問題二～四是類似的問題，這三個加起來也是我放棄的主因。首先問題二，我想改用 Nix 的主因是想要使用舊版套件，這是 homebrew 不允許的，homebrew 使用舊版套件的方式越來越麻煩，我基本上不會想拿他來安裝舊版套件，然而根據[此篇討論](https://www.reddit.com/r/Nix/comments/1iqtwtj/comment/md36gu6/?context=3)，Nix 要下載舊版套件首先要看一個[長到誇張的文檔](https://nixos.org/manual/nixpkgs/stable/#chap-overrides)，這個文檔甚至不是在說怎麼安裝舊版套件，而是在解釋 Nix 語言，解決方式不直觀，新手不問人是**幾乎不可能直接從文檔找到解決方法**。

第三個問題承續問題二，文檔很糟糕，不是說文檔太少，而是文檔過多，這裡列出我找到的文檔：

- [NixOS Manual](https://nixos.org/manual/nixos/stable/)
- [NixOS & Flakes Book \| 主页](https://nixos-and-flakes.thiscute.world/zh/)
- [主页 \| NixOS 中文](https://nixos-cn.org/)
- [Welcome to nix.dev — nix.dev documentation](https://nix.dev/)
- [Nixpkgs Reference Manual](https://nixos.org/manual/nixpkgs/stable/)
- [Home Manager Manual](https://nix-community.github.io/home-manager)
- [Zero to Nix](https://zero-to-nix.com/)

一個套件管理器有七個文檔，最離奇的是即便有這麼多文檔，還是沒有任何一個文檔提到問題二的解法。

第四個問題，工具過多並且介紹不清楚，有舊版的工具還有新版的 nix flakes 等等全部混在一起，理論上直接使用新版就好，但是你就要花心力過濾舊版資訊。除了新舊版本問題，home-manager, Nixpkgs, nix profile, devbox 等等各種不同工具也是全部混在一起，Nix Package Manager 和 NixOS 也全部混雜，非常考驗你的資訊整理能力。

第五個問題，報錯基本上沒有，這在[社群文檔](https://nixos-and-flakes.thiscute.world/zh/introduction/advantages-and-disadvantages)裡面就有提到了，我們都已經習慣以往的錯誤處理方式，在這裡報錯訊息卻要自己通靈。我想跳槽的原因是套件版本問題，既然都要解決衝突，那我為什麼不用以往熟悉的方式解決呢？

最後一個問題，我才裝了幾個套件安裝時間就已經比 Homebrew 還久了，網路上有避免重複編譯還是設定快取的方式，但是經過前面五個缺點之後我已經累了，這是最後一根稻草，決定放棄 Nix 套件管理器。

一次講了六個缺點之後這裡補充一個優點，官網的移除教學非常乾淨，至少沒有在我的電腦留下奇怪的東西，這個優點有點諷刺。

## 有用資源

附上一些有用資源，因為無用資源太多。

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

### 相關文章

寫這篇文章的動力是網路上表達能力糟糕透頂的中文文章，連把特色說明清楚都做不到，初學者看到 Nix 的特色會覺得很有趣，結果下一步馬上就會被這些文章迷惑。這幾篇是還不錯的文章，學習 Nix 的時候千萬不要從中文文章開始下手，看到最後你還是要回頭找英文文章。

- [NixOS: 选择与放弃 - rqdmap \| blog](https://rqdmap.top/posts/nixos/)
- [Nix 和 NixOS：你们安利方法错了 - Nayuki's Archive](https://nyk.ma/posts/nix-and-nixos/)
- [Recommendation: nix profile vs nix-env : r/Nix](https://www.reddit.com/r/Nix/comments/1buqjau/recommendation_nix_profile_vs_nixenv/)
