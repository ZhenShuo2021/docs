---
title: Shell 和他的設定檔，以及 Dotfiles
description: Shell 和他的設定檔，以及 Dotfiles
sidebar_label: Shell and Dotfiles
tags:
  - Linux
  - Terminal
keywords:
  - Linux
  - Terminal
last_update:
  date: 2024-12-29T19:49:30+08:00
  author: zsl0621
first_publish:
  date: 2024-12-29T19:49:30+08:00
---

# Shell 和他的設定檔，以及 dotfiles

又是本文檔最常說的一句話，沒啥文章能簡潔有力的介紹，老話一句，能用一句話講完的就不要寫成一篇文章。

## Shell: bash/zsh

Shell 是和作業系統互動的介面，所以人如其名，他是使用者操作作業系統所接觸到的那層「殼」，殼裡面是作業系統。

而 Bash 和 Zsh 都是 Unix shell，Zsh 可以視為 Bash 的超集，提供了更多現代特性和更好的定制性，本文只講 Zsh。

## Shell 設定檔

每次 shell 開啟前都會載入這些文件：

- `.zprofile` - 登錄 shell 啟動時載入，設置登錄會話變量。
- `.zshenv` - 所有 Zsh 會話啟動時載入，設置全局變量。
- `.zshrc` - 每個交互式 shell 啟動時載入，設置別名、函數等。
- `.zlogin` - 登錄 shell 啟動後載入，執行登錄後設置。
- `.zlogout` - 登錄 shell 退出時載入，清理工作或保存狀態。

基本上我們只會修改 `.zshrc`，根據此 [reddit 討論](https://www.reddit.com/r/zsh/comments/kwmrf4/help_me_understand_best_practices_re/)，`.zshenv` 裡面的設定必須是「不拖慢 shell、需要被所有子進程繼承的變量」，因為所有會話都會載入他，而 `.zshrc` 只有交互式會話才會載入。

這裡不討論載入順序，因為每個發行版的順序不見得完全相同。

## zshrc

舉例來說像是 GPG 需要確認 tty，需要執行 `GPG_TTY=$(tty)`，把這行放在 `.zshrc` 就可以在每次開啟 shell 時自動執行，避免每次都要手動執行的問題。這只是冰山一角，除此之外還可以設定：

1. 設定別名：簡化常用命令，例如 alias git-rm-merged="git branch -d `git branch --merged | grep -v '^*' | grep -v 'main' | tr -d '\n'"
2. 設定函數：創建自定義命令或自動化任務
3. 設定環境變量：如新增系統路徑
4. 設定主題和插件：可以設定 Zimfw, zsh4humans 等插件框架（不要用 Oh-My-Zsh，網路上教 omz 的都是菜雞！）

## Dotfiles

在 Unix 系統中大部分的設定檔都以 `.` 開頭，所以稱作 dotfiles。管理 dotfiles 可以讓你同步和備份各種設定，都不用特別說什麼，光是回想每個語言、套件、IDE、工具函式花了多少時間安裝和設定就知道管理這些設定檔有多重要。

除了在 Github 上搜尋 dotfiles 以外也可以找 awesome dotfiles，或者使用[我的設定檔](./fastest-zsh-dotfile)，特色是設定正確（網路上一大堆亂設定的），功能齊全而且啟動速度超快。
