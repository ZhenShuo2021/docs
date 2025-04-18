---
title: Shell 和他的設定檔，以及 Dotfiles
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

老話一句，沒啥文章能簡潔有力的介紹，能用一句話講完的就不要寫成一篇文章。

## 什麼是 Shell

Shell 是和作業系統互動的介面，所以人如其名，他是使用者操作作業系統所接觸到的那層「殼」，殼裡面是作業系統。

有非常多不同的 shell 可以選擇，從 `C Shell`、`Korn Shell`、`Bash` `Zsh`，甚至 `Fish Shell` `pwsh` `cmd`，我們最常碰到的是 Bash 和 Zsh 都是 Unix shell，Zsh 又可以視為 Bash 的超集，提供了更多現代特性和更好的定制性，所以本文只講 Zsh。

> Fish 則比較特別，他是給你方便操作用的，速度超快、內建功能多，但是指令不兼容其他 Unix shell。

## Shell 設定檔

每次 shell 開啟前都會載入不同文件，順序如下

:::info Zsh 載入順序

`.zshenv` → `.zprofile` → `.zshrc` → `.zlogin` → `.zlogout`

:::

每個文件的載入時機有所不同，直接上表格

| 文件名       | 非交互式 Shell（如腳本） | 非登入 Shell | 交互式但非登入 Shell  | 使用 `zsh` | 使用 `zsh -c "echo 123"` |
|--------------|--------------------------|--------------|----------------|------|---------------|
| `.zshenv`    | 載入                     | 載入         | 載入             |  載入  | 載入        |
| `.zprofile`  | 不載入                   | 不載入       | 不載入            |  不載入 | 不載入        |
| `.zshrc`     | 不載入                   | 載入         | 載入              |  不載入 | 不載入        |
| `.zlogin`    | 不載入                   | 不載入       | 不載入            |  不載入 | 不載入        |
| `.zlogout`   | 不載入                   | 不載入       | 不載入            |  不載入 | 不載入        |

使用 `zsh --no-rcs` 可以不載入任何文件。因為項目很多所以表格很複雜，網路上也講了很多不同文件放什麼，我整理的心得是**其實只分為三種**

1. `zshenv` 基本上總是載入，修改時要非常小心，執行腳本時也會載入 `zshenv`
2. `.zprofile` `.zlogin` `.zlogout` ***三者相同，只是載入時間點不同***，並且 macOS 的 `.zprofile` 行為和一般的 Linux 發行版略有不同
3. `.zshrc` 隨便改，在登入和交互式時載入此文件

除了用戶本身的設定檔，系統還會提前載入 `/etc/zshenv` 和 `/etc/zshrc`。

相關資訊如下：

- [關於 Linux 下 Bash 與 Zsh 啟動檔的載入順序研究](https://blog.miniasp.com/post/2021/07/26/Bash-and-Zsh-Initialization-Files)
- [Setting $PATH for zsh on macOS.md](https://gist.github.com/Linerre/f11ad4a6a934dcf01ee8415c9457e7b2)
- [What should/shouldn't go in .zshenv, .zshrc, .zlogin, .zprofile, .zlogout?](https://unix.stackexchange.com/questions/71253/what-should-shouldnt-go-in-zshenv-zshrc-zlogin-zprofile-zlogout)
- [ZSH: .zprofile, .zshrc, .zlogin - What goes where?](https://apple.stackexchange.com/questions/388622/zsh-zprofile-zshrc-zlogin-what-goes-where)
- [sambacha/dotfiles2](https://github.com/sambacha/dotfiles2)

## zshrc

`.zshrc` 你想在裡面做什麼都行，不過主要的設定有以下幾項

1. 設定別名：簡化常用命令，例如加入 `alias gloga="git log --graph --pretty='%Cred%h%Creset -%C(auto)%d%Creset %s %Cgreen(%ar) %C(bold blue)<%an>%Creset' --all"`，以後就可以輕鬆使用縮寫
2. 設定函數：創建自定義命令或自動化任務，例如設定 `export GPG_TTY=$(tty)` 這樣每次開啟終端都會自動刷新 `GPG_TTY`
3. 設定環境變數：如新增系統路徑
4. 設定主題和插件：可以設定 Zimfw, zsh4humans 等插件管理器，這樣就會有漂亮且方便的終端

> 附帶一提，插件管理器不要用 Oh-My-Zsh，網路上教 omz 的都是萌新！

## Dotfiles

在 Unix 系統中大部分的設定檔都以 `.` 開頭，所以稱作 dotfiles。管理 dotfiles 可以讓你同步和備份各種設定，都不用特別說什麼，光是回想每個語言、套件、IDE、工具函式花了多少時間安裝和設定就知道管理這些設定檔有多重要。

除了在 Github 上搜尋 dotfiles 以外也可以找 awesome dotfiles，或者使用[我的設定檔](fastest-zsh-dotfile)，特色是速度超快、功能齊全而且設定正確（網路上一大堆亂設定的）。
