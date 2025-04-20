---
title: Git 命令行體驗優化：行前準備
sidebar_label: 命令行優化：行前準備
slug: /git-bash-setup-in-windows
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-04-19T02:06:00+08:00
  author: zsl0621
first_publish:
  date: 2025-04-19T02:06:00+08:00
---

Git 其實有很多設定可以客製化，連 git branch 顏色、git diff 工具都可以自訂，但是最常用且最實用的應該是別名系統，本文不是只教你使用別名，而是教你怎麼直接從難用的 CMD/PowerShell 中解脫，直接在 Windows 裡使用經過筆者大量優化的 Zsh。

這裡假設你是 Windows 系統，Unix (Mac) 系統不需要看這篇文章準備直接改就好了，因為此客製化就是要在 Windows 裡面把 Git Bash 改的像 Unix 的 ZSH 一樣，Unix 用戶請直接跳過本文，直接進入[設定環節](/git/advanced-settings-and-aliases)。

## 觀念說明

畢竟 Git 就是從 Linux 世界出來的當然是 Linux 和他最相容，我曾經想嘗試在 Windows CMD/PowerShell 把他改的跟 Zsh 一樣但是徒勞無功，因為這兩個就是不同世界的東西。我們的修改步驟包含

1. 改用更好的終端機介面 (Windows Terminal)
2. 並且把 shell 改為 Git Bash，到此為止都算是標準操作還沒開始客製化
3. 套用我的 Shell 設定完成客製化

自吹一下，我的設定應該是速度最快且功能最多，因為 Git Bash 就是透過轉譯在 Windows 模擬 Linux 環境，然後網路教學又用了很慢的 Oh-My-Zsh，當你[慢到受不了](https://www.v2ex.com/t/1004868)想改又會發現自己是 Windows 系統遇到問題很難除錯 Linux 問題，所以這裡直接提供你一套正確、完整、快速、包含指令補全的設定方式。

### 什麼是 Shell

shell 顧名思義就是殼，對應的內核就是作業系統，意思是我們需要透過這個「殼」來和作業系統溝通，而不是直接存取作業系統的核心。

不同系統有不同的 shell，一個系統也會有多個 shell，例如 Windows 早期的 CMD 和後來的 PowerShell，以及 Unix 世界有 C shell/Bash/Z shell (Zsh)/Fish shell，甚至還有跨平台的 Nushell。

### 什麼是終端機模擬器

Shell 講白了就是一個命令解析器，而終端機模擬器（Terminal Emulator）則是我們操作 shell 的介面。模擬的意思是電腦早期是真的有硬體終端機，現在用軟體模擬所以稱作終端機模擬器，綜合以上說明，我們的目的是從難用的 CMD/PowerShell 中解脫，改為使用 Git Bash 作為 shell，並且在一個更好的介面 (終端機模擬器) 進行操作。

這裡我們會以 Windows Shell 作為範例，因為這網路上資源最多你最容易除錯，其他的終端機如 Alacritty/WezTerm 都要寫設定檔新手很難上手，Tabby 你可以嘗試使用看看，Warp 則見仁見智，如果你想玩這些終端可以參考我的文章[終端機大對決](/memo/useful-tools/cross-platform-terminal-comparison)。

> 你也可以在 Windows WSL 模擬一個 Linux 系統來獲得更好體驗，但是本文不過度延伸。

## 行前準備

圖文教學請參考[這篇 Medium 文章](https://medium.com/la-vida-tech-wacare/windows-terminal-git-bash-zsh-oh-my-zsh-c120ffe61f7c)，我不想再重複擷圖一次，只須看到**字體設定**這個章節即可，我們會使用更好的字體。

### Git Bash

確保安裝 Git 時有按照[安裝與設定](/git/installation)裡面說的勾選 `Add a Git Bash Profile to Windows Terminal`，如果沒有請重新安裝一次。

### MSYS2

安裝 [MSYS](https://packages.msys2.org/packages/zsh?repo=msys&variant=x86_64)，進入頁面後下載檔案 (xxx-x86_64.pkg.tar.zst)，解壓縮後覆蓋在 Git 安裝路徑，預設為 `C:\Program Files\Git`。

### Meslo NF

我們等一下會安裝 Powerlevel10k 主題，該作者甚至自己做了字體優化顯示效果，所以請用他的字體，到[此頁面](https://github.com/romkatv/powerlevel10k/blob/master/font.md)下載 MesloLGS NF ttf，框選後右鍵安裝，最後根據 Medium 說明的安裝方式設定 Windows Terminal 的字體。

## 下一步

現在我們才剛結束第一階段的終端機優化步驟，還沒進入優化 Git，我知道這滿痛苦的，當初從零設定自己也是搞了很久，請見下一篇文章說明 alias 如何設定。
