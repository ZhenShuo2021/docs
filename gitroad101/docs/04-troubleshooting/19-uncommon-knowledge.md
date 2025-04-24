---
title: Git 那些很少用到的知識
slug: /uncommon-knowledge
sidebar_label: 很少用到的知識
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-04-25T11:27:33+08:00
  author: zsl0621
first_publish:
  date: 2025-04-19T01:02:07+08:00
---

# {{ $frontmatter.title }}

本文不廢話的記錄看到的資訊，如你所見我不是專家，我只是看到文章並且簡單記錄，本文有簡單的也有難的，但是共通點是 99.9\% 的人用不到這些知識。

## .gitattributes

由於太少用了所以放在這麼後面才介紹，而且這些問題通常前人也已經幫你處理好了。此檔案和 .gitignore 一樣放在專案根目錄，用來控制 Git 對特定檔案的行為，例如：

- 設定文字檔換行符號 (crlf 和 lf)
- 指定合併策略
- 過濾 LFS 檔案

比如說可以設定讓 Git 知道要[如何判斷 word 檔案](https://iissnan.com/progit/html/zh-tw/ch7_2.html)，Git 會根據它的規則處理，專文介紹請見[使用git attributes正規化專案換行字元](https://www.astralweb.com.tw/use-git-attributes-normalization-project-for-break-line/)。

## 儲存庫結構

專案很大有多種不同的結構選擇，如 Monorepo, Multi-Repo, Monolith，簡單來說：

- **Monorepo**：一個儲存庫，多個專案共存  
- **Multi-Repo**：每個專案一個儲存庫，互相分離  
- **Monolith**：所有功能寫在同一個應用程式內，部署為單一單位

然而會處理到這個問題都是 leader 級別，所以不需要我來寫這個問題。

## Git 工作流程

大家最常見的 Git 工作流程肯定是 Git Flow，因為大部分教學第一個就會放他，實際上當然不只這一種，還有 GitHub Flow、GitLab Flow、One Flow 等等，根據自己的需求選擇，專文介紹請見 [TFS Git 筆記 - 分支管理策略](https://blog.darkthread.net/blog/git-branching-strategies/)。

## 超大型儲存庫

很多東西都會量變引起質變，儲存庫也是，當儲存庫有千萬級別的提交數量，6,000 萬行程式碼時，git status 也可能變成需要耗時 15 秒才能執行的指令，這是Canva 的情景：[We Put Half a Million files in One git Repository, Here's What We Learned](https://www.canva.dev/blog/engineering/we-put-half-a-million-files-in-one-git-repository-heres-what-we-learned/)

| Description | git status time | \% of files in checkout |
| --- | --- | --- |
| Fresh clone of canva monorepo | 10s | 100\% |
| Ignoring all .xlf except en-AU | 3s | 32\% |

不過 Canva 這篇文章標題很聳動內容卻沒寫什麼，花了一半的篇幅說明不要把不會動到的 `.xlf` 檔案簽出到儲存庫中，然後吹了一下自己使用內部的開發工具監控就沒了，比較有用的是微軟的這篇文章 [The largest Git repo on the planet](https://devblogs.microsoft.com/bharry/the-largest-git-repo-on-the-planet/)，文章說明一個大到嚇人的工作場景：*4,000 名工程師，每天在 440 個分支上產生 1,760 個“實驗室版本”*，最後介紹自己開發的開源加速工具 [VFSForGit](https://github.com/microsoft/VFSForGit)。

VFSForGit 的問題主要是他提供一個虛擬層，這會造成一些問題，最簡單的就是 GUI 不能用，所以又有一個 [scalar](https://github.com/microsoft/scalar) 工具出來。

除此之外 Git 也有一些神奇的內建工具 [git maintenance](https://git-scm.com/docs/git-maintenance) 和 [git fsmonitor--daemon](https://git.js.cn/docs/git-fsmonitor--daemon)，我的知識淺薄不獻醜，拋磚引玉如果熟悉歡迎補充。
