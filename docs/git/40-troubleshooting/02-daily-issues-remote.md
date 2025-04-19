---
title: Git 各種日常問題集合 - 遠端
sidebar_label: 日常問題 - 遠端
description: 介紹 Git 常見的本地和遠端問題，包含清除reflog記錄、正確使用rebase、git mv、以及如何加速clone等進階技巧。還解釋了常見錯誤誤導，並提供正確的 Git 操作方法。
slug: /daily-issues
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-02-12T23:19:00+08:00
  author: zsl0621
first_publish:
  date: 2025-02-12T13:35:00+08:00
---

# Git 各種日常問題集合 - 遠端

都是簡單的日常問題但是要花一點時間搜尋，所以這篇文章集中列出方便查詢。

## 遠端追蹤分支是什麼？和遠端分支一樣嗎？追蹤分支又是什麼？{#remote-checking-branches}

遠端追蹤分支 (Remote-tracking Branch) 是本地儲存庫用來記錄遠端分支最新狀態的本地參考，其名稱格式為 `<遠端名稱>/<分支名稱>`，例如預設的 `origin/main`。

執行 `git clone` 後，Git 會自動檢出 (checkout) 一個預設的本地分支，並將其設定為追蹤分支（Tracking Branch），該分支會與對應的遠端追蹤分支建立追蹤關係。例如 `git clone` 後預設檢出的 `main` 分支，會追蹤 `origin/main` 這個遠端追蹤分支，而 `origin/main` 也可稱為 `main` 分支的上游分支（Upstream Branch）。

所謂口語上的遠端分支就是在遠端中的本地分支，和遠端追蹤分支是不同的概念。

<br />

## 無法推送

有兩種可能，遠端分支設定錯誤或者遠端提交歷史比本地還要新。

比本地還新的話就使用 `git pull --rebase`，如果設定跑掉就用[設定遠端分支](#fix-remote-branch)，如果想要覆蓋就使用[安全的強制推送](#安全的強制推送)。

<br />

## 還是無法推送，重設遠端分支{#fix-remote-branch}

請見 Git 遠端指令的 [找不到遠端的處理方式](/git/concept-and-commands#remote-debug) 段落。

<br />

## 安全的強制推送

你以為我要講 force-with-lease 嗎，我要說的是更少人知道的 force-if-includes，請見[使用 Force if Includes 安全的強制推送](/git/force-if-includes)，裡面還有解釋 lease 到底在「租」什麼東西。

<br />

## 清除隱私資料

使用任意方式把目標提交從提交歷史中移除就可以了，不用擔心 reflog 紀錄，因為 reflog 紀錄壓根就不會被推送到遠端，如果要徹底清除本地紀錄可以使用 filter-repo，內建的 filter-branch 已經不被建議使用。

[賣課網又寫錯了](https://gitbook.tw/chapters/faq/remove-files-from-git)，想想每個人的 reflog 紀錄都不一樣，那怎麼可能被推送？作者書都寫完了結果還是不知道 Git 是「分散式」的「鏡像系統」，有搞清楚分散鏡像系統就不可能說出 reflog 紀錄被推送這句話。

你可能會覺得我很嚴格，可能作者就是剛好沒想到啊，你說的沒錯，那退一步來說，要寫書教別人之前至少要測試正確性吧，看起來是沒有。

<br />

## 只推送部分提交

```sh
git push <遠端名稱> <指定提交>:<遠端分支名稱>
```

又是賣課網，10 秒能講完的事情他拍了[七分鐘的影片](https://www.youtube.com/watch?v=VShhhq_5sMc)。

<br />

## 加速 Clone

請見我的文章[使用 Git Sparse Checkout 只下載部分專案以加速 Clone 速度](/git/reduce-size-with-sparse-checkout)。

其實 [The Will Will Web](https://blog.miniasp.com/post/2022/05/17/Down-size-your-Monorepo-with-Git-Sparse-checkouts) 就寫的很詳細，我覺得雖然詳細但不夠清楚，而且指令有部分更新，所以統整後寫成文章。
