---
title: Git 各種日常問題集合 - 本地
sidebar_label: 日常問題 - 本地
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

# {{ $frontmatter.title }}

都是簡單的日常問題但是要花一點時間搜尋，所以這篇文章集中列出方便查詢。

[[toc]]

## 正確 rebase

正確使用方式是移動到子分支後再使用 `git rebase main`，或者直接使用 `git rebase main <sub-branch>` 才對，原因請見[使用變基 Rebase 合併分支](/intermediate/rebase)。

[為你自己學 Git](https://gitbook.tw/chapters/branch/merge-with-rebase) 和 [Git 版本控制教學 - 用範例學 rebase](https://myapollo.com.tw/blog/git-tutorial-rebase/) 都寫錯了，這最誇張，但凡看過一次文檔都不可能寫成 `git rebase <子分支>`，能錯的這麼離譜=看不懂文檔=沒看。

<br />

## rebase onto 指定新基底

此用法相對來說比較複雜，但是複雜的原因來自於網路上的錯誤教學，請見[搞懂 Rebase Onto](../advance/rebase-onto)。

沒有信口雌黃，撰文時唯一能找到的正確文章是在搜尋結果第五頁 [Git合并那些事——神奇的Rebase](https://morningspace.github.io/tech/git-merge-stories-6/)，如果不是因為要寫「正確的」教學筆者才沒耐心每篇都點進去看，還要在一堆錯誤裡面判斷對錯。

> 謎之音：正確還有必要強調喔，不是阿，網路上就一大堆「錯誤的」教學。

<br />

## blob, tree, tag, commit, refs 是什麼？

refs 只是幫助人類記憶的名稱，只紀錄提交 hash 讓你直接用 refs 等於指定該提交。

其他四個是 Git 的基本構成，請見[關鍵字、符號和基本組成](/beginner/keyword)。

<br />

## HEAD 是什麼

[賣課網又錯了](https://gitbook.tw/chapters/using-git/what-is-head)，HEAD 代表目前檢出 (checkout) 的位置，不只是分支，真的要解釋的話他屬於文檔定義中的 commit-ish，commit-ish 代表所有能最終指向一個 commit 物件的標識符，例如 HEAD, tag, branchname, refs...。

<br />

## 為何要用 git mv

`git mv` 和一般的 `mv` 差異是可以讓 Git 直接索引檔案，需要這個指令的原因是 Git 會推測你要作什麼，但是操作複雜時他就猜不出來你正在重新命名，`git mv` 就是告訴 Git「我正在重新命名這個檔案」。

有三種情況會用到

1. 操作複雜時，避免 Git 視為兩個不同的檔案，例如大規模變更檔案名稱
2. 在不區分大小寫的檔案系統上更改檔案名稱的大小寫
3. 移動 submodule 時

賣課網寫了[這麼長一篇文章](https://gitbook.tw/chapters/using-git/rename-and-delete-file)整篇都在說用途是讓我們少打一個指令，別搞笑了大哥。

<br />

## 移除已經提交的檔案但不刪除

```sh
git rm --cached
```

<br />

## 清除 reflog 紀錄

```sh
git reflog expire --expire=now --all
```

<br />

## 清理 .git 垃圾（無用的 commit 紀錄）

修改或是移除 commit 時，原有 commit 不會直接被刪除而是會暫存，這就是為何可以使用 reflog 還原的原因。對於這些紀錄 git 有自動清理機制，但是也可以手動清除：

```sh
git gc --aggressive --prune=now
```
