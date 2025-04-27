---
title: Git 各種修改提交歷史的情境和解法
sidebar_label: 各種修改提交歷史的情境
author: zsl0621
description: 各種修改 commit 的情況和對應的解決方式。
slug: /edit-commits
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2024-09-07T14:10:12+08:00
  author: zsl0621
first_publish:
  date: 2024-09-07T14:10:12+08:00
---

# {{ $frontmatter.title }}

本文以情境為主軸介紹，每個段落都是一個獨立的情境，遇到問題查這篇文章基本上都能解決，不用再去網路上找半天。

以下是預備知識

- **大部分都用互動式變基完成，操作方式請見[使用互動式變基任意修改提交歷史](./interactive-rebase)**。
- `hash^` 的 `^` 代表該 hash 的前一個提交，`~n` 代表前 n 個提交。
- `--amend` 可以加上 `--no-edit` 表示不修改 commit 訊息。
- rebase 時 git 會自動幫你 checkout，這時候查看 `git status` 會顯示「互動式重定基底動作正在進行中」，使用 `git branch` 查看則會顯示目前分支為「無分支，重定 main 的基底」 (no branch, rebasing main)。
- 遇到合併衝突請看[如何解決合併衝突](/basic/keyword#進階)，實際操作過才會清楚。

## 修改提交訊息{#edit-commit-message}

1. 修改前一個提交：`git commit --amend`

2. 修改更早的提交：
   - `git rebase -i hash^`
   - 把想修改 message 的 commit 前面的 `pick` 改成 `r`
   - 跳出 commit message 視窗，直接修改
   - 附帶一提 Vim 編輯介面中的提交紀錄，上到下順序是從舊到新

## 修改前一個提交內容{#amend-previous-commit}

完成一個 feature 之後很開心的 commit，伸個懶腰馬上發現有 typo 要怎麼修改呢？

1. 修正 typo
2. `git add .`
3. `git commit --amend`

或者，放棄前一個 commit：`git reset --soft HEAD^`

話說這天天在發生...

## 修改更早的提交內容{#edit-earlier-commit}

發現舊 commit 有地方沒修好，要怎麼單獨修改那個 commit？

1. `git rebase -i hash^`
2. 該 hash 前面改成 edit 或者縮寫 e
3. 修改文件後加入追蹤 `git add <file name>`，注意不需提交
4. 完成 rebase `git rebase --continue`

::: info
互動式變基過程中應該使用 continue，如果使用 `git commit` 會變成插入一個新的提交。
:::

## 合併提交{#squash-commits}

覺得提交太瑣碎了想要合併：

1. `git rebase -i hash^`
2. 想要 **被合併** 的 commit `pick` 改 `s`，他會合併到前一個 commit
3. 完成 rebase `git rebase --continue`

## 插入提交{#insert-commit}

1. 互動式變基指定提交 `git rebase -i <hash>^`
2. 把該 hash 的 `pick` 改為 `edit` 或 `e`，儲存並且退出 vim 編輯器
3. 想怎麼改就怎麼改，甚至可以在這裡使用 `git cherry-pick`
4. 修改結束後使用 `git rebase --continue`

需要注意的是 rebase 比較危險，如果還不熟悉建議先在原本位置建立一個備份分支。如果 rebase 過程中想放棄就使用 `git rebase --abort`

## 拆分提交{#split-commit}

1. 同[插入提交](#insert-commit)的 1, 2
2. 清除最新一個提交 `git reset HEAD^`
3. 把檔案依序 `git add` `git commit`
4. 拆分結束後使用 `git rebase --continue`

## 修改提交的順序{#reorder-commits}

完成多項功能開發，想要整理，例如把 fix 和 feat 開頭的分類排放：

1. `git rebase -i hash^`
2. 直接調整提交順序並儲存
3. 完成 rebase `git rebase --continue`

## 刪除提交{#delete-commit}

1. `git rebase -i hash^`
2. 把該 commit `pick` 改成 `drop` 或整行刪掉
3. 完成 rebase `git rebase --continue`

## 不影響當前分支，修改一個特定的提交{#edit-specific-commit}

有兩種方式，分別對應小型和大型修改。

小型修改直接使用 `git rebase -i`，改為 edit 模式即可。

大型修改使用 `git checkout -b <new-branch-name> <hash>` 在指定 commit 建立新分支，直接修改後看你要怎麼處理這個修正。

## 結語{#conclusion}

每個情境了不起就五句話，某賣課網站可以把每個情境都寫成一篇文章還被 Google SEO 排到很前面，佩服佩服= =
