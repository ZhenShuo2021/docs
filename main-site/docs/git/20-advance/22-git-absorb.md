---
title: Git Absorb 繁體中文唯一教學
sidebar_label: Git Absorb 教學
author: zsl0621
slug: /git-absorb
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-05-21T17:31:33+08:00
  author: zsl0621
first_publish:
  date: 2025-05-21T17:31:33+08:00
---

我們常常會完成 A/B/C 提交後，推送前檢查發現每個提交都有些 typo 要修正，這時候有兩種解決方法，一種是 `rebase -i` 進去 edit，另一種是提交一個新的版本後再使用 `rebase -i`，然後修改順序並且改為 `fixup`，兩者都要手動進行重複操作非常麻煩。

除了本地修改，在審核 PR 的時候，如果有多個提交也要這樣修改，為了保持提交的原子性 atomic，我們也不會想新增一個 `fix: xxxx` 修改三個提交，而是希望對三個提交各自進行修正，這時候就需要我們的 [git absorb](https://github.com/tummychow/git-absorb) 出動了。

## 原理

如果已經很熟悉 `git rebase --autosquash` 請跳過此段。

`git rebase --autosquash` 用途是把指定 prefix 的提交進行自動處理，例如

1. fixup!
2. squash!

這些提交就會自動被 fixup/squash 到對應的提交裡面，這樣就達到我們前言裡面說的各自修正了，然而此指令很瑣碎，內建的指令要這樣使用：

```sh
git add <file>

git commit --fixup=<commit-ish>

git rebase -i <要被fixup的那個提交>^ --autosquash

# 操作 Vim 編輯器
```

這有兩個大問題，第一，要找到 `--fixup=<commit-ish>` 這無疑是 Git 裡面最討厭的步驟，要嘛一個一個數然後 `~N`，要嘛開新的 git log 然後手動輸入 hash；第二，字太多，工程師就是很懶，所以這時候 git absorb 指令就可以解決這個問題。

## git absorb

[git absorb](https://github.com/tummychow/git-absorb) 會自動檢查應該被合併到哪個提交中，所以就不再需要前面繁瑣的步驟了，不用再去翻 log 翻到眼睛都花掉，指令會自動判斷應該被合併到哪個提交中。

git absorb 指令預設是保守的，避免錯誤的合併讓你要拆分反而造成更多麻煩，所以有時候會提示 warning 說找不到，這時可以使用手動使用前面的方式完成。

### 安裝

```sh
# Windows
winget install tummychow.git-absorb

# MacOS
brew install git-absorb

# Ubuntu
apt install git-absorb
```

### 自動 rebase

因為這個指令的目的就是要簡化操作所以使用非常簡單：

```sh
git add $FILES_YOU_FIXED
git absorb --and-rebase
```

如果 rebase 過程中想要退出，請使用 `:cq` 告訴 Git 這是錯誤執行，Git 會取消這次 rebase。

如果作者不同請加上 `--force-author`

```sh
git absorb --and-rebase --force-author
```

如果搜尋範圍不夠大可以指定 `--base`

```sh
git absorb --and-rebase --force-author --base <base-commit>
```

### rebase 前檢查

使用 --and-rebase 會自動完成 rebase，如果你覺得這樣有點不安心，想要事前確認他會被 absorb 到哪個 commit 中，可以使用此指令：

```sh
git add $FILES_YOU_FIXED
git absorb # 不加上 and-rebase
git log # 檢查
git rebase -i --autosquash HEAD~<2N+1>
```

這裡的 `N` 代表 fixup 的數量，rebase 範圍要兩倍才會找到要被合併的目標提交。如果歷史記錄更遠那就需要手動指定 commit。
