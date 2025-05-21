---
title: Git Rebase Update-Ref 繁體中文唯一教學
sidebar_label: Rebase Update-Ref 詳解
author: zsl0621
slug: /rebase-update-ref
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-05-21T17:15:33+08:00
  author: zsl0621
first_publish:
  date: 2025-05-21T17:15:33+08:00
---

本文是繁體中文唯一一篇 git rebase --update-ref 教學，內容來自於 [Working with stacked branches in Git is easier with --update-refs](https://andrewlock.net/working-with-stacked-branches-in-git-is-easier-with-update-refs/)。

## 用途

文檔是這樣寫

> Automatically force-update any branches that point to commits that are being rebased.

有看沒有懂，有說明但不多，這裡幫他翻譯：**自動將 rebase 路徑中的其他分支也一起 rebase**，這樣的說明簡潔又精確。

## 範例

請使用 [rebase-onto-playground](https://github.com/ZhenShuo2021/rebase-onto-playground) 測試範例。

### 1. 主分支(dev)有新提交，需要更新整個堆疊

> 請注意所有範例都基於「開發者喜歡把 PR 分成多個子 PR 方便審核」為前提，所以才會有這麼多分支。

比如說我們在子分支開發時，遇到主分支更新了，這時候我們就必須 rebase 把 `D` 接到最新的 `I`，還沒修改前的提交歷史如下：

```sh
A --- B --- C --- I  (dev)
       \
        D --- E  (andrew/feature-xyz/part-1)
              \
               F --- G  (andrew/feature-xyz/part-2)
                     \
                      H  (andrew/feature-xyz/part-3)
```

不使用 --update-refs 重定基礎(第一步結果)

```sh
A --- B --- C --- I  (dev)
                  \
                   D' --- E'  (andrew/feature-xyz/part-1)
```

但是這樣只完成了一個分支，而part-2 和 part-3 仍舊指向舊位置

```sh
        D --- E  
              \
               F --- G  (andrew/feature-xyz/part-2)
                     \
                      H  (andrew/feature-xyz/part-3)
```

使用 --update-refs 則可以自動 rebase 路徑上的其他分支，一步完成所有分支的操作

```sh
# git checkout andrew/feature-xyz/part-3
# git rebase dev --update-refs

A --- B --- C --- I  (dev)
                  \
                   D' --- E'  (andrew/feature-xyz/part-1)
                          \
                           F' --- G'  (andrew/feature-xyz/part-2)
                                   \
                                    H'  (andrew/feature-xyz/part-3)

```

### 2. PR審核後，part-1分支有新變更，需要更新後續分支

> 請注意所有範例都基於「開發者喜歡把 PR 分成多個子 PR 方便審核」為前提，所以才會有這麼多分支。

這個情境是 part-1 在 PR 審核後需要修改，所以 part-2 part3 都需要對應更新

```sh
A --- B --- C  (dev)
       \
        D --- E --- J  (andrew/feature-xyz/part-1) # J是新的PR反饋提交
              \
               F --- G  (andrew/feature-xyz/part-2)
                     \
                      H  (andrew/feature-xyz/part-3)
```

不使用 --update-refs

```
A --- B --- C  (dev)
       \
        D --- E --- J  (andrew/feature-xyz/part-1)
                    \
                     F' --- G'  (andrew/feature-xyz/part-2)
```

而 part-3 仍指向舊位置

```sh
              \
               F --- G
                     \
                      H  (andrew/feature-xyz/part-3)
```

使用 --update-refs 可以一步完成

```sh
# git checkout andrew/feature-xyz/part-3
# git rebase andrew/feature-xyz/part-1 --update-refs

A --- B --- C  (dev)
       \
        D --- E --- J  (andrew/feature-xyz/part-1)
                    \
                     F' --- G'  (andrew/feature-xyz/part-2)
                              \
                               H'  (andrew/feature-xyz/part-3)
```

## 總結

簡單來說就是自動修改所有路徑中分支，在原文是寫堆疊 (stack)。
