---
title: 使用 Git Worktree 簡化分支切換
sidebar_label: Git Worktree 多工處理
author: zsl0621
slug: /git-worktree
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-01-14T20:41:33+08:00
  author: zsl0621
first_publish:
  date: 2024-09-10T16:15:33+08:00
---

# {{ $frontmatter.title }}

Git worktree 的目的是簡化在分支之間切換的指令步驟，甚至允許你同時修改兩個分支。舉例來說，以往我們在多個分支間來回切換的指令流程是

1. git stash
2. git switch \<feat\>
3. do something
4. git switch \<original-branch\>
5. git stash pop

每切換一次就要浪費四步指令只是為了切換分支，於是我們可以使用 worktree 功能解決這個問題。git worktree 解決了以下問題：

1. 每次都要 stash 步驟太多
2. 工作時間拉長也記不起來 stash 了什麼
3. 如果是要比較兩個分支差異使用 checkout 做不到同時顯示

總之就是所有需要同時處理的問題都可以使用 git worktree 解決。

## 觀念說明

Git worktree 允許你**在新資料夾中 checkout 指定分支，該資料夾中的所有操作都由原 Git 專案追蹤，即使這個新資料夾不在原專案的目錄下**。

> 新資料夾不強迫放在專案資料夾以外，但是這樣你的專案底下就會出現新增的 worktree，會看到資料夾顯示 untracked。

## TL;DR

worktree 指令就是幫分支直接建立一個獨立資料夾，對該分支進行任何操作就到該資料夾操作動作完全一樣，所以要對他進行任何新增移除提交、rebase/push/pull 都隨你高興，只是清除要回到主專案使用 `git worktree remove` 而已。小提醒，我們無法切換到正在使用 worktree 的分支。

```sh
# 新增一個 worktree，會建立和資料夾名稱相同的分支
git worktree add ../folder-name

# 到 worktree 進行工作，進行提交或者對照程式碼等等...

# 工作完成，回到主儲存庫
# 列出 worktree
git worktree list

# 移除完成工作的 worktree
git worktree remove ../folder-name

# 對完成任務的分支進行合併/rebase/push等等...
```

### 指令介紹

```sh
git worktree add [(-b | -B) <new-branch>] <path> [<commit-ish>]
    [-f] [--detach] [--lock [--reason <string>]] [--orphan]

# 指令範例，一般只會用這三種，頂多加上 --detach
git worktree add ../projectA-quick-look
git worktree add ../projectA-quick-look feat
git worktree add ../projectA-quick-look -b quick-look feat
```

第一個範例指令會建立一個 `../projectA-quick-look` 資料夾並且自動建立和他同名的分支；第二個範例指令指定 worktree 起始位置且不會建立新分支；第三個則是指定分支名稱和基底，這三個就是 worktree add 最常使用到的參數了。

剩下幾個常用參數的解釋如下：

- `-B`: `-b` 不會覆寫分支，`-B` 會
- `-f`: 在想要建立的 worktree 消失時使用，可強制建立工作樹，請見[文檔](https://git-scm.com/docs/git-worktree#Documentation/git-worktree.txt--f)
- `-d`: 無頭模式，和無頭分支一樣就是隨手看看或者試功能的時候使用
- `--lock`: 避免工作樹被 git 意外清除
- `--orphan`: 新增一個完全空白的分支

其餘指令很直觀沒什麼好解釋：

- `list`: 列出
- `remove`: 移除
- `move`: 移動到新位置
- `prune`: 清理遺失的 worktree，可能是因為資料夾被 git 以外的東西移動
- `lock/unlock`: 鎖定避免被 git worktree remove/prune 移除，以及解鎖
