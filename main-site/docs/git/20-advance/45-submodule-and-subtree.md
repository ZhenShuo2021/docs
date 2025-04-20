---
title: Git 子模組和子樹 Submodule vs Subtree
slug: /submodule-and-subtree
sidebar_label: 多儲存庫管理
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-04-20T19:54:07+08:00
  author: zsl0621
first_publish:
  date: 2025-04-20T19:54:07+08:00
---

巢狀儲存庫其實指的就是上面的 multi-repo，拆成多個儲存庫管理，而拆分的方式有兩種

1. submodule: 主儲存庫只追蹤子儲存庫的 commit hash，要更新子儲存庫比較麻煩，要移動到子庫、提交、回到主庫更新。
2. subtree: 直接把子儲存庫的副本，包含提交歷史一起放進主儲存庫追蹤，因此幾乎等同於 Monorepo。

大部分狀況下如果都拆分了應該選 submodule，不然為什麼要拆呢，使用情景可以看 [When to use git subtree?](https://stackoverflow.com/questions/32407634/when-to-use-git-subtree)，寫了六個優點翻譯過來就是「除了指令少之外沒有優點」，至於缺點的部分，第一個問題是管理耦合，第二個問題是使用狀態不明確（使用 submodule 的流程會非常明顯的告訴你正在操作不同的儲存庫），簡單表格比較如下：

| 功能 | Submodule | Subtree |
|------|-----------|---------|
| 分離程式碼 | ✅ 完全分離 | ❌ 部分混合 |
| 更新流程 | ❌ 較複雜 | ✅ 較簡單 |
| 協作友好度 | ❌ 需額外步驟 | ✅ 無需額外步驟 |
| 儲存庫大小 | ✅ 較小 | ❌ 包含完整歷史 |
| 操作明確性 | ✅ 清楚區分 | ❌ 邊界模糊 |

所以 subtree 唯一的使用情境只在需要獨立修改子儲存庫的情況下才應該使用，如果沒有這兩個需求一律建議 submodule

- 需要頻繁修改第三方程式庫
- 長期專案需保證所有程式碼的可用性

## Submodule 指令

再次強調，主儲存庫只會追蹤 submodule 的 hash，他根本不管裡面改了什麼東西，所以每次子庫修改完都要回到主庫更新。

### 新增子模組

```bash
git submodule add <repository-url> <path>
```

### 初始化與更新

當你複製一個包含子模組的專案時，需要額外步驟:

```bash
# 複製主專案
git clone <url>

# 初始化子模組
git submodule init

# 更新子模組，拉取內容
git submodule update

# 一次性完成上述步驟
git clone --recurse-submodules <url>
```

### 更新子模組

```bash
# 更新所有子模組到其遠端的最新版本
git submodule update --remote

# 更新特定子模組
git submodule update --remote <path>
```

### 修改子模組

```bash
# 進入子模組
cd <submodule-path>

# 進行修改後提交
git add .
git commit -m "更新子模組"
git push

# 回到主專案，提交子模組的變更
cd ..
git add <submodule-path>
git commit -m "更新子模組的引用"
git push
```

如果要移動子模組[直接使用 git mv 即可](https://stackoverflow.com/questions/4604486/how-do-i-move-an-existing-git-submodule-within-a-git-repository)。

### 移除子模組

[How effectively delete a git submodule.](https://gist.github.com/myusuf3/7f645819ded92bda6677?permalink_comment_id=4447152)，簡單來說就是

1. 修改 .gitmodules 對應內容
2. 修改 .git/config 對應內容
3. `git rm --cached path_to_submodule` 沒有後置斜線
4. `rm -rf .git/modules/path_to_submodule` 沒有後置斜線
5. 提交變更

## Subtree 指令

還沒用過所以不亂寫。
