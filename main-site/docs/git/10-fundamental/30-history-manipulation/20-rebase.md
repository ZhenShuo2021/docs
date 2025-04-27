---
title: 使用變基 Rebase 合併分支
sidebar_label: 變基 Rebase 合併分支
author: zsl0621
slug: /rebase
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-02-13T23:59:25+08:00
  author: zsl0621
first_publish:
  date: 2024-09-07T14:10:12+08:00
---

# 使用變基 Rebase 合併分支提交

筆者認為初學者對變基 (rebase) 感到畏懼的原因不是指令複雜，反而是網路文章每篇講的都不一樣，所以撰寫時同時參考 Git 官方文檔以及 Pro Git Book，多方比對和驗證保證本文解釋方式能和指令實際用途能夠對應，講這麼多目的就是要讓你只要讀這篇文章就夠了，不需要再去網路上查其他文章，因為網路上的文章參差不齊容易被誤導。

:::danger 提醒

Rebase 很危險，請用範例 repo 進行測試！

- [單分支範例](https://github.com/PIC16B/git-practice)  
- [仿照文檔的多分支範例方便對照測試](https://github.com/ZhenShuo2021/rebase-onto-playground)  
:::

## 什麼是變基 Rebase

介紹 rebase 前我們要了解 merge。merge 是把兩個分支合併成一個分支，以下方原始狀態為例，使用 git merge 會產生新的提交 F 紀錄合併，並且保留分支結構：

```sh
# 原始狀態

main     A---B---C---D---E
              \         
feature        A1--B1--C1 
```

```sh
# 使用 git merge，指令如下：
# git checkout main     # 切換分支到 main
# git merge feature     # 合併 feature 分支

main     A---B---C---D---E---F
              \             /
feature        A1---B1----C1 
```

但如果我們想保持提交樹的乾淨，這時候我們就可以使用 `git rebase` 完成。在原始狀態使用 git rebase 效果如下：

```sh
# 使用 git rebase，指令如下：
# git checkout feature  # 切換分支到 feature
# git rebase main       # 目標分支是 main

main     A---B---C---D---E
                           \
feature                    A1'--B1'--C1'
```

可以看到目前分支 (feature) 被接在目標分支 (main) 的後面而且沒有用於紀錄合併的提交。細心的人可能會發現多了 prime symbol `'`，這裡我們詳細解釋 `git rebase main` 實際上做了什麼：

1. 找到共同祖先 (B)
2. 找到需要被變基的提交並且暫存他們  
  這些提交包含<u>**從共同祖先到「目前分支」之間的所有提交，並且排除「目標分支」已經存在的提交**</u> (此範例沒有被剃除的提交，暫存 A1 B1 C1)
3. 將目標分支最後一個提交作為出發點，把暫存的提交逐個重演到目標分支後面 (接上後成為 A1' B1' C1')

所以總共只有三步驟，找到共同祖先，以祖先為起點開始尋找目標提交，重演這些提交。

:::tip 重演 (replay)
不只是簡單的將提交複製貼上，而是會重新生成 commit hash。
:::

## 口訣

re-base 的核心兩個單字分別代表 `重新` 以及 `基底`，表示此指令用於幫提交修改基底，我們可以用口訣記憶指令步驟

:::tip 口訣

<center>**比較「目前分支」和「目標分支」**</center>
<center>**把「目前分支」的提交移動到「目標分支」之後**</center>

:::

請務必記住口訣，這可以讓你正常使用好一段時間，筆者保證這段敘述的絕對正確，所有違背這段敘述的說明都是錯的。

> 口訣有稍微簡化，完整版本請見[搞懂 Rebase Onto](/git/rebase-onto)。

## 誰 Rebase 誰才對？{#correctly-rebase}

很多文章都錯誤使用 rebase，小到個人 medium 和部落格，大到系列教學文章，甚至是已經出書的人都寫錯，不就是只有一個參數的指令那究竟是哪裡用錯呢？他們<u>**錯在位於主分支使用 `git rebase feature`**</u>，這會造成主分支的提交歷史改變，而主分支是最穩定的分支，<u>**絕對不可能為了合併子分支而修改主分支既存的提交歷史**</u>。

正確的使用方式應該是先移動到要被合併的分支，再使用 `git rebase main`：

```sh
# 這四個指令代表把 feature-1 和 feature-2 在 main 之後進行重演
git switch feature-1
git rebase main
git switch feature-2
git rebase main

# 或者使用兩個參數，第二個參數代表預先 switch 到該分支，效果和上面的指令完全相同
git rebase main feature-1
git rebase main feature-2
```

如果你不相信筆者，那麼請看 [官方文檔](https://git-scm.com/docs/git-rebase/zh_HANS-CN) 或是教學書 [Pro Git](https://iissnan.com/progit/html/zh-tw/ch3_6.html)，如果還是很疑惑，那麼 Python Core Dev，微軟工程師拍的影片[十分钟学会正确的github工作流，和开源作者们使用同一套流程](https://www.youtube.com/watch?v=uj8hjLyEBmU&t=439s&pp=ygUM56K86L6y6auY5aSp)也是這樣用。

:::danger 再提醒一次

絕對不要在主分支上 rebase 其餘分支，因為這會修改穩定的主分支提交紀錄。

:::

## rebase 自動暫存 autostash

rebase 時要求目錄乾淨不能有尚未提交的變動，如果沒有清理乾淨就會出現

```sh
error: cannot rebase: You have unstaged changes.
error: Please commit or stash them.
```

可以加上 `--autostash` 讓他自動 stash，rebase 結束後會自動 stash pop，或者修改全局設定永遠自動 stash

```sh
git config --global rebase.autoStash true
```

## rebase 自動壓縮 autosquash

簡單來說就是自動把 "fixup!" 開頭的 commit 合併，我認為比起一般的操作速度沒有差，也就是隨便打提交訊息再使用互動式提交，所以不用記這個。

## git rebase --onto

過於複雜，他可以直接寫成一篇文章，請看 [搞懂 Rebase Onto](/git/rebase-onto)。

## 翻譯：變基和衍合

rebase 中文有變基和衍合兩種翻譯，衍代表散佈、滋生，我看不出來 rebase 從單字、處理方式到用途哪裡跟衍有關係，所以我投變基一票。

## 參考

- [官方英文文檔](https://git-scm.com/docs/git-rebase)
- [官方簡中文檔](https://git-scm.com/docs/git-rebase/zh_HANS-CN)
- [官方簡中 Pro Git](https://git-scm.com/book/zh/v2/Git-%e5%88%86%e6%94%af-%e5%8f%98%e5%9f%ba)
- [官方英文 Pro Git](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)
- [繁體中文 Pro Git](https://iissnan.com/progit/html/zh-tw/ch3_6.html)
