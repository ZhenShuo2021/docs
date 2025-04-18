---
title: Git 分支操作
slug: /branch
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-01-16T14:38:33+08:00
  author: zsl0621
first_publish:
  date: 2024-09-10T16:15:33+08:00
---

# {{ $frontmatter.title }}

分支操作主要有以下五大指令，扣掉複雜的 rebase 以外其餘根本不需要每個寫成一篇文章來介紹。本文只會有本地不會有遠端操作，遠端操作的第一篇文章要從[遠端儲存庫設定](/intermediate/remote-setup)開始。

```sh
git branch                           # 分支操作
git switch                           # 切換分支
git stash                            # 暫存檔案（非預存）
git rebase                           # 任意修改提交歷史
git revert                           # 恢復提交
```

## 分支 git branch

分支用於開發新功能、問題修復、或者是準備發佈新版本。

```sh
git branch                           # 查看分支清單
git branch <name>                    # 新建分支
git switch <name>                    # 切換到分支，等效於為 git checkout <name>

git branch -D <name>                 # 刪除
git branch -m <old> <new>            # 改名
```

## 切換 git switch

`git switch` 是專門用來切換分支的新版指令，比傳統的 `git checkout` 更明確，因為 `git checkout` 太過萬能。

```sh
git switch branch_name               # 切換到現有分支
git switch -d commit_hash            # 分離模式，用於臨時查看、不建立分支的切換，d=detach
git switch -c <new-name>             # 建立新分支並切換
git switch -c <new-name> <hash>      # 從指定的提交建立並切換
```

使用 checkout 來切換分支也完全沒問題，沒有任何差別，指令也大同小異不重複介紹。

## 暫存 git stash

這是一個特別的指令，會把目前所有尚未提交的檔案放進獨立的暫存空間中，使用情境主要有以下幾個：

1. 使用 git checkout/switch/rebase 時目錄不能有未提交檔案，可以用他暫存
2. 改到一半需要改一個更重要的東西
3. 改到一半需要跳到別的分支

可以看出都是暫時，是擋刀用的指令。

基本選項：

```sh
# 基本上只要記得前三個用法
git stash                            # 暫存變更
git stash -- <pathspec>              # 暫存指定檔案
git stash pop                        # 還原暫存並且移除紀錄，等同 apply + drop
git stash list                       # 查看暫存清單

# 比較少用的方式
git stash push -m "my_stash_name"    # 幫暫存取名（只是讓你查看名稱，不支援從名稱恢復）
git stash apply stash@{0}            # 恢復第一個暫存
git stash drop stash@{0}             # 刪除第一個暫存
git stash clear                      # 清除所有暫存
```

> 看不懂 pathspec？請見[看懂文檔](../beginner/read-git-docs)。

這裡是更少用的參數，但是偶爾有奇效

- git stash -u 可以把還沒被追蹤的文件也暫存進去
- git stash --staged 只暫存已經被 stage 的檔案

<br/>

## 合併 git merge

分久必合，有了分支下一步就是合併，`git merge` 是最淺顯易懂的合併方式。使用時我們一般來說都會先進入主分支以合併子分支，指令如下：

```sh
git switch main
git merge feature-branch
```

這樣就會把 `feature-branch` 的修改全部加在 `main` 分支中，並且建立一個新的提交用於記錄合併。`git merge` 預設模式是 fast-forward，意思是兩者的分支結構完全相同時，只會將目前分支的 HEAD 向前移動，不會建立新的用於記錄合併的提交。我們可以使用 `--no-ff` 關閉此模式，下方是兩者的差異示意圖：

```sh
# 初始狀態
  A---B---C  main
           \
            D---E---F  feature-branch
```

使用 Fast-Forward 的結果如下：

```sh
# 使用 Fast-Forward，直接把 main 從 C 移動到 F
# git merge feature-branch
  A---B---C---D---E---F  main and feature-branch
```

不使用 Fast-Forward 的則會新增一個專門用來紀錄合併的提交 G：

```sh
# 使用 --no-ff，建立記錄合併的提交 G
# git merge --no-ff feature-branch
  A---B---C-----------G  main
           \         /
            D---E---F  feature-branch
```

兩者的選擇端看是否需要保存分支結構。合併時可能會遇到衝突，這需要手動解決，解決方式請看 [如何解決合併衝突](../beginner/keyword#進階)。

<br/>

## 變基 git rebase

先解釋名詞變基，意思是「變換」目前分支的「基底」，用於取代 `git merge`，目的是簡化提交歷史，避免到處都是合併的結構導致閱讀和管理困難。由於比較複雜只講解他的基本邏輯：

::: tip 口訣

<!--@include: @/snippets/rebase-formula.md-->

:::

為了搞懂 rebase 看了很多文章，最後濃縮成這兩句話[^compress]，使用教學請見本教學的 [使用變基 Rebase 合併分支](../intermediate/rebase)。

[^compress]: 附帶一提，濃縮後還是正確的，網路上很多文章講的又臭又長結果還是講錯真的是來搞笑的，而且 git rebase 如果使用方式正確就不會修改到主分支的提交歷史和 hash，如果改變就代表用錯了，網路上就有滿山滿谷的錯誤教學，這些人不知道是有沒有發現自己錯，總之是沒幾個人回去修正自己的文章。

> [如何解決合併衝突？](../beginner/keyword#進階)

## 任意修改提交歷史 git rebase -i

互動式變基 (interactive rebase) 使用變基的原理實現對提交歷史進行任意修改，同時**使用方式非常簡單**，請見[使用互動式變基任意修改提交歷史](../intermediate/interactive-rebase)。

::: danger

git rebase -i 會修改歷史，再次強調修改提交歷史 **永遠只該用於個人分支**！

:::

變基時可能會遇到衝突，解決方式請見[如何解決合併衝突](../beginner/keyword#進階)。

## 恢復 git revert

用實際案例講解比較簡單。假設現在想撤銷提交 A，但是由於團隊合作最好別修改提交歷史，我們可以用 git revert 提交一個 negative A，這樣會產生一個新的提交把提交 A 抵銷，也不用修改歷史。

放在這裡的原因是團隊合作才會用到，一個人的話想怎麼改就怎麼改。

## 結語

到這邊你已經熟悉 Git 的所有常用指令了，只是需要一點時間熟悉，下一步請快轉到[遠端儲存庫設定](/intermediate/remote-setup)，進階文章初學者用不到建議暫時跳過以免越看越亂。
