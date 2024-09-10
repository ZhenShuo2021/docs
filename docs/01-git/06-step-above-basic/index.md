---
title: 常用指令
author: zsl0621
description: 最快速上手 Git 的文章沒有之一。
tags:
  - Git
  - Programming
keywords:
  - Git
  - Programming
last_update:
  date: 2024-09-10T16:15:33+08:00
  author: zsl0621
---

# Git 常用指令

一分鐘入門只學會一股腦記錄版本提交，後面才是 Git 真正的精華。接下來介紹日常常用的指令，會先根據是否操作分支分成基礎操作（單一分支）和進階操作（多分支），每個會先列出該操作的常用命令，介紹每個命令時會說明他使用的場景，再說明該命令的選項參數。



## 基礎操作
基礎操作包含了對於單一分支的各項操作，包含更靈活的 add/commit 檔案，以及檔案復原。
```sh
git reset                   # 修改檔案狀態
git checkout                # 同時處理檔案管理和切換分支
git restore                 # 檔案管理
```


### 新增檔案 git add
`git add` 除了一一新增或者新增全部以外，其實還有補丁模式可以更方便使用：
- `-p`, 補丁模式，互動式加入預存，常用選項為
  - y, yes
  - n, no
  - d, 該檔案之後都不要加入
  - s, 切成更小的區塊 (hunk)

或者是 `git add .` 預存全部檔案後，使用 `git reset *.py` 移除當前目錄下所有 `.py` 檔案，`**/*.py` 移除所有目錄的 `.py` 檔案。


### 放棄修改 git restore
restore 命令用於檔案管理，這裡的放棄指的是尚未提交的文件。git restore 參數為：

```sh
git restore [<options>] [--source=<tree>] [--staged] [--worktree]
```

- --source, -s  
  要恢復到的版本，例如 git restore --source=HEAD~3 恢復到三個版本前
- --staged, -S  
  把已預存 staged 檔案踢出，等於取消 add
- --worktree, -W  
  還原到目錄樹（白話文：還原沒有預存的程式碼，讓他從工作目錄中移除），這個參數預設開啟，使用 -S 時會關閉要手動打開。

#### 放棄未預存的程式碼
```sh
git restore <file>
```
等同於 git restore -W

#### 放棄程式碼，不管是否預存
```sh
git restore -S -W <file>
```
  
<details>
  <summary>註記：為什麼用新版指令 git restore？</summary>

新手學習這個指令是最好的，因為舊版本檔案管理指令混雜，例如 `git reset <file>` 可以指定檔案踢出預存，但是 `git reset --hard HEAD` 強制還原到 HEAD 又不能指定檔案，`git checkout -- .` 可以還原 unstaged 檔案，卻又不能處理 staged 檔案。

[參考資料](https://dwye.dev/post/git-checkout-switch-restore/)
</details>


### 清除提交 git reset
用於控制提交版本，reset 雖然聽起來是重設/還原，但實際做的事情是**清除**提交，分為三種：

```sh
# 只刪 commit，其他不動
git reset --soft <hash>

# 刪 commit 和 add
git reset --mixed <hash>

# 除了 commit 和 add 以外，連你的寫的程式都刪了，謹慎使用！
git reset --hard <hash>
```
接下來提供幾個使用情境方便記憶。

#### 不小心提交，想繼續編輯
```sh
git reset --soft HEAD^
```

這個指令會取消最新的提交，但保留所有程式碼修改（保留在預存區 staging area）。另外兩個選項主要分成兩種情況，mixed 用於選擇踢掉預存用於想要額外選擇預存檔案，而 hard 非常危險，commit, stage, 工作目錄**全部刪除**，需小心使用。

#### 提交了多個小變更，想整理成一個提交
```sh
git reset HEAD~3
```

這會移除包含現在的三個提交。

#### 只還原指定檔案到前一個提交
```sh
git reset HEAD~1 -- <file-name>
```

:::info 

1. HEAD 代表目前工作的 commit 位置
2. "^" 代表前一個提交，"~n" 代表前 n 個提交
3. -- 代表檔案分界線

:::


:::danger

雖然初學暫時不會碰到多人合作，但還是必須強調版本修改 **永遠只該用於個人分支**！

:::
到這邊就結束單一分支的基本操作了，接下來是多分支的操作。


## 進階操作
進階操作包含跨分支的操作，本文中只要實質上是操作分支的都放在這個類別。
```sh
git branch                  # 分支操作
git switch                  # 切換分支
git stash                   # 暫存檔案（非預存）
git rebase                  # 修改提交歷史
```


### 分支 git branch
當你工作變複雜一條分支不夠用就會用到這些，用於功能開發、問題修復、或者是發佈用的分支。
```sh
git branch                           # 查看
git branch <name>                    # 新建
git checkout <name>                  # 切換
git branch -D <name>                 # 刪除
git branch -m <old> <new>            # 改名
git merge "NAME"                     # 合併
```


### 暫存 git stash
這是一個特別的指令，會把所有檔案都放進獨立的 stash 中，再把工作目錄還原成上一次提交的版本。  
使用情境：
1. 使用 git rebase 時強制目錄不能有未存檔檔案
2. 改到一半需要改一個更重要的東西
3. 改到一半需要跳到別的分支

可以看出來他是一個暫時擋刀用的指令。

基本選項：
```sh
git stash                  # 暫存變更
git stash list             # 查看所有的 stash
git stash apply stash@{0}  # 恢復第一個暫存的變更
git stash drop stash@{0}   # 刪除 applied stash
git stash pop              # 等同 apply + drop
git stash clear            # 清除所有 stash
```

### 修改 git rebase
這是一個功能非常強大的指令，甚至有教學說[「不會 rebase, 等於沒學過 Git」](https://myapollo.com.tw/blog/git-tutorial-rebase/)。這裡只講解他的基本邏輯，引用自[码农高天](https://www.youtube.com/watch?v=uj8hjLyEBmU)，是我看過講的最好，最清楚也最簡短的說明：

<center>**將「目前分支」移到旁邊，把「目標分支」拿過來，再把移到旁邊的「目前分支」想辦法接上去**</center>
<br />
為了搞懂 rebase 看了很多文章，直到看到這句話才搞懂，真的不需要了解工具怎麼實現的，只要會用工具就好了。用都不會用就講原理的結果就是不會用也不懂原理，後面會有單獨的文章介紹 rebase。


:::danger

雖然初學暫時不會碰到多人合作，但還是必須強調版本修改 **永遠只該用於個人分支**！

:::

## 結語
到這邊你已經可以基本流暢的操作 git 了，需要修改過去提交再看下兩篇文章，不然可以快轉到[遠端儲存庫設定](/docs/git/remote-setup)。