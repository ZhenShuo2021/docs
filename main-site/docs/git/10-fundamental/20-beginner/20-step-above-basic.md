---
title: 基礎操作
slug: /step-above-basic
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2024-09-10T16:15:33+08:00
  author: zsl0621
first_publish:
  date: 2024-09-10T16:15:33+08:00
---

# Git 基礎操作

一分鐘入門只說明如何提交檔案，但是在版本之間切換才是版本管理工具的精華，接下來的本文和[下一篇文章](./branch)分別為基礎操作（單一分支）和分支操作（多分支），快速檢索的方式是從右側目錄快速定位操作內容，例如提交錯誤時可以找到「還原提交」定位到 git reset。

基礎操作包含以下指令，文章中會介紹如何更靈活的 add/commit 檔案，以及復原檔案。

```sh
git add                              # 預存檔案
git commit                           # 提交檔案
git reset                            # 還原提交狀態
git checkout                         # 檔案還原的舊版指令，本文不介紹
git restore                          # 檔案還原的新版指令
git reflog                           # git 操作救命稻草
```

---

## 新增檔案 git add

`git add` 使用 `<pathspec>` 來指定檔案，可以輕鬆的區分檔案，例如

```sh
git add *.py                         # 追蹤目前資料夾第一層的所有 py 檔
git add **/*.py                      # 追蹤所有 py 檔
git add src/**/*.py                  # 追蹤 src 資料夾中的所有 py 檔
```

除了此之外兩個方便的指令：

- `-u`: 只預存已追蹤的檔案。
- `-p`: 補丁模式，互動式加入預存，常用選項為
  - y, yes
  - n, no
  - d, 該檔案之後都不要加入
  - s, 切成更小的區塊 (hunk)
  - 不太需要用補丁模式，改用 [lazygit](https://github.com/jesseduffield/lazygit) 完成相同功能方便一百倍

> 看不懂 pathspec？請見[看懂文檔](/git/read-git-docs)。

### 略過特定副檔名的檔案

```sh
# 方法一：使用表達式  
git add . ':!*.txt' ':!*.py'

# 方法二：使用 git reset
git add .                            # 預存全部檔案
git reset '*.txt' '*.py'
```

---

## 提交檔案 git commit

- `git commit -am <message>`: 懶人指令，略過 `git add .`。
- `git commit -m "<Title><按兩下Enter>`: 提交有標題的 commit message 的方式，打好標題後按兩次 enter，到第三行繼續寫內容。

### 修改上一次提交

常常會提交完成後馬上發現訊息寫錯、程式 typo，要怎麼修改？

直接修改現在的檔案內容並且使用下列指令可以完成，amend 的意思是修復（複寫）最近一次的提交

- `git commit --amend`: 修改上次的提交訊息和檔案。
- `git commit --amend --no-edit`: 修改上次的提交檔案，訊息不變。

---

## 還原檔案 git restore

```sh
git restore [<options>] <pathspec>
```

用於檔案還原，只會回復檔案內容不會修改提交歷史，常用參數有三個，分別是

- `-s, --source`: 指定恢復的提交來源。
- `-S, --staged`: 還原已預存的檔案（白話文：取消 add 或稱作 unstage）。
- `-W --worktree`: 還原到工作目錄（白話文：還原檔案在工作目錄（硬碟）的狀態）。預設開啟，但是使用 -S 時會關閉，原因是讓你可以只取消預存而不是連檔案修改都還原了。

:::tip 為什麼用新版 git restore 不用舊版 checkout？

因為舊版本用於檔案還原的指令混雜，例如 `git reset <pathspec>` 可以指定檔案踢出預存，但是 `git reset --hard` 卻不能指定檔案；而 `git checkout -- .` 可以還原未預存的檔案，卻又不能處理已預存的檔案。

如果 checkout 已經用的很熟練繼續用當然也沒問題。

:::

### 放棄未預存的程式碼

```sh
git restore <pathspec>
```

### 放棄程式碼，不管是否預存

```sh
git restore -SW <pathspec>
```

### 查看舊版的檔案

```sh
git restore --source=<hash> <pathspec>
```

---

## 還原提交 git reset

```sh
git reset [<mode>] [<commit>] [<pathspec>]
```

用於還原提交，把現在的提交歷史直接還原到指定版本，由於直接還原到指定版本，所以包括版本本身提交歷史也都會保存不會消失。預設模式是 mixed，預設還原全部檔案。三種模式分別是代表

1. soft: 只清除 commit，其他不動 (不會 unstage)
2. mixed: 清除 commit 和 add (unstage)
3. hard: 除了 commit 和 add 以外，連你的寫的程式都刪了，謹慎使用！

三個模式可能讓人有點混亂，不過實際上只是 hard 加上是否改變 stage 的差別，一般來說預設就好。這個指令不常用，因為他的使用場景大部分都可被互動式變基取代，不過還是提供幾個使用情境方便記憶。

### 程式改壞了想回復到遠端版本

```sh
git reset --hard origin/main
```

origin 代表遠端別名，main 是遠端分支名稱。

### 復原錯誤的 Git 操作

有時候我們不小心改壞就可以使用 reset 直接復原到指定 commit，方式是

```sh
git reflog                           # 顯示 Git 操作版本歷史，按 q 離開
git reset <hash>                     # 還原到指定的版本
```

這兩個就是 reset 最常用的情境，下方幾個都不常用。

:::info

要進行複雜且不確定結果的操作時，應該先建立一個備份分支才是最佳解法，操作越複雜還原越麻煩。

:::

### 不小心提交，想繼續編輯

```sh
git reset --soft HEAD^               # 清除前一個提交
git reset --soft HEAD~3              # 清除前三個提交
```

這個指令會取消最新的提交，但保留所有程式碼修改，使用 `git commit --amend` 也可達成相同效果（但是 amend 僅限最近一次提交）。

:::info

1. HEAD 代表目前工作的 commit 位置
2. "^" 代表前一個提交，"~n" 代表前 n 個提交

:::

### 放棄未提交的修改

結束測試性的修改後，直接回到上一次提交，這會連程式碼全部刪除，需要謹慎使用。

```sh
# 等同於 git restore -SW .
git reset --hard HEAD
```

### 提交了多個小變更，想整理成一個提交

```sh
git reset HEAD~3
```

這會移除最近的三個提交，清除後再重新 add commit 為一個新提交。

這個情境用 reset 雖然做的到但是很麻煩，請用互動式變基指令，下方會介紹。

### 補充說明

reset 指令不常用，原因是這些情境都能被更方便的互動式變基 (interactive rebase) 取代，最常用的應該只有從遠端還原。

:::danger

雖然初學暫時不會碰到多人合作，但還是必須強調修改提交歷史 **永遠只該用於個人分支**！

:::

---

## 任意修改 git rebase

git rebase 實際上是對分支進行操作，原本應該放在後續文章，但是單純進行這裡要講的互動式變基 (interactive rebase) 時你完全感受不到分支操作，而他的功能之強大值得放在這裡，請見我寫的教學 [使用互動式變基 Interactive Rebase 任意修改提交歷史](/git/interactive-rebase)。

注意 rebase 本質也是在修改提交歷史，而修改提交歷史永遠只該用於個人分支。

---

## 救命稻草 git reflog

當操作錯誤時，git 的日誌功能 git reflog 可以還原操作。直接講使用方法：

```sh
$ git reflog
5293902 (HEAD -> main, origin/main, origin/HEAD) HEAD@{0}: commit: add article: python/regex
62b2d38 HEAD@{1}: rebase (finish): returning to refs/heads/main
62b2d38 HEAD@{2}: rebase (reword): add tags to article
72c5477 HEAD@{3}: rebase (start): checkout HEAD~2
37334c5 HEAD@{4}: commit: add: article tags  # <-- 現在想還原到這個修改

$ git reset --hard HEAD@{4}
```

這樣會回到 rebase 前的狀態。reflog 只會紀錄本地操作，推送到遠端再 clone 下來後不會有 reflog 紀錄，所以為你自己學 Git 的[這篇文章](https://gitbook.tw/chapters/faq/remove-files-from-git)寫錯了。

## 歷史查詢 git log

查詢提交歷史，但是我 99.9% 的時間都用 [alias](/git/advanced-settings-and-aliases#description) 和 GUI 完成，分別對應簡單和複雜的場景，所以沒有必要學這個指令，這裡列出的指令目的是作為 cheat sheet 速查表：

```sh
# 最基礎
git log

# 把兩個加入你的 alias
# 單分支
git log --graph --pretty='%Cred%h%Creset -%C(auto)%d%Creset %s %Cgreen(%ar) %C(bold blue)<%an>%Creset'

# 多分支
git log --graph --pretty='%Cred%h%Creset -%C(auto)%d%Creset %s %Cgreen(%ar) %C(bold blue)<%an>%Creset' --all

# 不要以 pager 模式顯示，直接印在終端
git --no-pager log

# 參數總集
git log -10 --oneline --show-signature --decorate --graph --stat -p --since="2 weeks ago"
```

每個指令的作用是

1. --oneline: 顯示一行
2. --show-signature: 檢查 GPG 簽名
3. --decorate: 顯示分支和標籤名稱
4. --graph: 顯示分支結構
5. --stat: 顯示行數變化數據
6. -p: 以 [patch 格式](/git/keyword#entry-level) 顯示
7. --since: 指定日期

要提醒的是在 pager 模式下可以使用 regex 來查找指定文字，例如使用參數總集的範例時想要直接跳到下一個提交不想要慢慢捲動，可以

1. 按下斜線進入搜尋模式
2. 輸入 regex `^\*\s\w{7}\s` 定位 hash
3. `n` 跳到下一個，`N` 返回上一個

> 此 regex 代表找到行開頭是 `star 空格` 再加上七個 ASCII 再加上一個空格的字串

## 差別比較 git diff

git diff 比起 git log 更沒必要學，最大的問題就是在終端看差異簡直是在折磨自己，這個指令無論是顯示方式和任務類型天生就都適合 GUI 操作，這也是我覺得 GitKraken 最有用的地方[^gui]，可以清楚且快速看出 diff，不要在終端機折磨自己。

除此之外要說明為何本文檔要強調自己不亂造詞，`git diff <commit> > name.patch` 可以生成 patch 檔案[^patch]，用途是方便在不同環境間分享程式修改，例如有時候我們偶爾會在 Google 搜尋到別人程式修改 email，他們就是傳 patch 檔，偏偏就有教學拿 patch 當「提交名詞」的替代，這不就讓人搞混了嗎。

[^gui]: VSCode 的 GitGraph 無法查看多個提交合併在一起的變更，GitKraken 的安裝版則可以一次顯示。
[^patch]: 使用 `git format patch -數字` 也可以生成 patch，使用 `git apply <file>` 可以套用 patch 檔案。

## 結語

到這邊就結束單一分支的操作，你已經可以基本的操作 Git 了，[下一篇文章](./branch)會介紹多分支操作。在初學階段個人使用時不太會用到分支功能，根據需求可以快轉到[遠端儲存庫設定](/git/remote-setup)。
