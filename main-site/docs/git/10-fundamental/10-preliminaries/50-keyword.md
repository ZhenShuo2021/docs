---
title: Git 中的關鍵字、符號和基本組成
sidebar_label: 關鍵字、符號和基本組成
description: 這篇文章介紹 Git 中的保留關鍵字，包含 HEAD, ^, ~ 等符號
slug: /keyword
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2024-09-10T17:51:07+08:00
  author: zsl0621
first_publish:
  date: 2024-09-10T17:51:07+08:00
---

本章節初學者請直接跳過以後再回來看，快點學會實際指令才是真的。

### 基本

1. HEAD：目前工作的 commit 位置
2. `^`：前一個提交，
3. `~n`：前 n 個提交
4. `--`：檔案與函式選項的分界線
5. blob, tree, commit, tag, refs：[Git 的基本單位](#basics)

範例：

```sh
git reset --hard HEAD^           # 回到前一個提交
git reset --hard HEAD^^^         # 回到前三個提交
git reset --hard HEAD~3          # 回到前三個提交
git checkout --options -- file   # 分界線
```

### 入門{#entry-level}

1. `@{n}`：代表分支或引用的歷史位置，用於 reflog
2. `+`：在變更的文件中，代表新增的行
3. `-`：在變更的文件中，代表刪除的行
4. `@@`：用於顯示差異的上下文行號，出現在 `git diff` 的輸出中
5. `...`：用於比較兩個分支間的共同祖先

範例1：

```sh
$ git reflog

e3f9a68 HEAD@{0}: commit message A
a7b8d9c HEAD@{1}: commit message B
```

之後可以使用 git reset --hard HEAD@{1} 還原。

範例2：

```sh
$ git show

diff --git a/src/css/custom.css b/src/css/custom.css
index 58e6b10..4fab7e0 100644
--- a/src/css/custom.css
+++ b/src/css/custom.css
@@ -4,6 +4,10 @@
  * work well for content-centric websites.
  */
 
+blockquote {
+  border-left-width: 4.5px; 
+}
+
```

- `--- +++`：代表提交的前後檔案  
- `@@ -4,6 +4,10 @@`：hunk 標記  
-4,6 表示原文件中從第 4 行開始，有 6 行內容被變更或刪除。  
+4,10 表示在新文件中從第 4 行開始，添加了 10 行內容。  

### 中階

Git 是快照系統不是差異系統，只是具備顯示差異的功能，並且將差異作為壓縮儲存空間的手段。

Git 是分散式系統，每個人都有一份完全相同的鏡像，遠端儲存庫只是同步手段而不是集中管理版本。

Git 是分散式鏡像系統，所以你的 reflog 記錄不會被推送到遠端，你應該沒聽過 reflog 衝突這種東西吧？所以[這篇文章錯了](https://gitbook.tw/chapters/faq/remove-files-from-git)。

### 進階

1. `<<<<<`, `=====`, `>>>>>`：標記合併衝突的區域

```git
<<<<< HEAD
目前的檔案內容
=====
修改的檔案內容
>>>>> hash
```

合併衝突的區域在 git rebase 中會有點反直覺。舉例來說，當我們位於 feat 分支使用 `git rebase main` 的衝突中，上面是 main，下面才是 feat，原因是 rebase 的 main 是原有的 base，feat 才是要被放進來的提交，所以才會違反直覺，但邏輯是正確的。

解決合併衝突的方式就是把 `<<<<<`, `=====`, `>>>>>` 清除掉就好了，看你要留下哪邊或是兩者都要，完整流程是當你使用 `<指令>`（如 merge/rebase/cherry-pick）遇到衝突時，先使用

1. `git status` 找到未合併的路徑 Unmerged paths
2. 編輯該文件選擇你要哪個版本
3. `git add` 預存後使用 `<指令>` --continue，再進行下一個合併衝突解析
4. 使用 --abort 還原成使用 `<指令>` 之前的狀態，使用 --quit 不還原強制退出，請謹慎使用 --quit

## 更進階：blob, tree, tag, commit, refs{#basics}

扣掉 refs 其他四個是 Git 的基本構成，不重要，對於使用沒有任何幫助。

檔案在 Git 中是一個 blob 物件

- blob 物件僅包含檔案的內容，不包含檔案名稱或任何其他元數據
- tree 記錄檔案位置和目錄結構，記錄 blob 和其他子 tree
- tag 物件用於標記特定的 commit
- commit 物件主要功能是記錄元資料，例如作者、編碼、tree、hash、前一個 commit 的 hash、提交訊息等等
- refs 用來指向特定 hash 的人類可讀名稱，如 `refs/heads/main` 指向 main 分支的最新提交，或者標籤，或者遠端分支。

把所有單位串連起來，**commit 指向 tree，tree 指向 sub-tree 和 blob**。這樣短短幾行已經是網路上的一整篇文章了，到底為什麼要寫那麼長，我看了很久才理解，理解完感受到這個知識一點也不重要。如果還是看不懂可以閱讀[加速几十倍 git clone 速度的 --depth 1，它的后遗症怎么解决？](https://blog.csdn.net/qiwoo_weekly/article/details/128710769)，雖然不是專門在講 Git 結構但是比網路上那些文筆糟糕的文章清楚多了。

如果你喜歡語言模型列表式的說明，會變成這樣：

1. Blob (二進位大型物件)

   - 只儲存檔案的實際內容。不包含檔名或其他中繼資料
   - 可以理解為檔案的純內容快照

2. Tree (樹狀結構)

   - 記錄整個目錄結構，包含檔案位置資訊
   - 指向所屬的 blob 物件和指向其他子目錄的 tree 物件
   - 類似檔案系統的目錄結構

3. Tag Object (標籤物件)

   - 用來標記特定的 commit
   - 通常用於版本發布

4. Commit Object (提交物件)

    - 指向一個 tree 物件，這個 tree 物件再指向其他 sub-tree 和 blob，最終形成一個完整的版本快照

5. Refs (參照)

   - 唯一用途就是提供人類可讀的名稱，並指向 commit hash，不是必要組成結構
   - 常見的例子:
     - `refs/heads/main` 指向 main 分支最新的 commit
     - 標籤名稱指向特定的標籤物件
     - 遠端分支的參照

物件之間的關係是：  
commit → tree → (sub-trees + blobs)  
這樣的結構讓 Git 能夠有效地追蹤和管理程式碼的版本變化。
