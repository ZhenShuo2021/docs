---
title: 附件：關鍵字和符號
description: 這篇文章介紹 Git 中的保留關鍵字，包含 HEAD, ^, ~ 等符號
tags:
  - Git
  - Programming
keywords:
  - Git
  - Programming
last_update:
  date: 2024-09-10T17:51:07+08:00
  author: zsl0621
---

### Git 中的關鍵字與符號

## 基本
1. HEAD：目前工作的 commit 位置
2. "^"：前一個提交，
3. "~n"：前 n 個提交
4. `--`：檔案與其他選項的分界線

範例：
```sh
git reset --hard HEAD^     # 回到前一個提交
git reset --hard HEAD^^^   # 回到前三個提交
git reset --hard HEAD~3    # 回到前三個提交
git checkout --options -- file   # 分界線
```

## 入門
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

## 進階
1. `<<<<<`, `=====`, `>>>>>`：標記合併衝突的區域
```git
<<<<< HEAD
目前的檔案內容
=====
修改的檔案內容
>>>>> hash
```

在 git rebase 中可能會有點反直覺，在 main 中使用 `git rebase feature` 的衝突中，上面是 feature，下面才是 main。原因是 rebase 的 feature 是放進來的 base，main 才是要被放進來的文件，所以才會違反直覺，但邏輯是正確的。