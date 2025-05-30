---
title: 使用 cherry-pick 揀選指定提交
sidebar_label: 引入提交 Cherry Pick
slug: /cherry-pick
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-02-12T00:44:00+08:00
  author: zsl0621
first_publish:
  date: 2025-02-12T00:44:00+08:00
---

Cherry-Pick 的意思是把指定提交複製過來，由於動作像是摘櫻桃所以取這個名字，不得不說外國人都是命名鬼才，和 force-with-lease 一樣[非常有創意](/git/force-if-includes#有趣小知識)。開發社群通常不鼓勵使用此指令，詳情請見 [Git Cherry Pick 的後遺症](https://blog.darkthread.net/blog/git-cherry-pick-cons/)，我們不管這個問題只學如何使用。

```sh
# 揀選一個提交
git cherry-pick <commit-hash>

# 揀選提交範圍從 1~n，記得加上`^`
git cherry-pick <commit-hash-1>^..<commit-hash-n>

# 揀選指定三個提交
git cherry-pick <commit-hash-1> <commit-hash2> <commit-hash3>
```

這些指令的作用揀選指定的提交，由左至右一個一個重演在現在的提交之後，如果遇到合併衝突，解決方式和 rebase 一模一樣。

> [如何解決合併衝突？](/git/keyword#進階)

cherry-pick 有幾個實用選項如下：

- `-e`: 手動設定提交訊息
- `-n`: 只引入提交內容不新增提交歷史
- `-x`: 自動加上註記說明是 cherry-pick 還有原始 hash
