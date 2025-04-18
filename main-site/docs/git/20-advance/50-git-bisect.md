---
title: 使用 Git Bisect 找出錯誤提交
sidebar_label: Git Bisect 找出錯誤提交
slug: /git-bisect
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-03-15T10:12:00+08:00
  author: zsl0621
first_publish:
  date: 2025-03-15T10:12:00+08:00
---

找出錯誤最快的方式是使用二分法切一半找問題，`git bisect` 指令就幫助我們完成這件事，只要提供開頭結尾分別是正確和錯誤的提交，Git 就會自動幫我們在提交歷史中切換，不過哪個提交有問題當然還是要自己確認。

## 使用方式

```sh
git bisect start <壞的 Commit> <好的 Commit>
```

之後就會開始二分法查找，在每次切換確認後輸入指令標記好壞版本

```sh
git bisect good
git bisect bad
```

重複動作直到找到開始改壞的那個提交。使用 `git bisect skip` 跳過，`git bisect reset` 停止搜尋。
