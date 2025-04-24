---
title: 幫重要版本打上標籤
author: zsl0621
slug: /tag
tags:
  - Git
  - 教學
keywords:
  - Git
  - 教學
last_update:
  date: 2025-04-25T11:27:33+08:00
  author: zsl0621
first_publish:
  date: 2024-10-20T02:21:33+08:00
---

# {{ $frontmatter.title }}

用標籤標示重要版本，分為 lightweight 和 annotated 兩種，官方建議使用 annotated。lightweight 是簡單的 refs ，annotated 則是完整的物件對象，包含作者名稱、日期、email、GPG 簽名等資訊。

新手暫時用不到此功能，但是他太簡單了所以放在前面講。

> 什麼是 refs？請見 [Git 中的關鍵字、符號和基本組成](../beginner/keyword#basics)。

## 常用指令

<div style="display: flex; justify-content: center; align-items: flex-start;">

| 功能 | 本地指令 | 遠端指令 |
|---|---|---|
| 列出 | `git tag` | `git ls-remote --tags origin` |
| 建立 | `git tag -a v1.0.0 -m "messages"` | `git push origin v1.0.0` |
| 刪除 | `git tag -d v1.0.0` | `git push origin --delete v1.0.0` |
| 推送 | `git push origin v1.0.0` |  |

</div>

```bash
# 上標籤並且簽名、加上訊息、指定 hash
git tag -s -a v1.0.0 -m "msg" <hash>

# 推送標籤
git push origin v1.0.0

# 列出標籤，使用 `-n9` 可以同時印出訊息
git tag -n<num>

# 列出特定標籤
git tag -l "v1.8.5*"

# 印出標籤訊息
git tag -l --format='%(contents)' <tag name>

# 編輯標籤（刪除並且重建）
git tag <tag name> <tag name>^{} -f -m "<new message>"
```

列出標籤的指令雜亂又複雜保證記不起來，建議直接用 alias 完成。僅適用 ZSH，在 .zshrc 加入這行：

```sh
alias 'gtl'='gtl(){ git tag --sort=-v:refname -n999 --format="[%(objectname:short) %(refname:short)] %(contents:lines=999)%0a" --list "${1}*" }; noglob gtl'
```

之後就可以使用 gtl 指令列出所有標籤，並且支援使用 gtl v0.2 列出所有 v0.2 開頭的標籤。如果想用更多這種奇特簡寫在[命令行優化](/intermediate/git-bash-setup-in-windows)有說明如何操作，包含讓 Windows 也可以使用 Zsh。

## 第二常用指令

| 功能 | 本地指令 | 遠端指令 |
|---|---|---|
| 推送所有標籤 | `git push origin --tags` |  |
| 更新本地標籤 | `git fetch origin --tags` |  |
| 檢出 | `git checkout v1.0.0` |  |
| 驗證 | `git tag -v v1.0.0` |  |
