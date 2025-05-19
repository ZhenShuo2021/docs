---
title: Unix/Linux 的 ripgrep 指令使用
sidebar_label: ripgrep
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-13T23:50:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-13T23:50:00+08:00
---

為何要用 ripgrep？

1. 超快
2. 指令超簡潔
3. 預設參數很人性化
4. 支援自動排除 .gitignore

這些優點足夠讓我們在日常使用 rg 而不是 find，比如說找到 markdown 的指令，他們兩個要寫成這樣：

```sh
# rg 自動套用 .gitignore
rg '^#+.*' -g '*.md'

# grep 要手動設定所有 exclusion GLOB
grep '^#+' -rnE --include='*.md' --exclude-dir={.git,node_modules,cache,build,public} -H --color=auto
```

速度測試方面，rg 也爆打 gnu grep:

| 指令                                                     | 時間  |
| ------------------------------------------------------- | ----- |
| `rg '^#+.*' -g '*.md' -uu`                              | 0.269 |
| `grep '^#+' -rnE --include='*.md' -H --color=auto`      | 2.078 |

不過速度其實沒有很重要，我們通常很少找那麼多檔案，重點是指令超簡潔，使用原版 grep 想各種過濾 GLOB 的同時 rg 用戶已經又刷了兩個短影片了，總結來說 rg 在易用性和速度都全方面勝過內建的 grep，日常用 rg，腳本用 grep。

## 常用參數

- `--iglob`: 無視大小寫的 GLOB
- `-E`: 設定檔案編碼
- `-N`: 不顯示行數
- `-u`: 等同 `--no-ignore`，不排除 .gitignore 等設定檔案
- `-uu`: 等同 `--no-ignore --hidden`，也不排除隱藏檔案和目錄
- `-uuu`: 等同 `--no-ignore --hidden --binary`，也等同 `grep -r`，搜二進制檔案
- `--sort=path`: rg 多線程搜尋，使用 sort 才可排序，也會被限制單線程搜尋

其餘和 grep 相同的參數

- `-l`: 只印檔名
- `-c`: 只印檔名和行數
- `-v`: 反轉匹配
- `-P`: 使用 PCRE2
- `-A` / `-B` / `-C`: 印出上下文範圍
- `--glob -g`: 設定 GLOB
- `-i --ignore-case`: 忽略大小寫
- `-w`: 匹配整個文字相符
