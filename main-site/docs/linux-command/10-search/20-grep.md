---
title: Unix/Linux 的 grep 指令使用
sidebar_label: grep
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

grep (global regular expression print) 的用途是找到符合 regex 的文字，是初學者比較容易搞混的指令，因為太多範例都是搭配管道符使用，其實他的語法是這樣：

```sh
SYNOPSIS
       grep [OPTION]... PATTERNS [FILE]...
       grep [OPTION]... -e PATTERNS ... [FILE]...
       grep [OPTION]... -f PATTERN_FILE ... [FILE]...
```

如果不提供 `-e` 或者 `-f` 參數，第一個參數 `PATTERNS` 將會被視為以換行符號分隔的多個 pattern。grep 使用有兩個注意事項：

1. ripgrep 不只比內建的 grep 還快而且更直觀好用，但還是要學他，因為運行腳本的環境通常不會有 rg 這種東西
2. macOS 內建 grep 版本很舊，請用 brew 安裝 gnu 新版取代

## 常用參數

- `--include` `--exclude` `--exclude-dir`: 尋找/排除哪些檔案，後面加上 GLOB 語法
- `-E` / `-P`: 使用 Extend (ERE) / Perl (PCRE) 版本的 regex，PCRE 是最強的，支援 lookahead/lookbehind/lazy 語法
- `-e`: 設定 pattern，只有一個的話不需指定，有多個 pattern 可以使用多個 `-e`
- `-v`: 反向匹配
- `-w`: 匹配整個文字相符
- `-i, --ignore-case` / `--no-ignore-case`: 是否忽略大小寫
- `-n`: 顯示行數
- `-r`: 遞迴搜尋
- `-l`: 只印檔名
- `-c`: 只印檔名和行數
- `-H, --with-filename` `-h, --no-filename`: 是否顯示檔案名稱
- `-A` / `-B` / `-C`: 輸出三兄弟，設定印出之後 (A)fter 之前 (B)efore 前後 (C)ontext 幾行的文字，參數後面加上數字使用

身為一個正常人類，我強烈建議你直接把在 `bashrc` / `zshrc` 裡面設定 `alias grep='grep --color=auto'`，沒有顏色眼睛會先瞎掉。

## 常用範例

先從基礎說起，雖然幾乎很少這樣用，但是不可能不學會基礎用法吧。

### 找到 `#` 的行

```sh
grep -n '#' README.md
```

### 找到包含圖片的行

我之前在清理 repo 就用到此指令

```sh
grep -iEHnr '\.(jpg|jpeg|png|gif)' --include='*.md' --exclude-dir={.git,cache,node_modules} --color=auto .
```

搭配 VS Code 神之方便，command + 左鍵按下就會直接跳到該檔案的該行。如果要搭配腳本執行，可以把 H 改為 h 就可以直接操作該行。

### 在指定檔案中找 pattern

雖然 grep 支援 GLOB 過濾檔案，但是複雜規則還是交給 find。

```sh
find /path/to/dir -type f -exec grep -H "pattern" {} \;
```

### 查看你的部落格寫了多少標題

```sh
grep '^#+' -rnE --include='*.md' --exclude-dir={.git,node_modules,cache} -H --color=auto | wc -l
```

`^` 限制從頭開始，`+` 匹配至少一個 `#`，`wc -l` 統計行數，附帶一提本文檔庫已經有 1688 個標題了。

### 解析日誌

找到 error 的行

```sh
tail -n 100 /var/log/syslog | grep -i "error"
```
