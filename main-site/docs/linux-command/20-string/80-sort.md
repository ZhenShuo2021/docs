---
title: Unix/Linux 的 sort 指令使用
sidebar_label: sort
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-19T01:56:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-19T01:56:00+08:00
---

`sort` 是 Linux 中用來排序輸入資料的工具，能根據字母、數字、欄位等多種條件排序，處理純文字表格特別實用。

sort 的語法如下：

```sh
SYNOPSIS
       sort [OPTION]... [FILE]...
```

若未指定檔案，則從標準輸入讀取資料。排序結果會輸出到標準輸出。

## 常用參數

- `-n`: 依數值排序
- `-r, --reverse`: 反向排序
- `-k`: 指定欄位
- `-h`: 排序人類可讀字元，如檔案大小
- `-M`: 月份排序
- `-t`: 指定分隔字元（預設為空白）
- `-u, --unique`: 去除重複行
- `-V, --version-sort`: 以版本排序

## 常用範例

### 預設排序（字典順序）

```sh
sort names.txt
```

### 數值排序

```sh
sort -n log.txt
```

### 依欄位排序

使用 [awk 教學](awk-1) 裡面的 Nginx log，以 HTTP method (GET/POST/...) 排列

```sh
sort -k5 log.txt
```

### 指定欄位分隔符號

假設資料為以冒號 `:` 分隔：

```txt
user1:100
user2:25
user3:50
```

以第二欄數字排序：

```sh
sort -t ':' -k2 -n users.txt
```

### 最消耗資源的進程

```sh
# CPU 資源
ps aux | sort -nrk 3,3 | head -n 3 | nl
ps --sort=-pcpu | head -n 6  # 同上，但是 BSD 不支援此參數

# 記憶體資源
ps aux | sort -nk +4 | tail -n 10
ps aux --sort -rss | head  # 同上，但是 BSD 不支援此參數
```
