---
title: Unix/Linux 的 cut 指令使用
sidebar_label: cut
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-18T23:55:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-18T23:55:00+08:00
---

cut 顧名思義就是剪去指定文字，最常用的方式還是在管道符後面使用，不過為了方便介紹還是使用和 [awk 教學](awk-1) 一樣的範例文本。

> macos 一樣要使用 `gcut` 才可以使用 gnu 版本的 cut，安裝指令為 `brew install coreutils`。

## 常用參數

- `-c`: 以「字元（character）」擷取
- `-f`: 以「欄位（field）」擷取（需搭配 `-d`）
- `-d`: 自訂欄位分隔符，預設為 TAB
- `--complement`: 顯示未被選中的部分
- `-s`: 當使用 `-f` 時，略過沒有分隔符的列

## 常用範例

### 依字元擷取

```bash
cut -c 1-13 log.txt

# 搭配管道符
ls -l | cut -c 1-25

# 多個範圍
ls -l | cut -c 1-3,20-25

# 反向匹配範圍
ls -l | cut -c 1-14 --complement
```

### 依欄位擷取

```bash
cut -d ' ' -f -5 log.txt
```

以空格為分隔符，擷取到第 5 欄。

## 範圍語法

- `-f 1`：第 1 欄
- `-f 1,3,5`：第 1、第 3、第 5 欄
- `-f 2-4`：第 2 到第 4 欄
- `-f -3`：第 1 到第 3 欄
- `-f 3-`：第 3 欄起直到最後
