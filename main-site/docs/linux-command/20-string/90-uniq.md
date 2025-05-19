---
title: Unix/Linux 的 uniq 指令使用
sidebar_label: uniq
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-19T02:14:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-19T02:14:00+08:00
---

`uniq` 以行為單位找出**相鄰**的重複行，因為只比較相鄰行所以常搭配 `sort` 使用。

uniq 的語法如下

```sh
SYNOPSIS
       uniq [OPTION]... [INPUT [OUTPUT]]
```

`INPUT` 是輸入檔案，不指定時讀取標準輸入；`OUTPUT` 是輸出檔案，預設輸出到標準輸出。

:::info
再次提醒，別忘了 `uniq` 只尋找相鄰的重複而不是比較整個檔案。
:::

## 常用參數

- `-c`：在每行前面顯示該行重複的次數
- `-d`：只顯示重複出現的行（出現次數≥2）
- `-u`：只顯示不重複的行（出現次數=1）
- `-i`：忽略大小寫差異比較行
- `-f N`：比較時忽略每行前面 N 個字元
- `-s N`：比較時忽略每行前面 N 個字元，類似 `-f` 但不跳過欄位，只是字元數

## 常用範例

以 file.txt 內容如下為例

```txt
  1 cherry
  3 apple
  2 banana
  1 cherry
```

### 計算每行出現次數

```sh
sort file.txt | uniq -c
```

輸出

```txt
  3 apple
  2 banana
  1 cherry
```

### 只顯示重複行

```sh
sort file.txt | uniq -d
```

輸出

```txt
  1 cherry
```

如果使用 `cat file.txt | uniq -d` 則不會有輸出。
