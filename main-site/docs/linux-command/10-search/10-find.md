---
title: Unix/Linux 的 find 指令使用
sidebar_label: find
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

find 用於找檔案，和 grep 並列我最愛的 linux 指令沒有之一，再搭配正則表達式可以找到所有你想找的資料。

## 常用參數

- `-name` / `-iname`，iname 不區分大小寫
- `-type f` / `-type d` 指定檔案或目錄
- `-mindepth` / `-maxdepth` 限制搜尋深度
- `-exec` / `-ok` 執行任務，`-ok` 執行前詢問，輸入 y/n 選擇是否

次要常用參數

- `-size` 指定檔案大小
- `-atime` / `-ctime` / `-mtime` 找到開啟、創建、修改時間
- `-user` / `-group` / `-perm`：依照擁有者、群組或權限搜尋

## 常用範例

### 指定名稱

1. 找到副檔名: `find . -name "*.jpg"`
2. 找到多個副檔名 `find . -name "*.jpg" -o "*.png"`

### 指定屬性

1. 大於 10MB: `find . -type f -size +10M`
2. 小於 500KB: `find . -type f -size -500k`
3. 找出 3 天內修改過的檔案: `find . -type f -atime +30`
4. 找到權限 644: `find . -type f -perm 644`

### 排除路徑

1. 排除目錄

```sh
find . -name "*.jpg" -o "*.png" -not -path "./path/to/*"
```

2. 排除多個目錄

```sh
find . -type d \( -path './node_modules' -o -path './ripgit/node_modules' -o -path './bar/node_modules' \) -prune -o -print -type f -name '*.md'
```

`prune` 和 `-not -path` 一樣，只是 prune 效率比較高。

3. 只要路徑包含指定名稱就跳過

```sh
find . -type d \( -path '*/node_modules/*' -o -path '*/ripgit/*' \) -prune -o -type f -name '*.md' -print
```

### 刪除檔案

1. 刪除圖片檔

```sh
find /path/to/directory -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" \) -delete
```

2. 刪除 .DS_Store

```sh
find /path/to/directory -type f -name ".DS_Store" -delete
```

### 執行任務

1. 把所有 `*.md` 重新命名為 `*.en.md`

```sh
find . -type f -name "*.md" -not -name "*.*.md" -exec bash -c 'mv "$0" "${0%.md}.en.md"' {} \;
```

`$0` 是找到的檔案，`0%` 移除 `.`，所以替換成 `${0%.en.md}` 就是檔案名稱移除副檔名後，再加上 `.en.md`。

2. 重新命名回 `*.md`

```sh
find . -type f -name "*.en.md" -exec bash -c 'mv "$0" "${0%.en.md}.md"' {} \;
```

## 搭配 grep

沒有要寫這段，因為這有無限可能
