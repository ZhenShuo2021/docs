---
title: Unix/Linux 的 read 指令使用
sidebar_label: read
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-19T16:52:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-19T16:52:00+08:00
---

read 用於從標準輸入或檔案中讀取資料。

## 常用參數

- `-r`: 不解析反斜線為轉義字元
- `-p <提示文字>`: 顯示提示訊息
- `-s`: 隱藏輸入內容
- `-t <秒數>`: 設定輸入等待的秒數，超時則結束
- `-n <字元數>`: 限定最多讀取的字元數
- `-d <字元>`: 指定輸入終止字元
- `-a <變數名>`: 將輸入以空白分割，存入陣列變數

## 常用範例

### 讀取多個變數

```bash
# 讀取多個變數
read -p "請輸入姓名和年齡: " name age
echo "姓名: $name, 年齡: $age"

# 讀取陣列
read -a colors -p "請輸入幾種顏色(用空格分隔): "
echo "第一個顏色: ${colors[0]}"
echo "所有顏色: ${colors[@]}"
```

### 指定分隔符號

使用 IFS 環境變數可以設定分隔符號，並且交給 read 讀取：

```sh
# 兩者相同
echo "owner_name/repo_name" | IFS='/' read -r owner repo
IFS='/' read -r owner repo <<< owner_name/repo_name
```

意思是設定臨時的環境變數 IFS 指定以 `/` 分隔字串。

因為 shell 全都是字串處理，所以除了使用 IFS 以外也有很多不同方式可以完成，比如說也可以用 `tr` 或者 `awk` 來完成，但就會非常麻煩：

<details>

<summary>其他方式</summary>

```sh
# tr
read -r owner repo <<< "$(tr '/' ' ' <<< owner_name/repo_name)"

# cut + tr
read -r owner repo <<< "$(echo owner_name/repo_name | cut -d'/' -f1,2 | tr '/' ' ')"

# awk
read -r owner repo <<< "$(echo owner_name/repo_name | awk -F/ '{print $1, $2}')"

# set
set -- owner_name/repo_name; owner=${1%/*} repo=${1#*/}
```

</details>

### 讀取檔案

處理檔案內容我整理為以下幾種使用方式

```bash
# 逐行讀取檔案
while read line; do
    echo "處理行: $line"
done < file.txt

# 逐行讀取檔案到陣列中（僅適用於 bash 4.0+)
mapfile -t lines < file.txt
for line in "${lines[@]}"; do
  echo "$line"
done

# 逐行讀取檔案到陣列中，不使用 mapfile
lines=()
while IFS= read -r line; do
  lines+=("$line")
done < file.txt

for line in "${lines[@]}"; do
  echo "$line"
done

# 讀取整個檔案，有多種方式，第一種效率最高
content=$(<file.txt)
content=$(cat file.txt)

# CSV 檔案處理
while IFS=, read name age city; do
    echo "姓名: $name, 年齡: $age, 城市: $city"
done < data.csv
```

### 設定條件

控制使用者輸入的方式或設定特定條件：

```bash
# 隱藏輸入內容
read -s -p "請輸入密碼: " password
echo -e "\n密碼長度: ${#password}"

# 設定輸入時限
read -t 5 -p "請在5秒內輸入你的名字: " name
echo "你好，$name"

# 限制輸入字元數
read -n 1 -p "請按任意鍵繼續..." key
echo -e "\n你按了: $key"
```

### 限制輸入

限制只能輸入 yes/no

```bash
yes_no_prompt() {
    local prompt="$1"
    local response
    while true; do
        read -p "$prompt (y/n): " response
        case "$response" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo]) return 1 ;;
            *) echo "請只輸入 yes/y 或 no/n" ;;
        esac
    done
}

if yes_no_prompt "是否要刪除此檔案"; then
    echo "開始刪除檔案..."
else
    echo "已取消操作"
fi
```
