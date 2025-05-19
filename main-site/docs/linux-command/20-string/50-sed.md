---
title: Unix/Linux 的 sed 指令使用
sidebar_label: sed
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-18T21:51:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-18T21:51:00+08:00
---

sed 和 awk 都是 Linux 中常用的字串處理，awk 注重於欄位分割處理，sed 則是注重在文字的替換。可以理解為搜尋功能比較弱且著重於字串修改的 grep。

一樣請出 Claude 3.7 寫教學，並且使用和 [awk 教學](awk-1) 一樣的範例文本。

## 📘 基礎篇

### 什麼是 sed

sed (Stream EDitor) 是一款強大的文字處理工具，它可以對文字流進行編輯，常用於文本替換、過濾和轉換。sed 並不直接修改原始檔案，而是將處理後的結果輸出到標準輸出，這使得它特別適合搭配其他 Unix 指令來使用。它的設計理念是處理文字串流，因此每次從輸入讀取一行，處理後輸出，再處理下一行。

### 基本語法結構

sed 指令的基本格式如下：

```bash
sed [選項] '指令' 檔案
```

常用選項包括：

- `-n`：不輸出模式空間內容，除非明確要求輸出
- `-e`：允許多個編輯指令
- `-i`：直接修改原始檔案
- `-r` 或 `-E`：使用擴展正則表達式

指令格式通常為：`[地址範圍]/指令/[標誌]`，例如：

```bash
# 顯示 log.txt 檔案中所有內容（相當於 cat）
sed '' log.txt

# 只顯示第一行
sed -n '1p' log.txt
```

### 文本替換

替換指令 `s` 是 sed 最常使用的指令，語法為：`s/原始文本/替換文本/[標誌]`：

```bash
# 將日誌中的 GET 替換為 [GET]
sed 's/GET/[GET]/' log.txt

# 將第一行的 HTTP/1.1 替換為 HTTP/2
sed '1s|HTTP/1\.1|HTTP/2|g' log.txt

# 將第一行的 HTTP/1.1 替換為 HTTP/2，並且寫回，BSD 版本需要加上 ''
sed -i '' '1s|HTTP/1\.1|HTTP/2|g' log.txt

# 使用 g 標誌替換該行所有匹配項
sed 's/Mozilla/Chrome/g' log.txt
```

替換指令常用標誌：

- `g`：全局替換（預設只替換每行第一個匹配項）
- `p`：配合 `-n` 選項，只印出有替換的行
- `i`：不區分大小寫

### 基本正則表達式符號

sed 支援基本正則表達式

```bash
# 找出包含 POST 的行
sed -n '/POST/p' log.txt

# 找出以 192.168 開頭的行
sed -n '/^192\.168/p' log.txt

# 找出以 +0000 結尾的行（注：實際日誌中可能包含引號和其他字元）
sed -n '/\+0000]/p' log.txt
```

### 行範圍選擇

sed 可以通過數字、正則表達式或兩者組合來指定處理範圍：

```bash
# 只處理第 3 行
sed -n '3p' log.txt

# 處理第 2 到第 4 行
sed -n '2,4p' log.txt

# 處理包含 "john" 的行
sed -n '/john/p' log.txt

# 處理從包含 "john" 到包含 "alice" 的行
sed -n '/john/,/alice/p' log.txt
```

### 輸出控制

sed 提供多種輸出控制指令：

- `p`：輸出當前模式空間的內容
- `d`：刪除模式空間的內容，不輸出
- `q`：退出 sed 處理

```bash
# 只顯示包含 Mozilla 的行
sed -n '/Mozilla/p' log.txt

# 刪除包含 Mozilla 的行
sed '/Mozilla/d' log.txt

# 顯示到第 3 行就退出
sed '3q' log.txt

# 刪除空行
sed '/^$/d' log.txt
```

通過這些基礎功能，我們可以實現許多簡單的文本處理任務。在 nginx 日誌處理上，這些指令已經能夠幫助我們提取和格式化我們需要的資訊，例如統計特定 IP 訪問、提取特定請求方法等。

## 📘 中階篇

### sed 與其他命令整合

sed 可以與 Unix 管道符號 (`|`) 結合使用，實現更複雜的處理流程：

```bash
# 先用 grep 過濾出 POST 請求，再用 sed 提取 URL 部分
grep "POST" log.txt | sed 's/.*"POST \([^ ]*\).*/\1/'

# 計算各種 HTTP 請求的數量
sed -n 's/.*"\([A-Z]*\) .*/\1/p' log.txt | sort | uniq -c

# 結合 awk 提取 IP 和請求方法
sed -n 's/\([0-9]*\.[0-9]*\.[0-9]*\.[0-9]*\).*"\([A-Z]*\) .*/\1 \2/p' log.txt | awk '{print $2 " from " $1}'
```

管道符號能將一個命令的輸出連接到另一個命令的輸入，使我們能夠將複雜任務分解為更簡單的步驟。結合 `grep`、`awk`、`sort` 等工具，sed 的功能可以得到極大的擴展。

### 篩選特定內容

sed 可以使用正則表達式進行複雜的內容篩選：

```bash
# 提取所有 IP 地址
sed -n 's/^\([0-9]\{1,3\}\(\.[0-9]\{1,3\}\)\{3\}\).*/\1/p' log.txt

# 提取所有狀態碼為 200 的 URL
sed -n '/" 200 /s/.*"\([A-Z]*\) \([^ ]*\).*/\2/p' log.txt

# 提取所有 Mozilla 用戶代理的日誌
sed -n '/Mozilla/p' log.txt

# 提取特定日期 (18/May/2025) 的日誌
sed -n '/18\/May\/2025/p' log.txt
```

這種篩選技術對於將大量日誌數據減少到我們真正關心的部分非常有用，尤其是當我們需要監控特定請求或分析特定用戶行為時。

### 使用變數與運算

sed 可以通過 shell 變數來增強其功能：

```bash
# 使用 shell 變數在 sed 中進行替換
DATE="17/May/2025"
sed -n "/${DATE}/p" log.txt

# 使用命令替換獲取今天日期並在 sed 中使用
TODAY=$(date +"%d/%b/%Y")
sed -n "/${TODAY}/p" log.txt

# 結合多個變數使用
USER="john"
METHOD="POST"
sed -n "/${USER}.*${METHOD}/p" log.txt
```

注意在使用變數時，sed 表達式需使用雙引號 (`"`) 而不是單引號 (`'`)，這樣 shell 才能進行變數替換。此技術在自動化腳本中特別有用，例如定期處理日誌或報告生成。

### 條件邏輯

sed 支援簡單的條件邏輯，例如否定條件、地址反轉等：

```bash
# 顯示不包含 Mozilla 的行
sed -n '/Mozilla/!p' log.txt

# 顯示不是 GET 請求也不是 POST 請求的行
sed -n '/GET/!{/POST/!p}' log.txt

# 只有當行包含 "john" 時才替換 "POST" 為 "[POST]"
sed '/john/s/POST/[POST]/g' log.txt

# 替換除了包含 "alice" 的行以外所有行中的 "HTTP/1.1" 為 "HTTP/2"
sed '/alice/!s/HTTP\/1.1/HTTP\/2/g' log.txt
```

這些條件結構允許我們在不同情境下應用不同的處理邏輯，極大地增強了 sed 的彈性和功能。

### 多行處理技巧

sed 預設以行為單位處理文字，但也提供了多行處理的功能：

```bash
# 使用 N 命令將下一行加入到模式空間
sed 'N; s/\n/ /' log.txt

# 將包含 "john" 的行與其下一行合併
sed '/john/{N; s/\n/ /}' log.txt

# 處理跨行匹配，找出從 POST 到下一個 GET 之間的所有行
sed -n '/POST/,/GET/p' log.txt

# 使用 H 和 G 命令保存和恢復模式空間
sed -e '/john/h' -e '$G' log.txt
```

多行處理命令包括：

- `N`：將下一行追加到模式空間，中間加換行符
- `D`：刪除模式空間中直到第一個換行符的內容
- `P`：輸出模式空間中直到第一個換行符的內容
- `H`：將模式空間內容追加到保持空間
- `G`：將保持空間內容追加到模式空間

這些功能在處理跨行結構或需要將多行合併處理時非常有用。

### 多重替換與分組

sed 支援在同一行中進行多次替換，以及使用分組和回溯引用：

```bash
# 在同一行中進行多次替換操作
sed 's/GET/\[GET\]/g; s/POST/\[POST\]/g' log.txt

# 使用分組捕獲 IP 和請求方法，然後重新組合
sed 's/\([0-9.]*\).*"\([A-Z]*\) \([^ ]*\).*/\1 made a \2 request to \3/' log.txt

# 使用 -e 參數指定多個編輯命令
sed -e 's/HTTP\/1.1/HTTP\/2/g' -e 's/Mozilla/Chrome/g' log.txt

# 交換日期和 IP 的位置
sed 's/\([0-9.]*\) - .* \[\([^]]*\)\]/\2 - \1/' log.txt
```

分組使用 `\(` 和 `\)` 標記，捕獲的內容可以使用 `\1`、`\2` 等回溯引用在替換部分引用。這在需要重組文本結構時特別有用，例如重新排序日誌字段或提取特定部分。

這些中級技術使 sed 能夠應對更複雜的文本處理需求，無論是格式化輸出、數據轉換還是複雜的條件篩選。在這個層級，sed 已經能夠處理大多數日常的文本處理任務，並為更高級的應用奠定基礎。

## 📘 高階篇

### 檔案處理技巧

在實際應用中，經常需要處理多個文件或進行複雜的檔案操作：

```bash
# 使用 -i 直接修改原始檔案
sed -i 's/GET/\[GET\]/g' log.txt

# 建立備份後修改原始檔案（生成 log.txt.bak 備份）
sed -i.bak 's/POST/\[POST\]/g' log.txt

# 從檔案讀取 sed 指令（script.sed 包含多條 sed 指令）
sed -f script.sed log.txt

# 將多個檔案的處理結果輸出到不同檔案
for file in *.log; do
  sed 's/GET/\[GET\]/g' "$file" > "${file%.log}.processed.log"
done
```

使用 `-i` 參數時要特別小心，因為它會直接修改原始檔案。在重要檔案上操作前，建議先建立備份或先測試命令確保正確性。

### 保留字串與回溯引用

高級 sed 腳本中，回溯引用和複雜捕獲組的使用至關重要：

```bash
# 提取並重新格式化 IP 和時間戳
sed -n 's/\([0-9.]*\).*\[\([^]]*\)\].*/IP: \1, Time: \2/p' log.txt

# 交換 HTTP 方法和 URL
sed 's/\(".*\)\(GET\|POST\|DELETE\)\( [^ ]*\)/\1\3 \2/' log.txt

# 為每個數字添加前綴
sed 's/\([0-9]\+\)/NUM-\1/g' log.txt

# 複雜的條件回溯引用：只替換包含特定用戶的行中的 URL
sed '/john/s#\("[A-Z]* \)\(/[^ ]*\)#\1/redacted#' log.txt
```

回溯引用可以創建非常複雜的文本轉換，例如重組數據格式、條件性替換或標記特定模式。掌握回溯引用能力對於編寫高效的 sed 腳本至關重要。

### 地址範圍進階用法

sed 的地址範圍功能極其強大，可以使用多種方式定義處理範圍：

```bash
# 使用兩個正則表達式定義範圍
sed -n '/POST/,/DELETE/p' log.txt

# 從正則表達式匹配到特定行號
sed -n '/POST/,+2p' log.txt

# 條件地址範圍：處理從 john 用戶到下一個 alice 用戶之間的行
sed -n '/john/,/alice/p' log.txt

# 使用地址步進：每 2 行處理一次
sed -n '1~2p' log.txt

# 複合條件：在特定範圍內只處理匹配額外條件的行
sed '/john/,/alice/{/POST/s/HTTP\/1.1/HTTP\/2/g}' log.txt
```

地址範圍語法的靈活性使得 sed 能夠精確地針對文本的特定部分進行操作，這在處理結構化日誌或需要特定上下文的處理時非常有用。

高階 sed 技術的掌握需要深入理解 sed 的工作原理和內部狀態，尤其是模式空間和保持空間的概念。這些技術結合起來，能夠創建極其強大的文本處理解決方案，從簡單的文本格式化到複雜的數據轉換和報告生成。

在實務應用中，這些高階技術常常與 shell 腳本的其他元素（如條件語句、循環和函數）結合使用，創建出完整的文本處理解決方案。例如，可以創建一個 shell 腳本來監控 nginx 日誌、提取異常請求、生成安全報告並發送郵件通知，而 sed 將是這個解決方案中不可或缺的核心工具。
