---
title: Unix/Linux 的 curl + wget 指令使用
sidebar_label: curl + wget
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-17T01:47:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-17T01:47:00+08:00
---

cURL 用來發送 HTTP 請求，學他就好不用學 wget，因為 curl 更多參數用法，輸出控制更方便，兩者空間也差不多，除此之外再請語言模型列出其他優點

- 支援更多 HTTP 方法與 header 操作
- 更適合與 RESTful API 互動
- 輸出格式更易控制
- 支援更多協定（如 SCP、SFTP 等）
- 內建 JSON payload 操作與 token header 更直觀

需要使用 wget 的情況在下方章節會說明。

## 常用參數

- `-O`: 以遠端檔案名稱下載檔案
- `-o`: 指定下載檔案的本地檔名
- `-C -`: 續傳下載（從中斷處繼續）
- `-H`: 自訂 HTTP 標頭
- `-X`: 指定 HTTP 方法（如 GET、POST、PUT、DELETE 等）
- `-d`: 傳送資料（POST 請求的 body）

腳本安裝四劍客：

- `-f`: HTTP 錯誤時直接失敗（不輸出內容）
- `-s`: 靜默模式，不顯示進度或錯誤訊息
- `-S`: 顯示錯誤訊息（需搭配 `-s` 使用）
- `-L`: 跟隨 HTTP 重定向

## 常用範例

### 發送請求

1. GET

    ```bash
    curl https://example.com
    ```

2. POST

    ```bash
    curl -X POST -d "a=1&b=2" https://example.com
    ```

3. 加上標頭

    ```bash
    curl -H "Authorization: Bearer TOKEN"
    ```

4. 送出 JSON

    ```bash
    curl -H "Content-Type: application/json" -d '{"key":"value"}' https://api.example.com
    ```

### 檔案下載

1. 下載檔案 `-O`

    ```bash
    curl -O https://example.com/file.zip
    ```

2. 自訂檔名 `-o`

    ```bash
    curl -o myfile.zip https://example.com/file.zip
    ```

3. 續傳 `-C -`

    ```bash
    curl -C - -O https://example.com/large.zip
    ```

### 腳本安裝四劍客

`-fsSL` 參數應該到處都看過，分別是以下用途

- `-f`: HTTP 錯誤時直接失敗  
- `-s`: 靜默模式  
- `-S`: 顯示錯誤（需搭配 `-s`）
- `-L`: 跟隨重定向

## wget

wget 天生比較適合下載大檔案，因為他的續傳比較好用：

```sh
wget -c https://example.com/largefile.zip
```

除此之外功能都沒有 curl 好，他擁有的唯一一個 curl 沒有的功能就是遞迴下載網站：

1. 下載離線版網站

    ```sh
    wget -mpEk "url"
    ```

2. 遞迴下載網站資源，下載某目錄下所有 `.zip` 檔案

    ```sh
    wget -r -np -l 1 -A zip http://example.com/download/
    ```

- `-r`: 遞迴
- `-np`: 不往上層目錄走
- `-l 1`: 只遞迴 1 層
- `-A zip`: 只下載副檔名為 zip 的檔案
