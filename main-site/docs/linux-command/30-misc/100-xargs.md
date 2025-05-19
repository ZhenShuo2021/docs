---
title: Unix/Linux 的 xargs 指令使用
sidebar_label: xargs
tags:
  - Linux
  - Cheatsheet
keywords:
  - Linux
last_update:
  date: 2025-05-15T16:22:00+08:00
  author: zsl0621
first_publish:
  date: 2025-05-15T16:22:00+08:00
---

xargs (extended arguments) 用途是把文本依照空隔分開，通常會再搭配指令執行。

## 常用參數

- `-n`: 每次傳幾個參數
- `-d --delimiter`: 設定分隔符
- `-E`: 設定結束字元
- `-p --interactive`: 互動式，每次執行前確認
- `-r --no-run-if-empty`: 輸入為空則不執行

## 常用範例

先從基礎說起，雖然幾乎很少這樣用，但是不可能不學會基礎用法吧。

```sh
xargs -E EOF
```

接著輸入

```txt
a
bb
ccc
EOF
```

最後就會被拆成 a, bb, ccc 三個輸出，`-E EOF` 設定 EOF 是結束字元，不設定的話使用 `ctrl+D` 也可以結束輸入。

### 移除 pip 所有套件

xargs 比較常見的是搭配管道符號使用，比如以下：

```sh
pip freeze | xargs pip uninstall -y
```

`pip freeze` 會輸出以下

```sh
pip freeze
certifi==2025.4.26
charset-normalizer==3.4.2
colorama==0.4.6
idna==3.10
requests==2.32.3
urllib3==2.4.0
```

然後交給 xargs 移除。

### 拆分字串

變成每個字一行，最後輸出給 a.txt。

```sh
xargs -n 1 echo <<< "a b c d" > a.txt
```

`<<<` 把字串重定向到指令的 stdin，也可以改用 heredoc 完成：

```sh
xargs -n 1 echo <<EOF > b.txt
```

然後輸入你要的文字，最後輸入 EOF 結束。
