---
title: PEP 更新紀錄
tags:
  - Python
  - PEP
  - cheatsheet
keywords:
  - Python
  - PEP
  - cheatsheet
last_update:
  date: 2025-04-23T22:49:00+08:00
  author: zsl0621
first_publish:
  date: 2024-11-29T16:20:00+08:00
---

# 目的

簡單紀錄重要的 PEP proposal 到底講了啥方便未來查詢，不然純數字太難記了。

看到的才會放進來，太舊而且大家已經熟悉的不會放進來。

## PEP 更新紀錄

- PEP 8: 基本命名和程式風格規範
- PEP 440: 發布套件的版本命名規範
- PEP 517/518: 套件發布的設定規範，用於 pyproject.toml
- PEP 585: 更新 type hint，用於 Python 3.9/3.10 之後
- PEP 621/631/639: 專案 metadata，[Poetry 要 2.0 之後才會支援 621](https://github.com/orgs/python-poetry/discussions/5833)
- PEP 723: Inline script metadata，腳本內部可設定依賴
- PEP 744: JIT 編譯
- PEP 751: Python 官方的鎖定檔案 lockfile，用於替代 requirements.txt，[uv 在 0.6.15 支援](https://github.com/astral-sh/uv/releases/tag/0.6.15)
