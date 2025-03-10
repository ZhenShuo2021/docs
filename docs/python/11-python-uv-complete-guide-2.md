---
title: UV Python 教學，最佳虛擬環境管理工具（下）
sidebar_label: UV 虛擬環境套件（下）
tags:
  - Python
  - 虛擬環境
keywords:
  - Python
  - 虛擬環境
last_update:
  date: 2025-02-15T10:04:00+08:00
  author: zsl0621
first_publish:
  date: 2024-11-19T14:22:30+08:00
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Python UV 教學，最佳虛擬環境管理工具（下）

本篇文章介紹 **[uv](https://github.com/astral-sh/uv)** 的操作指令，從安裝到發布套件都包含在內，還有抄作業環節，直接複製貼上就能用，適合沒寫過 pyproject.toml 的人快速上手。如果不清楚自己是否該選擇 uv 請看我寫的[Python 虛擬環境管理套件比較](./virtual-environment-management-comparison)。

由於篇幅過長所以拆成兩篇，[上篇](python-uv-complete-guide)在這裡，下篇的指令使用率比較低，但仍然是我挑出來比較實用的指令。

## 發布套件

### 編譯 requirements.txt

https://docs.astral.sh/uv/concepts/projects/sync/#exporting-the-lockfile  

```sh
uv export --no-emit-project --locked --no-hashes -o requirements.txt -q
```

每次都要手動打太麻煩，使用 pre-commit 一勞永逸，自動檢查 lock 檔案是否變動並且匯出。pre-commit 的使用範例可以參考筆者寫的[文章](/memo/python/pre-commit-first-try#pre-commit-configyaml)。

```yaml
# .pre-commit-config.yaml

repos:
  # 使用官方 pre-commit
  - repo: https://github.com/astral-sh/uv-pre-commit
    rev: 0.5.14
    hooks:
    - id: uv-export
      args: ["--no-emit-project", "--locked", "--no-hashes", "-o=requirements.txt", "-q"]
      # 當 uv.lock 變化時才觸發 (當你升級或增減套件)
      files: ^uv\.lock$

  # 另一個範例：改用本地 uv，改用 pip compile
  - repo: local
    hooks:
    - id: run-pip-compile
      name: Run pip compile
      entry: bash -c 'rm -f requirements.txt && uv pip compile pyproject.toml -o requirements.txt --annotation-style line -q'
      language: system
      files: ^uv\.lock$
```

`uv export` 和 `uv pip compile` 某種程度上有些相似又不完全相同，前者用於處理 lockfile，後者用於編譯模糊的主要依賴文件 `requirements.in`，由於還在開發階段就不深入討論。uv 提供各種不同文件的編譯方式，基本上兼容所有想得到的依賴文件。

### 構建套件

https://docs.astral.sh/uv/reference/cli/#uv-build

```sh
uv build --no-sources
```

### 發布套件，以 test.pypi 為例

需要指定 build 路徑，預設在 dist 資料夾中。使用時輸入的帳號是 `__token__`，密碼則是 pypi 提供的 token，注意此指令還在實驗階段隨時可能變動。

```sh
uv publish --publish-url https://test.pypi.org/legacy/ dist/*
```

### 整合 Github CI

一般來說我們不會每次發布都打 build publish，而是使用自動化流程完成套件發布，下方直接附上 Github Actions 方便抄作業，實測沒問題可以直接複製貼上使用。這個設定不使用已經被建議棄用的 token 方式，而是遵照官方的<u>**最佳實踐**</u>使用[可信任發行者](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)方式，在每次 tag 名稱是 `vN.N.N.N` 或 `vN.N.N` 時以及發布 release 時才會啟動，並且建議開啟[手動驗證](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)。

開啟手動驗證的方式是進入專案首頁後

1. 點擊上方 Code/Issues 那排最右邊的 Settings
2. 點擊左側列表的 Environments
3. 如果成功設定會有一個環境名稱是 `publish_pypi`
4. 勾選 Required reviewers 並且設定人員，最多六名。

```yaml
name: PyPI Publish

on:
  release:
    types: [created]

  push:
    tags:
      - 'v*.*.*.*'
      - 'v*.*.*'

jobs:
  publish:
    name: Build and Publish to PyPI
    environment: publish_pypi
    runs-on: ubuntu-latest

    permissions:
      id-token: write
      contents: read

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: uv.lock

      - name: Set up Python
        # 不指定子版本會自動使用最新版
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Build package
        run: uv build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # for test.pypi
        # with:
        #   repository-url: https://test.pypi.org/legacy/
```

### Github Dependabot

開發中，尚未支援。

更新進度請查看 [dependabot/dependabot-core#10478](https://github.com/dependabot/dependabot-core/issues/10478)，uv 建議的替代方案是使用 [Renovate](https://docs.astral.sh/uv/guides/integration/dependency-bots/)，或者也有用戶自己做的[簡易 Actions](https://github.com/EdmundGoodman/update-bot)。

## 使用 `uv tool` 取代 `pipx`

https://docs.astral.sh/uv/guides/tools/

此功能用於取代 pipx，把提供命令行執行的工具安裝在全局環境，例如我一開始只是想測試 uv 時也是用 pipx 安裝的。uv tool 特別的地方是沒有安裝也可以執行，會把套件安裝在一個臨時的虛擬環境中。

使用範例參考官方文檔

```sh
# 安裝 ruff
uv tool install ruff

# 執行 ruff，uvx 等效於 uv tool run ruff
uvx ruff

# 當套件名稱和命令行名稱不一樣時的指令
# 套件名稱 http，需要透過 httpie xxx 執行
uvx --from httpie http

# 升級
uv tool upgrade

# 指定相依套件版本
uv tool install --with <extra-package> <tool-package>
```

## 建立符合 PEP 723 的腳本

PEP 723 規範 Python 腳本的開頭設定以支援辨識該腳本需要哪些套件才能運行，否則像是以往需要多一個 requirements.txt 紀錄套件才能執行腳本。他的格式如下：

```py
# /// script
# # 支援寫註解，要多加上一個 #
# requires-python = ">=3.8"
# dependencies = [
#     "requests<3",
#     "beautifulsoup4",
# ]
# ///
```

設定需要的套件和 Python 版本，現在 uv 可以直接幫你寫：

```sh
uv init --script example.py --python 3.12
uv add --script example.py 'requests<3' 'beautifulsoup4'
```

第一行是建立文件，已經有的話可以跳過該行。

## 從 .env 檔案讀取環境變數

UV 甚至支援讀取 .env 檔，讓你在本地測試時可以隨意的切換各種不同 env 參數，使用範例如下：

```sh
echo "MY_VAR='Hello, world!'" > .env
uv run --env-file .env -- python -c 'import os; print(os.getenv("MY_VAR"))'
```

如此一來你就可以使用多個不同的 .env 檔輕鬆的切換設定而不需要修改文件。

## 結束

本文介紹了從安裝到平常使用，到 pyproject.toml/.pre-commit-config.yaml 抄作業，到發布套件，以及取代 pipx，甚至於建立 PEP 723 腳本/讀取 env 的全部介紹。由於這個工具很新隨時會變動，網路上資訊也少，如果有問題麻煩告知我再修正。
