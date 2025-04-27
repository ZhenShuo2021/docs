---
title: uv Python 教學，最佳專案管理工具（下）
sidebar_label: uv 專案管理工具（下）
slug: /uv-project-manager-3
tags:
  - Python
  - 專案管理工具
  - 套件管理工具
  - 虛擬環境管理工具
keywords:
  - Python
  - 專案管理工具
  - 套件管理工具
  - 虛擬環境管理工具
last_update:
  date: 2025-04-23T22:49:00+08:00
  author: zsl0621
first_publish:
  date: 2024-11-19T14:22:30+08:00
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Python uv 教學，最佳專案管理工具（下）

本文介紹 [uv](https://github.com/astral-sh/uv) 的操作指令，下篇的指令使用率比較低，但仍然是我挑出來比較實用的指令。

## 建立符合 PEP 723 的腳本

PEP 723 規範 Python 腳本的開頭設定以支援辨識該腳本需要哪些套件才能運行，否則以前需要額外的 requirements.txt 紀錄套件才能執行腳本。PEP 723 格式如下：

```py
# /// script
# # 支援寫註解，要多加上一個井字號
# requires-python = ">=3.8"
# dependencies = [
#     "requests<3",
#     "beautifulsoup4",
# ]
# ///
```

這設定了 Python 版本和依賴套件，寫這些很麻煩，不過 uv 可以直接幫你寫：

```sh
uv init --script example.py --python 3.8
uv add --script example.py 'requests<3' 'beautifulsoup4'
```

第一行是建立文件，已經有文件可以跳過第一行，直接使用第二行就會幫你加上設定。

## 使用 `uv tool` 取代 `pipx`

https://docs.astral.sh/uv/guides/tools/

此功能用於取代 pipx，把提供命令行執行的工具安裝在全局環境，例如筆者一開始測試 uv 時也是用 pipx 安裝的。uv tool 特別的地方是沒有安裝也可以執行，會把套件安裝在一個臨時的虛擬環境中。

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

## 建立符合 PEP 751 的鎖定文件{#pep751-lockfile}

Python 不只[專案管理工具一團亂](https://chriswarrick.com/blog/2023/01/15/how-to-improve-python-packaging/#summary)，鎖定文件也沒有規範，直到這個月 (2025/4) PEP 751 被採納之後現在終於有統一的 pylock.toml 用以替代 requirements.txt，uv 相關指令如下

```sh
# 從 uv.lock 生成 pylock.toml
uv export -o pylock.toml

# 從 pylock.toml 安裝套件
uv pip sync pylock.toml
```

值得注意的是 pylock.toml 不支援 uv 全部功能，所以 uv 還是會繼續使用 uv.lock 管理鎖定文件。

## 在 Docker 中使用 uv

uv 對 Docker 的支援也很完善且積極，文檔介紹了各種常見問題，直接查看[文檔](https://docs.astral.sh/uv/guides/integration/docker/)。

## 發布套件

### 編譯 requirements.txt

https://docs.astral.sh/uv/concepts/projects/sync/#exporting-the-lockfile  

> 鎖定檔案的相關資訊可以參考[建立使用 PEP 751 的鎖定文件](#pep751-lockfile)

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

更新進度請查看 [dependabot/dependabot-core#10478](https://github.com/dependabot/dependabot-core/issues/10478)，uv 建議的替代方案是使用 [Renovate](https://docs.astral.sh/uv/guides/integration/dependency-bots/)，或是用戶自己做的[Github Actions Bot](https://github.com/EdmundGoodman/update-bot)。

## 結束

本文介紹了從安裝到平常使用，到 pyproject.toml/.pre-commit-config.yaml 抄作業，到發布套件，以及取代 pipx，甚至於建立 PEP 723 腳本/讀取 env 的全部介紹。
