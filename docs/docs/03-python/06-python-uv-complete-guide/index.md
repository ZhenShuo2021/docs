---
title: UV Python完整教學：從安裝到發佈套件，最佳虛擬環境管理工具
description: UV Python完整教學：從安裝到發佈套件，Python 最佳虛擬環境管理工具
sidebar_label: UV 虛擬環境管理套件教學
tags:
  - Programming
  - Python
  - 虛擬環境
keywords:
  - Programming
  - Python
  - 虛擬環境
last_update:
  date: 2024-11-24T17:20:00+08:00
  author: zsl0621
---

# UV Python完整教學：從安裝到發佈套件，最佳虛擬環境管理工具
本篇文章介紹 uv 的日常操作指令，從安裝到發布套件都包含在內，一部分是作為自己的 cheatsheet。筆者一向不喜歡寫這種純指令的文章，因為是網路上已經充斥一堆類似文章了沒必要又一篇浪費讀者作者雙方時間，但是本文是全中文圈第一個完整介紹操作的文章所以沒這問題。

除此之外本文還有抄作業環節，直接複製貼上就能用，適合沒寫過 pyproject.toml 的人快速上手。

如果不清楚自己是否該選擇 uv 的請觀看[上一篇文章](/docs/python/virtual-environment-management-comparison)。

## 簡介
以一句話形容 uv，那就是完整且高效的一站式體驗。uv 是 2024/2 才首發的新工具，簡單摘要幾個特點：

1. 由 rust 撰寫，標榜快速，比 Poetry 快十倍以上
2. 支援 uv.lock 鎖定套件版本
3. 支援 Python 版本管理，取代 pyenv
4. 支援全域套件安裝，取代 pipx
5. 支援 pyproject.toml
6. 發展快速，發布不到一年已經有 26k 星星

把特點二和三加起來就是我們的最終目標了，既支援 lock 檔案管理套件，又支援 Python 版本管理，也沒有 pipenv 速度緩慢且更新停滯的問題，是目前虛擬環境管理工具的首選。使用體驗非常流暢，和原本的首選 Poetry 互相比較，uv 內建的 Python 版本管理非常方便，不需再使用 pyenv 多記一套指令，不需要 pipx 管理全局套件，本體雖然不支援建構套件，但是設定完 build-system 使用 `uv build` 和 `uv publish` 一樣可以方便的構建和發布，做了和 pip 類似的接口方便以往的使用者輕鬆上手，再加上[超快的安裝](https://astral.sh/blog/uv-unified-python-packaging)和解析速度錦上添花，筆者認為目前虛擬環境管理工具首選就是他了。

身為新穎又備受矚目的套件，目前的更新速度非常快，[兩個月就把問題解決了](https://www.loopwerk.io/articles/2024/python-uv-revisited/)。

為何選擇 uv？把結尾的話放到這邊來：「一個工具完整取代 pyenv/pipx，又幾乎包含 poetry 的功能，速度又快」，這麼多優點是我可以一次擁有的嗎，太夢幻了吧。

## 前置作業

### 安裝 uv
https://docs.astral.sh/uv/getting-started/installation/

自行閱讀文檔，不浪費時間複製貼上，值得注意的是使用獨立安裝程式安裝的話 uv 可以更新自己，其餘方式沒有這功能。  



### 設定 Python 版本
https://docs.astral.sh/uv/concepts/python-versions/  
https://docs.astral.sh/uv/guides/install-python/  

選用這種工具的用戶應該都需要管理多個 Python 版本，所以先從 Python 版本管理開始說明。

``` sh
# 安裝指定版本
uv python install 3.12

# 列出基本版本
uv python list

# 列出所有版本
uv python list --all-versions

# 只列出安裝版本
uv python list --only-installed

# 找到執行檔路徑
uv python find
```

### 初始化專案
https://docs.astral.sh/uv/reference/cli/#uv-init

設定好 Python 版本後就是初始化專案，使用 `app` 參數設定專案名稱，使用 `build-backend` 參數設定專案的構建後端，也可以直接使用 `uv init`。

```sh
uv init --app test --build-backend hatch
```


### 建立虛擬環境
https://docs.astral.sh/uv/pip/environments/ 

接下來是建立虛擬環境，名稱和 Python 版本都是可選項目。

```sh
uv venv <name> <--python 3.11>
source .venv/bin/activate
```

## 套件管理
### 生產套件管理
https://docs.astral.sh/uv/concepts/projects/dependencies/

此處是有關套件處理相關的指令，常用的大概有以下幾項。

```sh
# 把套件加入 pyproject.toml 中
uv add

# 把套件從 pyproject.toml 中移除
uv remove

# 列出所有套件
uv pip list

# 基於 pyproject.toml 對目前環境中的套件進行同步，包含開發者套件
uv sync

# 同步但忽略開發者套件
uv pip sync pyproject.toml

# 建立鎖定檔案
uv lock

# 移除所有套件 https://docs.astral.sh/uv/pip/compile/
uv pip freeze > unins && uv pip uninstall -r unins && rm unins

# 升級指定套件，不指定則全部升級
uv lock --upgrade-package requests
uv lock --upgrade  <package>

# 重新驗證套件快取
uv sync --refresh
```


### 開發套件管理
https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies

此處設定有關開發者套件的所有項目，[有兩種方式](https://stackoverflow.com/questions/78902565/how-do-i-install-python-dev-dependencies-using-uv)，根據pyproject.toml設定有所不同：

- 方式一：使用正統的開發套件設定 [tool.uv]

```toml
# toml 的設定
[tool.uv]
dev-dependencies = [
  "pytest",
]

# 在命令行中使用這個指令進行新增和同步
uv add --dev pytest
uv sync
```

- 方式二：使用旁門左道的可選套件 [project.optional-dependencies]

```toml
# toml 的設定
[project.optional-dependencies]
network = [
    "httpx>=0.27.2",
]

# 在命令行中使用這個指令進行新增和同步
uv add httpx --optional network
uv pip install -r pyproject.toml --extra dev
```


### 重設環境中所有套件
https://docs.astral.sh/uv/pip/compile/#syncing-an-environment

```sh
# 同步txt
uv pip sync requirements.txt

# 同步toml
uv pip sync pyproject.toml
```

### 使用 uv add 和 uv pip 安裝套件的差異
https://docs.astral.sh/uv/configuration/files/#configuring-the-pip-interface

`uv add` 用於正式專案套件，`uv pip` 則是臨時測試，不會寫入pyproject.toml。

## 🔥 pyproject.toml 範例 🔥
既然 uv 的一站式體驗這麼好，那本文也提供一站式體驗，連 `pyproject.toml` 基礎範例都放上來提供參考，複製貼上後建立 README.md 空檔案，使用 `uv venv --python xxx` 還有 `uv sync` 就完成了，細節再自己慢慢改就好。

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/package_name"]

[project]
name = "your-project-name"
version = "0.1.0"
description = "project description"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["beautifulsoup4>=4.12.3", "requests>=2.32.3"]


[tool.uv]
dev-dependencies = [
    "ruff >= 0.7.1",
    "mypy >= 1.13.0",
    "pre-commit >= 4.0.0",
    "isort >= 5.13.2",
]

[project.optional-dependencies]
network = [
    "httpx>=0.27.2",
]
```

如此輕鬆就可以完成 Python 版本下載和設定 + 虛擬環境建立 + 套件安裝，對比以往的 pyenv + poetry 組合我們需要使用這麼繁瑣的指令：

```sh
# 下載和設定版本
pyenv install 3.11.5
pyenv local 3.11.5

# 確認 Python 版本
python --version
poetry config virtualenvs.in-project true
poetry env use python3.11.5

# 啟動虛擬環境，安裝套件並且檢查
poetry shell
poetry install
poetry install --with dev
poetry show
```

在 uv 指令數量減少一半，而且 Poetry 的 "etry" 有夠難打每次敲快一點就打錯。

```sh
# 建立虛擬環境 + 安裝
uv venv --python xxx

# 進入環境
source .venv/bin/activate

# 安裝所有套件
uv sync

# 檢查
uv pip list
```

如果需要更進階的功能，比如說把依賴套件分群組管理，請參照[官方文檔](https://docs.astral.sh/uv/concepts/projects/dependencies/)中的 dependency-groups。

## 發布套件

### 編譯 requirements.txt
https://docs.astral.sh/uv/pip/compile/

```sh
uv pip compile pyproject.toml -o requirements.txt
```

每次都要手動打太麻煩，使用 pre-commit 一勞永逸，自動檢查和匯出套件解析結果，pre-commit 的使用範例可以參考筆者寫的[文章](/memo/python/first-attempt-python-workflow-automation#pre-commit-configyaml)。

```yaml
# .pre-commit-config.yaml

repos:
  - repo: local
    hooks:
    - id: run-pip-compile
      name: Run pip compile
      entry: bash -c 'uv pip compile pyproject.toml -o requirements.txt'
      language: system
      files: ^pyproject.toml$
```

### 構建套件
https://docs.astral.sh/uv/reference/cli/#uv-build

```sh
uv build
```

### 發布套件，以 test.pypi 為例
需要指定 build 路徑，預設在 `dist` 中。使用時輸入帳號是 `__token__`，密碼則是 pypi 提供的 token。此指令還在實驗階段隨時可能變動

```sh
uv publish --publish-url https://test.pypi.org/legacy/ dist/*
```

### 整合 Github CI
一般來說我們不會每次發布都慢慢打 build publish，會使用自動化流程完成套件發布，uv 的流程設定請參考[這篇文章](https://www.andrlik.org/dispatches/til-use-uv-for-build-and-publish-github-actions/)。

## uv tool 取代 pipx
https://docs.astral.sh/uv/guides/tools/

此功能用於取代 pipx，將提供命令行執行的工具全局安裝，例如我一開始只是想測試 uv 時也是用 pipx 安裝的。uv tool 特別的地方是沒有安裝也可以執行，會把套件安裝在一個臨時的虛擬環境中。

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

## 結束！
本文應該是繁體中文圈第一篇完整介紹文章，內容從安裝到平常使用，到 pyproject.toml/.pre-commit-config.yaml 抄作業，到發布套件，以及取代 pipx 全部介紹，並且不講廢話。

由於這個工具很新隨時會變動，網路上資訊也少，如果有問題麻煩告知我再修正。

整體下來最心動的就是不需要 pyenv/pipx，也不用記 poetry 對應 Python 解釋器的指令，全部都濃縮在 uv 一個套件中，加上執行速度快，更新很勤勞（2024/11 看下來每天都有 10 個 commit，嚇死人），社群狀態又很健康 (對比兩個競爭對手 [PDM is a one-man-show, like Hatch](https://chriswarrick.com/blog/2024/01/15/python-packaging-one-year-later/))，一個工具完整取代 pyenv/pipx，又幾乎包含 poetry 的功能，速度又快，難怪竄升速度這麼快。
