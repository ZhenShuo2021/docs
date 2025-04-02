---
title: uv Python 教學，最佳專案管理工具（中）
sidebar_label: uv 專案管理工具（中）
description: 本文介紹 uv 的操作指令，主要介紹日常指令，本文的重點是 uv run，網路上沒幾篇文章知道這個強大的指令。
slug: /uv-project-manager-2
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
  date: 2025-03-27T18:52:00+08:00
  author: zsl0621
first_publish:
  date: 2024-11-19T14:22:30+08:00
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Python uv 教學，最佳專案管理工具（中）

本文介紹 [uv](https://github.com/astral-sh/uv) 的操作指令，主要介紹日常指令，如果不清楚自己是否該選擇 uv 請看我寫的 [Python 專案管理工具比較](./best-python-project-manager)。

筆者一向不喜歡寫這種純指令的文章，因為網路已經充斥一堆相似文章了沒必要又一篇浪費讀者作者雙方時間，但是本文是全中文第一個完整介紹操作的文章所以沒這問題，本文的重點是 [uv run](#uv-run)，網路上沒幾篇文章知道這個強大的指令。

## 套件管理

### 生產套件管理

https://docs.astral.sh/uv/concepts/projects/dependencies/

此處是有關套件處理相關的常用指令，熟記這些指令之後就可以替換掉 venv/pyenv/poetry/pipx 了。

```sh
# 安裝套件並且設定到 pyproject.toml 中
uv add

# 移除套件並且從 pyproject.toml 的設定中移除
uv remove

# 列出所有已經安裝的套件
uv pip list

# 更新 uv.lock，使用uv add時該檔案不會自動更新
# 此檔案是詳細的套件版本鎖定檔案，用於提供可復現的運行環境
# 加上-U可以允許原有的套件更新
uv lock

# 基於 uv.lock 對目前環境中的套件進行同步，包含開發者套件
uv sync

# 同步但忽略開發者套件
uv sync --no-dev

# 在虛擬環境中執行指令
uv run <commands>

# 移除所有套件（只移除環境中的套件不會移除 toml 中的套件）
uv pip freeze > u && uv pip uninstall -r u && rm u

# 升級指定套件或全部升級
uv sync -P <package>
uv sync -U

# 重新驗證套件快取，--fresh 在任何指令下都可使用
uv sync --refresh
```

### 開發套件管理

https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies

設定開發套件，此區域的套件不會被構建和發布，使用 `--dev <pkg>` 新增，還可以用 `--group` 幫開發套件設定套件群組方便管理。比如說我們需要 pytest 進行單元測試，只需要使用 `uv add --dev pytest` 把 pytest 新增到 dev 群組中而不會影響生產套件。下方的範例我們新增了 pytest 作為 dev 群組，以及 ruff 作為 lint 的群組。

```toml
# 把 pytest 套件新增到 dev 群組，等同於 uv add --group dev pytest
uv add --dev pytest

# 再把 ruff 套件新增到 lint 群組
uv add --group lint ruff

# toml 對應的更新
[dependency-groups]
dev = ["pytest"]
lint = ["ruff"]
```

### 可選套件管理

https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies

幫專案增加可選套件，這用於發佈在 PyPI 時讓用戶決定是否安裝完整的套件功能。

:::info 可選套件
舉例來說，像是 httpx 的 http2 功能是可選，如果我們想安裝 httpx + http2 要使用 `pip install 'httpx[http2]'` 才會安裝 http2 這個可選套件
:::

```toml
# 在命令行中使用這個指令，新增可選套件 matplotlib 到 plot 群組
uv add matplotlib --optional plot

# toml 中的對應的更新
[project.optional-dependencies]
plot = ["matplotlib>=3.6.3"]
```

這樣設定之後 matplotlib 就會變成可選套件。

### 重設環境中所有套件

https://docs.astral.sh/uv/pip/compile/#syncing-an-environment

把套件版本同步到生產版本，移除虛擬環境裡沒有被文件設定的套件。

```sh
# 同步txt
uv pip sync requirements.txt

# 同步toml
uv pip sync pyproject.toml

# 或者更乾淨重新安裝，這個指令會刷新快取
uv sync --reinstall --no-dev

# 直接清除快取檔案
uv clean
```

### 使用 uv add 和 uv pip 安裝套件的差異

https://docs.astral.sh/uv/configuration/files/#configuring-the-pip-interface

`uv add` 用於正式專案套件，和 `uv remove` 成對使用，會修改 pyproject.toml；`uv pip` 則是臨時測試，不會寫入 pyproject.toml。

## 🔥 強大的 uv run 功能 🔥{#uv-run}

https://docs.astral.sh/uv/guides/scripts/
https://docs.astral.sh/uv/reference/cli/#uv-run

經過上面的設定我們知道 uv 可以設定開發套件和開發群組，結合這些功能可以讓日常的開發輕鬆許多，這些輕鬆主要體現在 `uv run` 指令之上。有了 `uv run` 之後我們連虛擬環境都不用進入就可以直接執行腳本，但是他真正的特色是支援靈活的版本和依賴切換，例如我們可以

1. 使用 `--with <pkg>` 臨時測試某些套件而不需安裝
2. 使用 `--group` `--only-group` `--all-groups` `--no-group` 設定執行時包括哪些開發群組的套件
3. 使用 `--extra` `--all-extras` `--no-extra` 指定包括哪些可選套件
4. 使用 `--with-requirements` 指定包括 txt 文件的套件執行
5. 使用 `--find-links` 可以直接包括來自 .whl/tar.gz/.zip/URL 的套件
6. 使用 `--python` 允許我們指定使用不同的 Python 版本執行
7. 使用 `--isolated` 在臨時的隔離空間獨立運行
8. 使用 `--no-sync` 可以關閉運行前的同步功能  
9. 使用 `--no-dev` 忽略開發套件運行
10. 使用參數包含網址時會臨時下載並且被視為腳本執行

光看這些選項可能沒什麼感覺，我們稍微討論一下在實際開發中這些選項提供了多大的方便性。想像需要臨時測試一個套件的情境，以前要先 pip install 安裝，然後執行腳本，事後還要從環境中移除，但是現在這三個步驟直接被濃縮成一個 `--with <pkg>` 了，類似的情境也發生在想要搭配可選套件進行測試，現在只要使用 `--extra` 選項就可以自動包含該群組的套件，甚至使用 `--find-links` 連安裝包都可以使用；或者是臨時想要在一個乾淨的環境執行，現在只需要 `--isolated` 就取代掉以前需要三四步指令才能完成的設定；`--python` 選項乍看之下是提供測試不同 Python 版本使用，但是我們可以把他當作 pyenv 來用，使用 `uv run --python 3.12 python -m venv .venv` 叫 3.12 版本的 Python 來建立虛擬環境[^pyenv]，等效於 pyenv-virtualenv 的功能，非常方便。

以往這些指令都要在不同的套件搭配各自的參數完成，現在只需要放在一個列表就可以涵蓋數個不同開發場景的指令組合，提供非常強大的開發便利性，經過一段時間的使用後我認為 `uv run` 這個功能相較於速度這個特色才是他最吸引人的地方。

附帶一提這些參數大多數也都適用於 uv sync 等指令。

[^pyenv]: 使用 `uv venv --python 3.12` 是透過 uv 建立虛擬環境，無法在虛擬環境中使用 `pip`。

### 從 .env 檔案讀取環境變數

還沒完，uv 甚至支援讀取 .env 檔，讓你在開發時可以隨意切換各種不同 env 參數，使用範例如下：

```sh
echo "MY_VAR='Hello, world!'" > .env
uv run --env-file .env -- python -c 'import os; print(os.getenv("MY_VAR"))'
```

如此一來你就可以輕鬆的切換多個不同的 .env 檔而不需要修改文件。

### 結合 Jupyter

https://docs.astral.sh/uv/guides/integration/jupyter/

筆者患有 Jupyter 設定障礙，每次設定都覺得異常痛苦所以很少用他，但是 uv 已經整合好了完全沒有這個問題，不用再去網路上看過時的教學除錯，只需要一句 `uv run --with jupyter jupyter lab` 就完成，官方文檔中有更詳細的教學說明。

### 設定預設群組

使用 `uv sync` 預設同步生產套件和 dev 套件這兩類套件，預設同步的套件可以在 pyproject.toml 設定 default-groups 修改同步的目標。

```toml
# 設定 uv sync 同步時除了 dev 也同步 foo 群組
[tool.uv]
default-groups = ["dev", "foo"]
```

## 🔥 pyproject.toml 範例 🔥{#pyproject-toml-example}

既然 uv 的一站式體驗這麼好，那本文也提供一站式體驗，連 `pyproject.toml` 基礎範例都放上來提供參考，一個簡易的 `pyproject.toml` 內容應該如下所示

<details>

<summary>pyproject.toml</summary>

```toml
[project]
name = "your-project-name"  # 必填
version = "0.1.0"  # 必填
description = "project description"
authors = [{ name = "your-name", email = "your-email@example.com" }]
maintainers = [{name = "your-name", email = "your-email@example.com"}]
urls.repository = "https://your-repo-url"
urls.homepage = "https://your-project-site"
license = {text = "MIT License"}  # 也可以用檔案 license = { file = "LICENSE" }
readme = "README.md"
# 發布到 PyPI 的關鍵字和搜尋分類，可選
keywords = [
    "xxx",
    "xxx-toolkit",
]
classifiers = [
    "Topic :: Multimedia",
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: Implementation",
    "Programming Language :: Python :: Implementation :: CPython",
    "License :: OSI Approved :: MIT License",
]

# 宣告依賴關係
requires-python = ">=3.10"
dependencies = ["beautifulsoup4>=4.12.3", "requests>=2.32.3"]

# 開發群組
[dependency-groups]
dev = [
    "mypy>=1.13.0",
    "ruff>=0.7.1",
    "pre-commit>=4.0.0",
    "pytest>=8.3.3",
    "isort>=5.13.2",
]

# # 可選群組
# # 使用 uv build 構建完自己的包後使用這個指令安裝
# # uv pip install "dist/your_project_name-0.1.0-py3-none-any.whl[bs4-sucks]"
# # 再使用 uv pip list 就可以看到 lxml 被成功安裝
# [project.optional-dependencies]
# bs4-sucks = [
#     "lxml",
# ]

# # 如果需要打包套件就使用這些
# [build-system]
# requires = ["hatchling"]
# build-backend = "hatchling.build"

# [tool.hatch.build.targets.wheel]
# packages = ["src/package_name"]

# # 幫 cli 套件設定入口點
# # 請注意，除了 `project.scripts` 外 `build-system` 和 `tool.hatch.build.targets.wheel` 都要一起設定才能啟用
# # https://docs.astral.sh/uv/concepts/projects/config/#entry-points
# [project.scripts]
# my-cli-command = "package_name:main_function"
```

</details>

現在要開始展示 uv 的方便了，我們只需要建立一個空的專案資料夾，貼上後此文件使用 `uv sync` 就完成 Python 版本下載和設定 + 虛擬環境建立 + 套件安裝：

```sh
# 一行完成下載和設定 Python、建立虛擬環境、安裝套件、建立uv.lock
uv sync

# 一般來說我們會 clone 專案，所以我們甚至連 `uv sync` 都可以跳過，直接使用 `uv run` 也會自動安裝
uv run <專案入口指令>
```

但是如果專案使用 Poetry 我們就要用 pyenv + Poetry 組合才能完成同樣工作，還需要使用這麼繁瑣的指令才能完成一樣的任務，而且 Poetry 的 "etry" 有夠難打每次敲快一點就打錯。

```sh
# 下載和設定版本
pyenv install 3.11.5
pyenv local 3.11.5

# 確認 Python 版本
python --version
poetry config virtualenvs.in-project true
poetry env use python3.11.5
# 或者使用 poetry env use $(pyenv which python)

# 安裝套件，啟動虛擬環境並且檢查
poetry install
poetry shell
poetry show
```

## 🔥 Github Workflow 範例 🔥

作業要抄就要抄的徹底，這是包含多作業系統 + 多 Python 版本的 Github Workflow 檔案，用於在 push/pull requests 時自動執行 pytest，實際測試過沒問題，也是複製貼上就能用：

```yaml
name: Test
on: [push, pull_request]
permissions:
  contents: read

env:
  DAY_STATUS: "GOOD"

jobs:
  tests:
    name: Quick Test
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.13']  # uv 看不懂 3.x 代表最新版，所以要手動更新

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      # 或是使用原本的 setup-python，接受 3.x 語法
      # - name: Set up Python
      #   uses: actions/setup-python@v4
      #   with:
      #     python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: uv.lock
          python-version: ${{ matrix.python-version }}

      - name: Test with python ${{ matrix.python-version }} on ${{ matrix.os }}
        run: uv run --frozen pytest

      - name: Environment variable example
        if: runner.os == 'Linux' || runner.os == 'macOS'
        run: uv run echo Today is a $DAY_STATUS day
```

## 結語

文章太長了所以拆成幾篇，本文從日常使用，到 pyproject.toml/.pre-commit-config.yaml 抄作業涵蓋八成的日常使用指令，下篇在[這裡](uv-project-manager-3)，會包含套件發佈的和其他細節指令的教學。

整體下來最心動的就是 `uv run` 的強大功能，以及不需要 pyenv/pipx，也不用記 Poetry 有關 Python 解釋器的指令，這麼多功能全部都濃縮在 uv 一個套件中，加上執行速度快，更新很勤勞（2024/11 看下來每天都有 10 個 commit，嚇死人），社群狀態很健康 (競爭對手 [PDM is a one-man-show, like Hatch](https://chriswarrick.com/blog/2024/01/15/python-packaging-one-year-later/))，一個工具完整取代 pyenv/pipx，幾乎包含 Poetry 的所有功能，速度又快，難怪竄升速度這麼誇張。
