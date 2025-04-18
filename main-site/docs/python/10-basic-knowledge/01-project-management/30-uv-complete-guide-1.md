---
title: uv Python 教學，最佳專案管理工具（上）
sidebar_label: uv 專案管理工具（上）
slug: /uv-project-manager-1
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

# Python uv 教學，最佳專案管理工具（上）

本系列文章介紹 [uv](https://github.com/astral-sh/uv) 的日常操作指令，從安裝到發布套件都包含在內，還有抄作業環節，直接複製貼上就能用，適合沒寫過 pyproject.toml 的人快速上手，本文只講到如何初始化環境，專案管理本身請見[下一篇教學](uv-project-manager-2)。

如果不清楚自己是否該選擇 uv 請看我寫的 [Python 專案管理工具比較](./best-python-project-manager)。

## 簡介

以一句話形容 uv，那就是完整且高效的一站式體驗。uv 是 2024/2 才首發的新工具，簡單摘要幾個特點：

1. 由 rust 撰寫，標榜快速，比 Poetry 快十倍以上
2. 使用 PubGrub 演算法[解析套件依賴關係](https://docs.astral.sh/uv/reference/resolver-internals/)
3. **<u>取代 pyenv</u>**：支援 Python 版本管理[^global]
4. **<u>完美取代 pip/pip-tools</u>**：支援 lockfile 鎖定套件版本
5. **<u>完美取代 pipx</u>**：支援全域套件安裝
6. 發展快速，發布不到一年已經有 26k 星星

把特點 2\~4 加起來就是我們的最終目標了，有更好的套件解析演算法，不只支援 lockfile 管理套件，也支援 Python 版本管理，還沒有 pipenv 速度緩慢且更新停滯的問題，是目前專案管理工具的首選，和原本的首選 Poetry 互相比較，uv 內建的 Python 版本管理非常方便，不再需要 pyenv 多記一套指令；支援安裝全域套件，完美取代 pipx 管理全域套件；做了 pip 的接口方便用戶輕鬆上手，除此之外還有最重要的 `uv run` 功能提供了[非常優秀的開發便利性](uv-project-manager-2#uv-run)，再加上[超快的安裝和解析速度](https://astral.sh/blog/uv-unified-python-packaging)錦上添花，筆者認為目前專案管理工具首選就是他了。

為何選擇 uv？我會說：「一個工具完整取代 pyenv/pipx，幾乎包含 Poetry 的所有功能，速度又快」，這麼多優點是我可以一次擁有的嗎，太夢幻了吧。

身為新穎又備受矚目的套件，目前的更新速度非常快，[兩個月就把問題解決了](https://www.loopwerk.io/articles/2024/python-uv-revisited/)。

> 更新：發展不只是快而是超快，才一個禮拜過去他又多了一千個星星，筆者文章都還沒校完稿，放上圖片讓大家看到底有多粗暴，有人直接飛天了

> 再度更新：2024/12/12 星星數成功超越 Poetry，確實是最受歡迎的環境管理套件了

![Star History Chart](https://api.star-history.com/svg?repos=python-poetry/poetry,astral-sh/uv,pypa/pipenv,pypa/hatch,pdm-project/pdm,conda/conda,pyenv/pyenv-virtualenv&type=Date)

<br/>
<br/>

[^global]: 只剩下等效於 `pyenv global` 的設定全局 Python 功能<s>還不支援但[已經在規劃中](https://github.com/astral-sh/uv/issues/6265)</s>已經放進 [preview 版本](https://github.com/astral-sh/uv/releases/tag/0.5.6)中，加上 `--preview --default` 參數即可使用，目前實測還很早期實測完全不能用。

## TL;DR

如果沒有要發布套件也沒有複雜的開發管理，只使用日常七騎士就可以使用 uv 完美取代舊有工具，使用這七個指令即使不懂 pyproject.toml 也可輕鬆使用 uv，他會變成一個簡單、方便又超快的 venv + pip + pipx + pyenv 的全能工具。

```sh
# 初始化工作區
uv init --python 3.10

# 新增套件
# 首次執行 uv add 時會自動執行 uv venv 以建立虛擬環境
# uv venv 則會根據工作區設定自動下載 Python
uv add <pkg>

# 移除套件
uv remove <pkg>

# 檢查套件
uv pip list

# 更新 lock 檔案的套件版本，更新指定套件或全部套件
uv lock -P <pkg>
uv lock -U

# 根據 uv.lock 同步虛擬環境的套件
uv sync

# 執行程式
uv run main.py
```

<details>
<summary>pip 的接口</summary>

uv add/remove 會寫入到 pyproject.toml，如果無論如何也不想使用 pyproject.toml，`uv pip` 提供了對應以往 pip 的接口，但是既然都用 uv 了應該用 add/remove 方式比較好，所以把這段放到折疊頁面中。

```sh
# 安裝
uv pip install

# 從文件安裝
uv pip install -r requirements.txt

# 移除
uv pip uninstall

# 寫出版本資訊
uv pip freeze > requirements.txt

# 更新全部版本@Unix
uv pip freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 uv pip install -U
```

:::danger
注意：<u>uv pip 不使用 pip，只是呼叫方式類似的 API 接口</u>！
:::

</details>

## 初始化環境

### 安裝 uv

https://docs.astral.sh/uv/getting-started/installation/

使用以下指令進行獨立安裝程式，其餘安裝方式請自行閱讀文檔。

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

（可選）幫終端機加上指令自動補全，執行這些指令完成指令補全設定

```sh
# macOS and Linux 用戶檢查自己是哪個 shell 
ps -p $$

# 根據對應 shell 選擇指令
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc

# Windows 只有 PowerShell 支援補全，CMD 不行
if (!(Test-Path -Path $PROFILE)) {
  New-Item -ItemType File -Path $PROFILE -Force
}
Add-Content -Path $PROFILE -Value '(& uv generate-shell-completion powershell) | Out-String | Invoke-Expression'
```

使用方式是指令打到一半按下 <kbd>Tab</kbd> 即可自動補全。

:::info 廣告時間
如果這是你第一次優化 shell 相關，可以參考[我的 Zsh 設定](https://github.com/ZhenShuo2021/dotfiles)，支援 macOS 和 Ubuntu，特色是極簡外觀、功能齊全而且啟動速度超快，基本上已經到速度極限不會有人的啟動速度比我的快。
:::

### 移除 uv

移除方式請見官方文檔的[指令教學](https://docs.astral.sh/uv/getting-started/installation/#uninstallation)。

### 設定 Python 版本

https://docs.astral.sh/uv/concepts/python-versions/  
https://docs.astral.sh/uv/guides/install-python/  

首先從 Python 版本管理開始說明，其實這個指令很少用到，因為 uv 可以聰明的自動完成 Python 安裝。

``` sh
# 安裝/移除指定版本
uv python install 3.12
un python uninstall 3.12

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

https://docs.astral.sh/uv/concepts/projects/init/

設定好 Python 版本後就是初始化專案，使用 `app` 參數設定專案名稱，使用 `build-backend` 參數設定專案的構建後端，也可以不輸入任何參數使用預設值初始化專案。

```sh
uv init project-name [--package | --lib] [--python <version>]
```

這會建立一個基礎專案。如果要建立一個 CLI APP，那麼 `--package` 是你的好幫手，會建立一個 src layout 的 CLI 專案並且幫你設定好構建後端、函式入口點；加上 `--build-backend <name>` 修改構建後端，預設是 hatch。

:::tip

- 如果你的專案需要使用 rust/C/C++ 等外部函式庫，請參照[官方文檔](https://docs.astral.sh/uv/concepts/projects/init/#projects-with-extension-modules)說明。
- uv 不支援從 Conda 庫安裝包，詳情請見 [#1703](https://github.com/astral-sh/uv/issues/1703)，如果有這個需求請改用 [pixi](best-python-project-manager#pixi)。

:::

### 建立虛擬環境

https://docs.astral.sh/uv/pip/environments/

接下來是建立虛擬環境，名稱和 Python 版本都是可選參數非必要，尤其是如果使用了 `uv init` 建立的專案目錄就不用再輸入，因為所有設定都在 `uv init` 完成了。

```sh
uv venv [name] [--python <version>]
```

直接使用這個指令的情境通常是隨手的專案，懶的管 pyproject.toml 這些設定，用他加上 `uv pip` 這等同於古早時代的專案管理，但是速度超快。

:::tip 虛擬環境
`source .venv/bin/activate` 代表進入虛擬環境，之前看到有人發 issue 問到底該不該進入虛擬環境，我現在找不到這個 issue 但是記得答案是 no，理由是 `uv run` 就可以直接使用，也免去在虛擬環境中切換的麻煩。我的看法是不用進入，且進入了也不影響，如果有設定 `[project.scripts]` 作為腳本入口，兩者差別在於需不需要在 `my-cli-command` 前面加上 `uv run`。

本文有提供[範例](uv-project-manager-2#pyproject-toml-example)介紹如何設定腳本入口。
:::

## 小結

避免閱讀負擔把文章拆成幾篇，接著請看[中篇](uv-project-manager-2)，是重點中的重點。
