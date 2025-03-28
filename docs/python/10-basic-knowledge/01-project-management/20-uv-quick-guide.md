---
title: uv 快速手冊
sidebar_label: uv 快速手冊
slug: /uv-quick-guide
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
  date: 2024-12-24T21:53:30+08:00
  author: zsl0621
first_publish:
  date: 2024-12-24T21:53:30+08:00
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# uv 快速手冊

覺得 [Python uv 教學](./uv-project-manager-1/) 太細節，所以本文如題是一個快速手冊，著重長話短說，目標是搞懂工具的大方向。如果不知道是否應該選擇 uv 請看[專案管理工具比較](./best-python-project-manager)。

## uv vs Poetry

Poetry 對比 uv 在功能上則更為相似，hatch 因為功能特殊所以沒得比較（沒有 lock 檔案，沒有內建虛擬環境管理，主要用於跨平台和多環境）。

實際使用上 uv 的 [`uv run` 功能非常方便](/python/uv-project-manager-2#uv-run)，可以輕鬆的搭配不同的 Python 版本或套件測試，你可以臨時啟用、關閉指定套件，而且 uv 支援 PEP 723，所以你甚至可以在完全沒有虛擬環境和專案的情況下執行腳本，而且支援安裝全域套件（取代 pipx），支援管理 Python 版本（取代 pyenv），這些事情 Poetry 全都做不到。

在這個部落格中寫了五篇關於 uv 的[文章](https://www.loopwerk.io/articles/tag/uv/)，我們可以看到 uv 的發展之快以及齊全的功能。該作者原本使用 poetry，五篇文章一路從 uv 還不夠好寫到全面遷移 uv，並且在他自己的 dotfiles 裡面[把 pyenv 和 poetry 都移除了](https://github.com/search?q=repo%3Akevinrenskers%2Fdotfiles+uv&type=commits)，可見 uv 有多強大。

雖然都在稱讚 uv 方便，但是兩者的選擇我認為應該取決於你的使用場景，需要更成熟穩定且經過多數用戶驗證的工具選擇 Poetry，想擁有一站式的良好使用體驗，並且不在意穩定版本的選擇 uv。

## uv pip/uv add 差別

:::tip TL;DR

使用 `uv add` 作為平時安裝套件的指令，`uv pip` 則是臨時測試套件的安裝指令。

:::

uv add 會把套件紀錄在 pyproject.toml 中，而 uv pip 僅是讓以往 pip 用戶方便使用的接口，實際上 uv 並不使用 pip 安裝套件，官方文檔也表示越少用的 pip 指令其執行結果和原本的 pip 差異越大。除此之外<u>**所有 uv 的功能都只能用於使用 uv add 安裝的套件**</u>，網路上單純把 uv 作為取代 pip 的文章完全浪費了 uv 這個強大的工具。

## 安裝和開發

使用 `uv add` 安裝套件並且紀錄到 pyproject.toml，`uv remove` 移除。除了一般的套件安裝以外，還有開發群組 `uv add --dev` 以及套件群組 `uv add --group my-group my-package` 功能，開發套件是群組的一種。uv 方便之處就在於能輕鬆的切換群組。

比如說我們想在「只使用某種套件」的環境下執行腳本，只需要使用 `uv run --group` 或者 `uv run --only-group` 而不用開第二個虛擬環境或是手動安裝；使用 `uv run --with <pkg>` 則可以在這次執行時臨時加上該套件，不會真正安裝到虛擬環境中。uv 預設同步的套件包含開發群組套件，在 pyproject.toml 中設定 `default-groups` 可以修改預設使用哪些群組。這是 uv 非常方便的功能，也是為何筆者會說網路上的文章沒有完全使用好這個工具。

接下來，當我們使用 uv pip 安裝了很多測試套件後，可以使用 `uv sync` 指令還原到 pyproject.toml 中設定的套件，這裡翻譯[uv sync 官方文檔](https://docs.astral.sh/uv/reference/cli/#uv-sync)：

:::tip uv sync

更新專案環境。

根據 uv.lock 同步依賴。預設執行精確 (exact) 同步：uv 會移除未被聲明為專案依賴 (pyproject.toml 中設定) 的套件。使用 `--inexact` 可保留多餘的套件。注意如果多餘的套件會和專案依賴發生衝突則仍然會被移除。此外，若使用 `--no-build-isolation`，uv 不會移除多餘套件，以避免刪除可能的建置依賴。

若專案還沒有建立虛擬環境會自動建立。

除非使用 `--locked` 或 `--frozen` ，否則在同步前會重新鎖定專案。

uv 會在當前目錄或其父目錄中搜尋專案。若無法找到專案，uv 將報錯並退出。

注意，從鎖定檔安裝時，uv 不會針對已撤回的套件版本提供警告。
:::

## 更新套件

使用 `uv sync -U` 更新所有套件，使用 `uv sync -P <pkg>` 更新指定套件。此指令同時也會更新 lockfile，而 uv lock 指令同樣也有 -U/-P 參數，差別是其只會更新 lockfile。

如果要更新 pyproject.toml 裡面的套件版本請參考 [這個 issue](https://github.com/astral-sh/uv/issues/6794) 裡面的腳本完成，看起來官方沒有要馬上解決這個問題。

## 專案容量

uv 大量使用快取功能也是速度快的一個原因，在 uv 中，所有套件都會被集中放置在同一目錄 (~/.cache/uv)，再使用硬連結把套件連結到每個專案的虛擬環境資料夾中，所以就算專案再多也不會重複下載套件，使用 `uv cache clean` 可以清除所有快取檔案。

把硬連結 (hard link) 和軟連結 (symlink) 一起解釋，軟連結就是以前的捷徑功能，把原始檔案刪除後捷徑也沒用了，硬連結則是兩個完全相同的檔案指向同一實際儲存位置，只要任一檔案還在，實際儲存空間就不會消失。

## Workspace

原本有考慮這段要不要放在 *專案管理工具比較* 裡面，但是那篇已經太多內容了而且筆者自己也沒用過，於是放在這裡。

這是從 rust 來的概念，用於在程式越來越龐大時能把複雜的程式拆分成小項目 (library) 方便管理，每個項目有自己的 pyproject.toml，但是工作區共用一個 lock 檔案，詳情請參考[文檔](https://docs.astral.sh/uv/concepts/projects/workspaces/)。

## 全域 Python

作為取代 pyenv 的最後一步，全域 Python 截至 v0.5.11 都還不能用，請參考[這個 issue](https://github.com/astral-sh/uv/issues/6265)。

## 結束

本文僅作為一個快速手冊讓初學者快速掌握使用方式，詳細指令請參考[Python uv 教學](./uv-project-manager-1)。
