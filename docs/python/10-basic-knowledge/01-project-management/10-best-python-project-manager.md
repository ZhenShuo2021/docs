---
title: Python 專案管理工具全面比較
sidebar_label: 選擇正確的專案管理工具
slug: /best-python-project-manager
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


# Python 專案管理工具全面比較 - uv, Poetry, Pixi, Hatch, PDM, Conda, Mamba, Pipenv, venv, virtualenv

你知道 Python 有超過 10 個專案管理工具嗎？每個工具的出現都是為了解決前一個工具的痛點，然而網路教學文章總是劈哩啪啦說一堆指令，卻沒有回答我心中的問題：<u>**我該選擇這個工具嗎**</u>？我們應該要知道的是如何選擇，而不是照著指令複製貼上等出問題才知道到底適不適合。

本文是全中文唯一一篇完整整理 Python 專案管理工具的文章，介紹內容包含

- uv
- Poetry
- Pixi
- Hatch
- PDM
- Conda, Mamba, miniconda, miniforge
- Pipenv
- pip-tools
- venv, virtualenv, virtualenvwrapper, pyenv-virtualenvwrapper
- Twine, Flit, tox, nox

沒錯，[Python 的專案管理就是個災難](https://chriswarrick.com/blog/2023/01/15/how-to-improve-python-packaging/)，有高達 19 個工具，本文會詳細介紹讓你知道如何選擇。

## TL;DR

文章太長，簡而言之只有這三種選擇

1. 小專案用 `uv`，使用方式簡單同時速度飛快
2. 稍微大一點的專案**一律在 Poetry 和 uv 之間二選一**，核心差別是 Poetry 已經是穩定版，uv 還在開發階段
3. 科學開發需要 C 語言或 R 語言擴展，建議使用 Pixi 或 Mamba 而不是常見的 Conda，原因是 Conda 現在和未來都不會支援 pyproject.toml，[#12462](https://github.com/conda/conda/issues/12462) 已經存在兩年都沒有提交

對於依賴解析這個課題，[Pixi 已經和 uv 合作開發](https://github.com/astral-sh/uv/issues/1572#issuecomment-1957318567)，uv 的[解析演算法](https://docs.astral.sh/uv/reference/resolver-internals/)原本就已經非常好，聯手後可以把他們視為 Python 世界最先進的依賴解析工具，再來是 Poetry，再來就是 others。

對於 Python 科學開發這個課題，由於需要 Python 以外的庫，所以大部分人一直都使用 Conda，不過 Pixi 寫了一篇文章說明[從 Conda 切換到 Pixi 的七個理由](https://prefix.dev/blog/pixi_a_fast_conda_alternative#reason-7-native-lockfiles)，我認為最重要的幾個原因是完整兼容 Conda 庫、PyPI 兼容、有 task 功能以及原生<u>**支援鎖定文件**</u>。

:::info
科學運算的開發者可能不熟悉鎖定文件的概念。鎖定文件（Lock File）**精確記錄**專案所有依賴的確切版本、依賴關係和校驗和，確保不同開發環境中能夠**重現**完全相同的環境，避免 *It works on my machine, but not yours* 問題。

只要有鎖定文件就能復現完全一樣的環境，想像一下數據要重新驗證的時候，都已經在死線了結果還在處理版本問題、環境裝不出來、模擬結果不一樣...
:::

比較表格如下，只有新工具在裡面，因為已經沒有任何理由再去用舊工具了。

<table
  style={{
    width: "100%",
    borderCollapse: "collapse",
    textAlign: "center",
  }}
>
  <thead>
    <tr>
      <td style={{ width: "5%" }}></td>
      <td style={{ width: "5%" }}>Python<br />版本管理</td>
      <td style={{ width: "5%" }}>鎖定文件</td>
      <td style={{ width: "7%" }}>速度</td>
      <td style={{ width: "16.66%" }}>優點</td>
      <td style={{ width: "16.66%" }}>缺點</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/python-poetry/poetry">Poetry</a></td>
      <td>❌</td>
      <td>✅</td>
      <td>快</td>
      <td style={{ textAlign: "left" }}>成熟穩定，高度成熟的依賴解析器，良好社群支援，完整功能</td>
      <td style={{ textAlign: "left" }}>需配合 pyenv 管理 Python 版本，沒有其餘方便功能</td>
    </tr>
    <tr>
      <td><a href="https://github.com/astral-sh/uv">uv</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>非常快</td>
      <td style={{ textAlign: "left" }}>速度極快，使用 PubGrub 演算法解析依賴，一站式體驗，取代多個工具 (pyenv/pip/pipx/venv)，<code>uv run</code> 功能強大</td>
      <td style={{ textAlign: "left" }}>開發中但是問題相對較少</td>
    </tr>
    <tr>
      <td><a href="https://github.com/prefix-dev/pixi/">Pixi</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>非常快</td>
      <td style={{ textAlign: "left" }}>支援 Conda 套件庫，使用 uv 的依賴解析功能，可視為支援 Conda 庫的 uv，<code>pixi task</code> 功能強大</td>
      <td style={{ textAlign: "left" }}>開發中但是問題相對更多。知名度低，資源少，需要自己看文檔</td>
    </tr>
    <tr>
      <td><a href="https://github.com/pypa/hatch">Hatch</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>中等</td>
      <td style={{ textAlign: "left" }}>遵循 PEP 標準、跨平台一致性、可設定多環境</td>
      <td style={{ textAlign: "left" }}>不只用戶少，專案僅一人維護，不支援 lockfile</td>
    </tr>
    <tr>
      <td><a href="https://github.com/pdm-project/pdm">PDM</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>慢</td>
      <td style={{ textAlign: "left" }}>遵循 PEP 標準</td>
      <td style={{ textAlign: "left" }}>不只用戶少，專案僅一人維護，速度慢，相較其他工具無顯著優點</td>
    </tr>
  </tbody>
</table>

由於 Pixi 太新，如果遇到無法解決的問題真的需要回到 Conda 體系，請用以 C++ 重寫 Conda 的 [Mamba](https://github.com/mamba-org/mamba) 而不是 Conda，兩者功能相同但是 Mamba 速度更快。

<br/>

:::tip
筆者個人推薦 uv，懶得看文章可以直接看我寫的 [uv 使用教學](./uv-quick-guide)。
:::

## 工具目的

要知道怎麼選擇前得先清楚工具本身的目的，專案管理工具主要目的是解決依賴管理和環境隔離的問題，核心功能有

1. <u>**基礎功能**</u>：為每個專案建立獨立的 Python 環境，避免不同專案間的依賴不會互相干擾。
2. <u>**依賴套件解析**</u>：解析不同套件的依賴，這是非常重要的問題，除了避免依賴衝突，也增加可復現性。
3. <u>**pyproject.toml 支援**</u>：pyproject.toml 是 PEP 518/621 提出的 Python 專案標準配置文件，取代傳統的 setup.py，作為專案不同工具間的統一設定標準，提高專案的<u>**可復現性**</u>。
4. <u>**版本鎖定機制**</u>：支援鎖文件 (lockfile) 鎖定套件精確版本，確保開發、測試和生產環境一致性，增加<u>**可復現性**</u>，不支援此文件基本上就直接淘汰。
5. Python 版本管理：能管理和下載 Python 版本，不再需要額外工具管理不同的 Python 版本
6. 套件構建
7. 套件發佈

這些功能缺一不可，尤其是一到四，一是最基本的功能，二在不同工具之間就會開始有差異，三和四淘汰掉很多老舊工具，五往後是有的話最好，如果支援就不用再使用不同工具完成這些任務。

不會寫 pyproject.toml 也別擔心，我在 [uv 教學](/python/uv-project-manager-1) 中有完整介紹。

## 選擇要點

上述七項是作為 Python 專案管理器的主要目的，了解目的之後接下來說明如何比較，我認為以下是最重要的幾個項目：

1. <u>**基本可用**</u>，連基本功能都無法滿足的直接淘汰，例如不支援 pyproject.toml，或是依賴解析演算法很糟糕
2. <u>**使用方便**</u>，指令簡潔明瞭且符合直覺，不需要記憶過多特殊語法或繁複操作步驟
3. <u>**社群活躍**</u>，有活躍的社群支援代表遇到問題比較容易找到解決方案，以及工具本身有持續更新維護
4. <u>**速度可接受**</u>，速度慢到不可接受就算功能再齊全都不會想用他，速度問題在 CI/CD 階段尤其重要
5. <u>**套件生態**</u>，整合 CI/CD，支援 C/Rust 等擴展，文檔完整性等等...

本文不會討論到這麼詳細但還是把這些要點列出來（原因是很多工具連基本可用的條件都達不到）。

其實開發者這麼多很難有大家都是笨蛋只有我最聰明要跟別人用不一樣的的情況發生，所以選擇前我們觀察 Github 星星數就大概知道應該要選哪個工具（扣掉 Pipenv，他開空頭支票騙人），以下排序依照發布時間，星星數統計於撰文當下 (2024/11/19)，首先是時代的眼淚區：

1. [virtualenv](https://github.com/pypa/virtualenv) (2007, 4.8k stars)
2. [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)(2012, 6.4k stars)
3. [Conda](https://github.com/conda/conda) (2012, 6.5k stars)
4. [Pipenv](https://github.com/pypa/pipenv) (2017, 24.9k stars)

![Star History Chart](https://api.star-history.com/svg?repos=pypa/pipenv,pypa/virtualenv,conda/conda,pyenv/pyenv-virtualenv&type=Date)

接著是新時代的挑戰者（外來種入侵等級）

1. [Poetry](https://github.com/python-poetry/poetry) (2018, 31.8k stars)
2. [PDM](https://github.com/pdm-project/pdm) (2019, 8k stars)
3. [Hatch](https://github.com/pypa/hatch) (2022, 6.1k stars)
4. [Pixi](https://github.com/prefix-dev/pixi) (2024, 3.1k stars)
5. [uv](https://github.com/astral-sh/uv) (2024, 26.7k stars)

![Star History Chart](https://api.star-history.com/svg?repos=python-poetry/poetry,astral-sh/uv,prefix-dev/pixi,pdm-project/pdm,pypa/hatch&type=Date)

## 詳細比較：時代的眼淚組

<details>

<summary>時代的眼淚</summary>

時代的眼淚組連身為套件管理器的基本功能都無法滿足，文章幾經更新後覺得這段沒必要看了，反正又不會用他們。當然如果你是初學者想要搞清楚這些工具在做甚麼、有什麼差異、前因後果先來後到也可以閱讀本段落，本段落依照發表時間由舊到新介紹。

### venv/virtualenv

venv 是內建於 Python 的管理工具，是基於 virtualenv 的管理工具，差別是 virtualenv 需要額外安裝，他們本質上是相同的，差別只有 venv 不支援 Python2，於是我們知道**完全沒有任何理由使用 virtualenv**。這兩個只能用於建立虛擬環境，沒有任何依賴解析管理功能，一般是使用 pip 完成依賴解析管理。

### Pyenv

專注於 Python 版本管理的套件。

pyenv 專注於管理多個不同 Python 版本以及負責切換版本，不包含任何額外功能，需要和其他管理套件協同工作：他只是拼圖的其中一塊而已，但是把自己的這塊做好。pyenv 不支援 Windows，需要使用 [pyenv-win](https://github.com/pyenv-win/pyenv-win)。

### pyenv-virtualenv

pyenv 的插件，結合 virtualenv 後變成既有 Python 版本管理又有套件管理功能的完整工具，從這裡開始才真正進入戰場，涵蓋環境管理、Python 版本管理以及依賴解析管理（還是使用 pip 解析，無優化）。

### Conda

這段可以總結成「數學算超快但是答案全錯」，Conda 的情況就是這個笑話。

Conda 筆者用的可多了，使用體驗就是兩個字「快逃」。Conda 整合 Python 版本管理、虛擬環境管理和套件依賴解析，完美整合自家贊助的 Spyder IDE，安裝完 Spyder 後能直接使用虛擬環境不需任何設定，還把套件打包二進制安裝速度更快。聽起來很棒，但 Conda 有一個致命問題是套件怎麼裝怎麼崩潰，即使是全新環境也一樣。

Conda 使用自己的套件依賴解析器而不是 pip，所以混用會出問題，你說為甚麼要混用？因為內建的永遠出現衝突問題，就拿筆者最常使用的套件 Tensorflow/Matplotlib/Numba 三個來說，三個都依賴 Numpy，三個都是知名套件，但是三個一起裝就是 boom 爆炸，筆者當時甚至研究出先使用 Conda 安裝 A，再使用 pip 安裝 B，最後再用 Conda 安裝 C 才能成功安裝，任何改變都會出錯。平常最好祈禱自己不要手癢想更新，更新就是套件再重裝一次。

Conda 在 2023 年把內部的解析演算法改成 [libmamba](https://github.com/conda/conda-libmamba-solver)，特色只說了速度更快沒有提到正確性。當一個套件管理器連套件都裝不成功的時候，我才不關心你快不快。

Conda 有很多衍生版本如 miniconda, Mamba, Miniforge, Anaconda Navigator 等等，筆者沒有深入研究細節，畢竟連最有名的 Conda 功能都這麼糟糕了我完全不指望衍生版本能給我多大驚喜，用過 miniconda 沒有感受到任何差異，這幾個衍生版本筆者只知道分別是輕量化的、C++ 重寫的、C++ 重寫加上輕量化的、GUI 版本的 Conda。

### pipenv

顧名思義，是整合了 pip 和虛擬環境管理的工具，目的是簡化依賴管理與環境設置。總共改善了這些問題：

- 引入 Pipfile 和 Pipfile.lock 解析依賴與鎖定
- 開發/生產環境分離
- 使用自行開發的依賴解析器

pipenv 整合環境管理和依賴管理，使用 pipfile 和 pipfile.lock 管理，並且支援 `--dev` 選項隔離開發用套件，如此豐富的功能整合看起來很美好，但是實際上[不美好🚨](https://note.koko.guru/posts/using-poetry-manage-python-package-environments)。本工具筆者並未使用過所以上網做了一下功課，看到此篇文章：[Pipenv: promises a lot, delivers very little](https://chriswarrick.com/blog/2018/07/17/pipenv-promises-a-lot-delivers-very-little/)，整整四千字的心得大罵 pipenv，罵人能罵到四千字也是不容易，所以我們大概可以知道該如何選擇，其主要問題包含效能、指令操作繁瑣、僅支持特定的工作流程、不支援跨目錄工作，而且更新停滯。

對於一個 2017 年才發布的套件，在 2018 年就被罵不更新，2020 年該文章作者還繼續追蹤但是情況依舊。查看其 [Github release](https://github.com/pypa/pipenv/releases)，整個 2023 從二月之後就沒更新了，如此緩慢的更新不建議使用。

都 2024 了如果還有文章推薦 pipenv 建議送上 uBlacklist 伺候。

### 其他

[virtualenvwrapper](https://github.com/python-virtualenvwrapper/virtualenvwrapper) 則是基於 virtualenv 的工具，把所有環境整合在同一目錄，並且提供更多額外功能，本質上還是 virtualenv，[pyenv-virtualenvwrapper](https://github.com/pyenv/pyenv-virtualenvwrapper) 同理，兩者也都沒更新了，不列入討論。

</details>

## 詳細比較：新時代的挑戰者

新時代的挑戰者 aka 外來種入侵，他們強大的功能打的時代眼淚組毫無招架之力，換句話說，完全沒有理由使用時代眼淚組。

### uv

如果要以要一句話形容 [uv](https://github.com/astral-sh/uv)，那就是完整且高效的一站式體驗，解決了時代眼淚組的所有問題。

uv 是 2024/2 才首發的新工具，簡單摘要幾個特點：

1. 由 rust 撰寫，標榜快速，比 Poetry 快十倍以上
2. 使用 PubGrub 演算法[解析套件依賴關係](https://docs.astral.sh/uv/reference/resolver-internals/)
3. **<u>取代 pyenv</u>**：支援 Python 版本管理[^global]
4. **<u>完美取代 pip/pip-tools</u>**：支援 lockfile 鎖定套件版本
5. **<u>完美取代 pipx</u>**：支援全域套件安裝
6. **<u>強大且方便的 `uv run` 指令</u>**
7. 支援 pyproject.toml 和 lockfile
8. 發展快速，發布不到一年已經有 26k 星星

把特點 2\~4 加起來就是我們的最終目標了，有更好的套件解析演算法，不只支援 lockfile 管理套件，也支援 Python 版本管理，還沒有 pipenv 速度緩慢且更新停滯的問題，是目前虛擬環境管理工具的首選。

為何選擇 uv？我會給出這個結論：「一個工具完整取代 pyenv/pipx，幾乎包含 poetry 的所有功能，更不要說還有強大的 `uv run` 和超快的速度」，這麼多優點是我可以一次擁有的嗎，太夢幻了吧。

[^global]: 只剩下等效於 `pyenv global` 的設定全局 Python 功能還不支援但[已經在規劃中](https://github.com/astral-sh/uv/issues/6265)，加上 `--preview --default` 參數即可使用，目前實測還很早期，連 venv 都不能跑。

:::tip 使用心得

uv 內建的 Python 版本管理非常方便，不再需要 pyenv 多記一套指令，還支援安裝全局套件以取代 pipx，支援使用 `uv build` 和 `uv publish` 功能可以方便的構建和發布套件，還做了 `uv pip` 方便以往的 `pip` 用戶輕鬆上手（網路上那些只知道 uv 速度很快，但是真正實用的功能完全沒用到的文章也是這樣來的），除此之外還有最重要的 `uv run` 功能提供了非常優秀的開發便利性，再加上[超快的安裝和解析速度](https://astral.sh/blog/uv-unified-python-packaging)錦上添花，筆者認為目前虛擬環境管理工具首選就是他了。

當然也不是完美的，有幾個小缺點：

- 使用 Rust 撰寫，Python 開發者不好進行貢獻
- [還不支援 dependabot](https://docs.astral.sh/uv/guides/integration/dependency-bots/#dependabot)
- 太新，連英文都沒有幾篇文章說明如何使用，不過別擔心筆者寫了使用教學，從安裝到發布套件一應俱全。

:::

:::tip 開始使用 uv

- 可以參考筆者寫的 [uv 使用教學](./uv-quick-guide)。  
- 非常優秀的開發便利性？請見 uv 使用教學的[這個段落](uv-project-manager-2/#uv-run)。  

:::

### Poetry

雖然文章開頭說「新工具的出現就是要解決舊有工具的痛點」，但前面好像都沒什麼提到，因為很多工具功能缺陷連比都沒得比。[Poetry](https://github.com/python-poetry/poetry) 有以下幾個特色：

1. 支援 pyproject.toml 和 lockfile
2. 內建虛擬環境管理
3. 使用自己的依賴解析器
4. 內建套件發布工具
5. 已經是穩定版本，競爭對手 uv 還是開發版本
6. 但是不支援管理 Python 版本

Poetry 最大的優勢是支援 pyproject.toml，可以設定從開發到發布的所有項目，使用自行開發的依賴解析器，支援 poetry.lock 完成依賴鎖定管理，背景使用 pip 讓兼容過往工具，看起來很美好，而且真的很美好。對比 pipenv 有更快的套件下載和更簡易的指令操作，以及更好的開發環境（整合各種開發工具的設定）；對比 uv 則原生支援套件打包發布。使用體驗上，用戶人數足夠多，相關資訊比 PDM/hatch 更充足，cli 提示做得很不錯，以往筆者用過的工具都需要不斷切換視窗搜尋指令，使用 Poetry 可以明顯感受到頻率降低很多，唯一美中不足的是不包含 Python 版本管理，需要結合 pyenv 使用。

> Poetry 不支援 PEP 621 的問題已計畫將於 [2.0 版本](https://github.com/orgs/python-poetry/discussions/5833)開始支援。

Poetry 的使用指令可以觀看這兩篇文章：簡短的 [\[note\] Python Poetry](https://pjchender.dev/python/note-python-poetry/) 和更豐富的 [Pyenv + Poetry 相關指令](https://hackmd.io/@OpenAIDocuments/ByIK7hTEa)，新手用戶不用管太多只要會 `poetry init` + `poetry add` 就可以了。

:::tip 和 uv 比較
Poetry 對比 uv 在功能上則更為相似，hatch 因為功能特殊所以沒得比較（沒有 lock 檔案，沒有內建虛擬環境管理，主要用於跨平台和多環境）。

實際使用上 uv 的 [`uv run` 功能非常方便](uv-project-manager-2#uv-run)，可以輕鬆的搭配不同的 Python 版本或套件測試，你可以臨時啟用、關閉指定套件，而且 uv 支援 PEP 723，所以你甚至可以在完全沒有虛擬環境和專案的情況下執行腳本，而且支援安裝全域套件（取代 pipx），支援管理 Python 版本（取代 pyenv），這些事情 Poetry 全都做不到。

在這個部落格中寫了五篇關於 uv 的[文章](https://www.loopwerk.io/articles/tag/uv/)，我們可以看到 uv 的發展之快以及齊全的功能。該作者原本使用 poetry，五篇文章一路從 uv 還不夠好寫到全面遷移 uv，並且在他自己的 dotfiles 裡面[把 pyenv 和 poetry 都移除了](https://github.com/search?q=repo%3Akevinrenskers%2Fdotfiles+uv&type=commits)，可見 uv 有多強大。

雖然都在稱讚 uv 方便，但是兩者的選擇我認為應該取決於你的使用場景，需要更成熟穩定且經過多數用戶驗證的工具選擇 Poetry，想擁有一站式的良好使用體驗，並且不在意穩定版本的選擇 uv。
:::

### Pixi

[Pixi](https://github.com/prefix-dev/pixi) 做的最正確的事情就是站在巨人的肩膀上，基本上我們可以把他理解為支援 Conda 庫的 uv。

對於科學開發所需要的 C 語言、R 語言套件，他支援[直接使用 Conda 的套件庫](https://prefix.dev/blog/pixi_global)；對於套件版本解析，Pixi 使用現今 Python 世界裡面[最強大的 uv 作為依賴解析工具](https://github.com/prefix-dev/pixi/pull/863)；對於 lockfile 鎖定文件原生支援，連 conda-lock 作者都說 Pixi 就是 Conda 加上鎖文件的[正確實踐結果](https://prefix.dev/blog/pixi_a_fast_conda_alternative#reason-7-native-lockfiles)。

光是上一個段落就已經打爆 Conda 了，完全想不到任何回頭使用 Conda 的理由，除此之外，對於大型專案需要的多平台設定，Pixi 還支援[每個平台獨立設定](https://pixi.sh/latest/features/multi_platform_configuration/)。和 `uv tool` 相同，Pixi 也支援[安裝全域套件](https://pixi.sh/latest/features/global_tools/)，也就是說可取代 pipx，加上和 Conda 一樣可管理 Python 版本，代表不需要使用 pyenv 作為 Python 版本管理工具，這些全都是 Poetry 沒有的功能。

不只如此，我們還可以用 [task 功能](https://pixi.sh/latest/features/advanced_tasks/) 設定執行的指令和參數，我們可以把他當作強化版的 `uv run`，因為 `uv run` 不支援設定任務和參數。不要小看這項功能，假設有三種 scenario 要跑，我們設定 task 以後就不用再打一堆參數選項了，訓練模型或者跑模擬都不需要搜尋指令記錄上鍵按半天。

缺點當然有，資源少是最大問題，基本上必須得翻文檔，另外就是還在開發階段，例如 [Safouane Chergui](https://ericmjl.github.io/blog/2024/8/16/its-time-to-try-out-pixi/) 說的版本問題就需要特別處理，最後一個是我最擔心的問題，Pixi 不如 Conda, Poetry 經過時間的試煉，也沒有像 uv 一樣有足夠龐大的用戶數量和更新頻率，願景很大但是知名度不夠，未來發展令人擔憂。

- [Let's stop dependency hell](https://prefix.dev/blog/launching_pixi)
- [It's time to try out pixi!](https://ericmjl.github.io/blog/2024/8/16/its-time-to-try-out-pixi/)
- [Pixi Global: Declarative Tool Installation](https://prefix.dev/blog/pixi_global)
- [Towards a Vendor-Lock-In-Free conda Experience](https://prefix.dev/blog/towards_a_vendor_lock_in_free_conda_experience)
- [7 Reasons to Switch from Conda to Pixi](https://prefix.dev/blog/pixi_a_fast_conda_alternative#reason-7-native-lockfiles)

### PDM/hatch

由於是開發管理工具所以不只關注環境管理本身，而是更加注重對於 PEP 的支援，但筆者沒這麼厲害所以不知道確切的影響是什麼，以下大致整理一下網路資訊。

- hatch: 專注在開發的工具
  1. 旨在建立跨平台間能具有相同環境和操作的套件（例如解決 Windows 不支援 Pyenv）
  2. <u>**允許一個專案使用多個環境**</u>
  3. 不支援 lock 鎖定套件版本（[未來會支援](https://github.com/pypa/hatch/issues/749)）

- PDM: [基於 PEP 582 而生的套件](https://pdm-project.org/zh-cn/latest/usage/pep582/)
  1. PEP 582 用於免虛擬環境進行 Python 管理，然而此提案最終被拒絕
  2. 不支援 Python 版本管理，速度[比 poetry 慢](https://astral.sh/blog/uv)，社群比 poetry 小，找了老半天好像找不到什麼優點

這兩個工具最大的問題是使用人數少，開發社群更小，小到只有一個人在開發，bug 基本上要看那個人心情好不好想不想解決。

:::note
Chris Warrick: [PDM is a one-man-show, like Hatch.](https://chriswarrick.com/blog/2024/01/15/python-packaging-one-year-later/)
:::

### 其他

- [pip-tools](https://github.com/jazzband/pip-tools) 用戶數太少不列入討論。
- [Twine](https://github.com/pypa/twine) 只支援套件發佈
- [Flit](https://flit.pypa.io/en/stable/) 只支援套件打包和發佈
- [tox](https://tox.wiki/) 和 [nox](https://nox.thea.codes/) 主要用於測試和管理虛擬環境。

## 總結

Poetry/uv 最好，兩者選擇取決於你喜歡更成熟的架構還是需要稍微自己摸索但是更好的整體使用體驗。hatch 很好，但是著重於管理多版本並存問題，套件依賴需要其他工具幫忙，如果專案夠複雜可以選擇，但是需要用到的人不會來看這篇文章。需要使用 C/C++ 等擴展的科學開發請直接選擇 Pixi，Conda 很糟糕，就算真的要用 Conda 生態也是選使用 C++ 重寫的 Mamba，然後別忘了祈禱自己不要遇到依賴解析問題。

總結

1. 一律使用 Poetry 或 uv
2. 科學開發使用 Pixi 或 Mamba
3. 會用到 hatch 的人不會來看這篇文章
4. 沒有其他選項，任何其餘選項都比不過上述工具

本文解決了現有網路文章缺少的選擇分析，透過完整說明功能和套件現況，讀者可以清楚知道自己應該選擇哪個虛擬環境管理套件。

## 參考文章

- [Python Packaging, One Year Later: A Look Back at 2023 in Python Packaging](https://chriswarrick.com/blog/2024/01/15/python-packaging-one-year-later/)
- [Pipenv: promises a lot, delivers very little](https://chriswarrick.com/blog/2018/07/17/pipenv-promises-a-lot-delivers-very-little/)
- [An unbiased evaluation of environment management and packaging tools](https://alpopkes.com/posts/python/packaging_tools)，此文章很有條理的比較，把套件管理器分為五個象限：虛擬環境管理、套件依賴解析、Python 版本管理、套件打包和套件發布
- [Poetry versus uv](https://www.loopwerk.io/articles/2024/python-poetry-vs-uv/)
- [Comparing the best Python project managers](https://medium.com/@digitalpower/comparing-the-best-python-project-managers-46061072bc3f)
- [Hatch or uv for a new project?](https://www.reddit.com/r/Python/comments/1gaz3tm/hatch_or_uv_for_a_new_project/)

一些資料來源

- [套件發布日期](https://www.warp.dev/blog/prose-about-poetry)
- [pyenv 發布日期](https://github.com/pyenv/pyenv/blob/6393a4dfce90615792751c6567c07a118a961ff9/CHANGELOG.md?plain=1#L1366)
- [pipenv 發布日期](https://chriswarrick.com/blog/2018/07/17/pipenv-promises-a-lot-delivers-very-little/)
- [conda 發布日期](https://docs.anaconda.com/anaconda/release-notes/#anaconda-0-8-0-jul-17-2012)

## 後話

其實看星星數就知道該怎麼選了，工程師又不是笨蛋，好用的星星數量自然多。pipenv 在 2017 橫空出世試圖解決所有問題，看似是明日之星但是騙完讚隔一年就迅速殞落。隨之而來的是 Poetry/PDM/hatch，從現在回頭看 PDM 其基於 PEP 582 的優勢隨著該提案被拒絕已經不復存在，速度又[比 Poetry 還慢](https://astral.sh/blog/uv)，社群又小，已經無明顯優勢。hatch 網路上的中文文章可以說是幾乎為零，需要用他的估計都是大型專案。這樣回頭看就很合理了，通用工具套件 pyenv 星星最多，最全面的套件 Poetry 第二高，pipenv 在 pyenv 發布五年後登高一呼騙完讚馬上倒地，全面又高效的 uv 不到一年已經超越 Poetry。

撰文整理時發現套件數量越寫越多，正好看到這篇文章：[How to improve Python packaging, or why fourteen tools are at least twelve too many](https://chriswarrick.com/blog/2023/01/15/how-to-improve-python-packaging/)，平常看到可能無感，寫的時候心理偷笑了一下，因為真的越寫越多寫不完。裡面還大罵 PyPA，因為這麼多套件都是 PyPA 相關的，不整合四散的套件，而且功能最豐富的 Poetry/PDM 反而都不是由 PyPA 維護的，最後的結尾是 `Furthermore, I consider that the PyPA must be destroyed`。

本文針對虛擬環境管理套件之間應該如何選擇進行介紹，網路上一堆文章只會說明指令操作，同樣的文章已經存在，再多十篇也沒有意義。其實有點像是做研究，一堆人都做過了，那做這個幹嘛？還是你的效能（文筆）特別好？都沒有的話重複寫文章只是浪費自己和讀者的時間而已，特別是網路 2024 了還在教 pipenv, virtualenv 的直接送上 uBlacklist，那種人寫的文章沒有任何參考性，連功課都不做就在寫文章，不只是浪費時間甚至在誤導人。

這篇文章寫給研究所時的自己，筆者研究訊號處理，整間實驗室只會算數學沒人會套件管理，當初沒多想學長用 Conda 就跟著用 Conda 了，不知道有這麼多替代品。
