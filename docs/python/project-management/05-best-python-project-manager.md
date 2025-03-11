---
title: Python 套件管理工具全面比較
sidebar_label: 套件管理工具全面比較
slug: /best-python-project-manager
tags:
  - Python
  - 套件管理工具
  - 虛擬環境管理工具
keywords:
  - Python
  - 套件管理工具
  - 虛擬環境管理工具
last_update:
  date: 2025-03-16T23:29:00+08:00
  author: zsl0621
first_publish:
  date: 2024-11-19T14:22:30+08:00
---


# Python 套件管理工具全面比較 - uv, Poetry, Pixi, Hatch, PDM, Conda, Mamba, Pipenv, venv, virtualenv

你知道 Python 有超過 10 個套件管理工具嗎？每個工具的出現都是為了解決前一個工具的痛點，然而網路教學文章總是劈哩啪啦說一堆指令，卻沒有回答我心中的問題：<u>**我該選擇這個套件嗎**</u>？於是本文是全中文圈唯一一篇完整整理 Python 專案管理工具的文章，我們應該要知道的是如何選擇，而不是照著指令複製貼上等出問題才知道到底適不適合。

## TL;DR

只有這三種考量

1. 小專案用 `uv`，使用方式簡單同時速度飛快
2. 稍微大一點的專案**一律在 Poetry 和 uv 之間二選一**，主要考量在於 Poetry 已經是穩定版，uv 還在開發階段
3. 科學開發屬於特殊情況，建議使用 Pixi 或 Mamba，原因是 Conda 現在和未來都不會支援 pyproject.toml，[#12462](https://github.com/conda/conda/issues/12462) 已經存在兩年都沒有提交

對於依賴解析這個課題，[Pixi 已經和 uv 合作開發](https://github.com/astral-sh/uv/issues/1572#issuecomment-1957318567)，uv 的[解析演算法](https://docs.astral.sh/uv/reference/resolver-internals/)原本就已經非常好，現在強強聯手後可以把他們視為 Python 世界最先進的套件解析工具，再來是 Poetry，再來就是 others。

對於 Python 科學開發，Pixi 寫了一篇文章說明[從 Conda 切換到 Pixi 的七個理由](https://prefix.dev/blog/pixi_a_fast_conda_alternative#reason-7-native-lockfiles)，我認為最重要的幾個原因是 PyPI 兼容、task 功能以及鎖定文件。

比較表格如下，只有新時代挑戰者組別，因為時代眼淚不會用了

| 工具 | Python 版本管理 | 鎖定文件 | 速度 | 優點 | 缺點 |
|------|---------------|---------|------|---------|---------|
| **uv** |  ✅ | ✅ | 非常快 | 速度極快，使用 PubGrub 演算法解析依賴，一站式體驗，取代多個工具 (pyenv/pip/pipx/venv) | 開發中，問題相對較少 |
| **Poetry** | ❌ | ✅ | 快 | 成熟穩定，高度成熟的依賴解析器，良好社群支援，完整功能 | 需配合 pyenv 管理 Python 版本 |
| **Pixi** | ✅ | ✅ | 非常快 | 支援 Conda 套件庫，使用 uv 的依賴解析能力 | 開發中，問題相對更多，知名度低，資源少，需要自己看文檔 |
| **Hatch** | ✅ | ❌ | 中等 | 遵循 PEP 標準，跨平台一致性，優秀的版本管理，可設定多環境 | 用戶少，專案僅一人維護，不支援 lockfile |
| **PDM** | ✅ | ✅ | 慢 | 遵循 PEP 標準 | 用戶少，專案僅一人維護，速度慢，相較其他工具無顯著優點 |

由於 Pixi 太新，如果遇到問題需要退回 Conda 體系，請用以 C++ 重寫 Conda 的 [Mamba](https://github.com/mamba-org/mamba) 而不是 Conda。

:::tip
筆者推薦 uv，懶得看文章可以直接看我寫的 [uv 使用教學](./uv-quick-guide)。
:::

## 工具目的

要知道怎麼選擇前得先清楚工具本身的目的，專案管理工具主要目的是解決依賴管理和環境隔離的問題，核心功能有

1. <u>**基礎功能**</u>：為每個專案建立獨立的 Python 環境，避免不同專案間的依賴不會互相干擾。
2. <u>**依賴套件解析**</u>：解析不同套件的依賴，這是非常重要的問題，除了避免依賴衝突，也增加可復現性。
3. <u>**pyproject.toml 支援**</u>：pyproject.toml 是 PEP 518/621 提出的 Python 專案標準配置文件，取代傳統的 setup.py，集中管理專案的各項設定，可提高專案的<u>**可復現性**</u>，不支援此文件基本上就直接淘汰。
4. 版本鎖定機制：支援鎖文件 (lockfile) 鎖定套件精確版本，確保開發、測試和生產環境一致性，一樣也是為了增加可復現性。
5. Python 版本管理：能管理和下載 Python 版本，不再需要額外工具管理不同的 Python 版本
6. 套件構建
7. 套件發佈

這些功能缺一不可，尤其是一到四，一是最基本的功能，二在不同工具之間就會開始有差異，三和四淘汰掉很多老舊專案，五往後是有的話最好，如果支援就不用再使用不同工具完成這些任務。

不會寫 pyproject.toml 也別擔心，我在 [uv 教學](/python/python-uv-complete-guide) 中有完整介紹。

## 選擇要點

上述七項是作為 Python 專案管理器的主要目的，了解目的之後接下來說明如何比較，我我認為以下是最重要的幾個項目：

1. <u>**基本可用**</u>，連基本功能都無法滿足的直接淘汰，例如不支援 pyproject.toml，或是依賴解析演算法很糟糕
2. <u>**速度可接受**</u>，速度慢到不可接受就算功能再齊全都不會想用他，速度問題在 CI/CD 階段尤其重要
3. <u>**使用方便**</u>，指令簡潔明瞭且符合直覺，不需要記憶過多特殊語法或繁複操作步驟
4. <u>**社群活躍度**</u>，有活躍的社群支援代表遇到問題比較容易找到解決方案，以及工具本身能持續更新維護
5. <u>**套件生態**</u>，整合 CI/CD，支援 C/Rust 等擴展，文檔完整性等等...

本文不會討論到這麼詳細但還是把這些要點列出來（原因是很多工具連基本可用的條件都達不到）。

選擇前我們可以透過 Github 星星數一探究竟，依照發布時間順序，星星數統計於撰文當下 (2024/11/19)，首先是時代的眼淚區：

1. [virtualenv](https://github.com/pypa/virtualenv) (2007, 4.8k stars)
2. [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)(2012, 6.4k stars)
3. [Conda](https://github.com/conda/conda) (2012, 6.5k stars)
4. [Pipenv](https://github.com/pypa/pipenv) (2017, 24.9k stars)

![Star History Chart](https://api.star-history.com/svg?repos=pypa/pipenv,pypa/virtualenv,conda/conda,pyenv/pyenv-virtualenv&type=Date)

接著是新時代的挑戰者（外來種入侵等級）

1. [Poetry](https://github.com/python-poetry/poetry) (2018, 31.8k stars)
2. [PDM](https://github.com/pdm-project/pdm) (2019, 8k stars)
3. [Hatch](https://github.com/pypa/hatch) (2022, 6.1k stars)
4. [pixi](https://github.com/prefix-dev/pixi) (2024, 3.1k stars)
5. [uv](https://github.com/astral-sh/uv) (2024, 26.7k stars)

![Star History Chart](https://api.star-history.com/svg?repos=python-poetry/poetry,astral-sh/uv,prefix-dev/pixi,pdm-project/pdm,pypa/hatch&type=Date)

## 詳細比較：時代的眼淚組

<details>

<summary>時代的眼淚</summary>

時代的眼淚組連身為套件管理器的基本功能都無法滿足，文章幾經更新後覺得這段沒必要看了，反正又不會用他們。

### venv/virtualenv

venv 是內建於 Python 的管理工具，是基於 virtualenv 的管理工具，差別是 virtualenv 需要額外安裝，他們本質上是相同的，差別只有 venv 不支援 Python2，於是我們知道**完全沒有任何理由使用 virtualenv**。這兩個只能用於建立虛擬環境，沒有任何依賴解析管理功能，一般是使用 pip 完成依賴解析管理。

### Pyenv

專注於 Python 版本管理的套件。

pyenv 專注於管理多個不同 Python 版本以及負責切換版本，不包含任何額外功能，需要和其他管理套件協同工作：他只是拼圖的其中一塊而已，但是把自己的這塊做好。pyenv 不支援 Windows，需要使用 [pyenv-win](https://github.com/pyenv-win/pyenv-win)。

### pyenv-virtualenv

pyenv 的插件，結合 virtualenv 後變成既有 Python 版本管理又有套件管理功能的完整工具，從這裡開始才真正進入戰場，涵蓋環境管理、Python 版本管理以及依賴解析管理（還是使用 pip 解析，無優化）。

### Conda

這段可以總結成「數學算超快但是答案全錯」，Conda 和這個笑話的情況完全一樣。

Conda 筆者用的可多了，使用體驗就是兩個字「快逃」。Conda 整合 Python 版本管理、虛擬環境管理和套件依賴解析，完美整合自家贊助的 Spyder IDE，安裝完 Spyder 後能直接使用虛擬環境不需任何設定，還把套件打包二進制安裝速度更快。聽起來很棒，但 Conda 有一個致命問題是套件怎麼裝怎麼崩潰，即使是全新環境也一樣。

Conda 使用自己的套件依賴解析器而不是 pip，所以混用會出問題，你說為甚麼要混用？因為內建的永遠出現衝突問題，就拿筆者最常使用的套件 Tensorflow/Matplotlib/Numba 三個來說，三個都依賴 Numpy，三個都是知名套件，但是三個一起裝就是 boom 爆炸，筆者當時甚至研究出先使用 Conda 安裝 A，再使用 pip 安裝 B，最後再用 Conda 安裝 C 才能成功安裝，任何改變都會出錯。平常最好祈禱自己不要手癢想更新，更新就是套件再重裝一次。

Conda 在 2023 年把內部的解析演算法改成 [libmamba](https://github.com/conda/conda-libmamba-solver)，特色只說了速度更快沒有提到正確性。當一個套件管理器連套件都裝不成功的時候，我才不關心你快不快。

Conda 有很多衍生版本如 miniconda, Mamba, Miniforge, Anaconda Navigator 等等，筆者沒有深入研究細節，畢竟連最有名的 Conda 功能都這麼糟糕了我完全不指望衍生版本能給我多大驚喜，用過 miniconda 沒有感受到任何差異，這幾個衍生版本筆者只知道是輕量化的、C++ 重寫的、C++ 重寫加上輕量化的、UI 版本的 Conda。

### pipenv

顧名思義，是整合了 pip 和虛擬環境管理的工具，目的是簡化依賴管理與環境設置。總共改善了這些問題：

- 引入 Pipfile 和 Pipfile.lock 解析依賴與鎖定
- 開發/生產環境分離
- 使用自行開發的依賴解析器

pipenv 整合環境管理和依賴管理，使用 pipfile 和 pipfile.lock 管理，並且支援 `--dev` 選項隔離開發用套件，如此豐富的功能整合看起來很美好，但是實際上[不美好🚨](https://note.koko.guru/posts/using-poetry-manage-python-package-environments)。本套件筆者並未使用過所以上網做了一下功課，看到此篇文章：[Pipenv: promises a lot, delivers very little](https://chriswarrick.com/blog/2018/07/17/pipenv-promises-a-lot-delivers-very-little/)，整整四千字的心得大罵 pipenv，罵人能罵到四千字也是不容易，所以我們大概可以知道該如何選擇，其主要問題包含效能、指令操作繁瑣、僅支持特定的工作流程、不支援跨目錄工作，而且更新停滯。

對於一個 2017 年才發布的套件，在 2018 年就被罵不更新，2020 年該文章作者還繼續追蹤但是情況依舊。查看其 [Github release](https://github.com/pypa/pipenv/releases)，整個 2023 從二月之後就沒更新了，如此緩慢的更新不建議使用。

都 2025 了如果還有文章推薦 pipenv 建議送上 uBlacklist 伺候。

### 其他

[virtualenvwrapper](https://github.com/python-virtualenvwrapper/virtualenvwrapper) 則是基於 virtualenv 的工具，把所有環境整合在同一目錄，並且提供更多額外功能，本質上還是 virtualenv，[pyenv-virtualenvwrapper](https://github.com/pyenv/pyenv-virtualenvwrapper) 同理，兩者也都沒更新了，不列入討論。

</details>

## 詳細比較：新時代的挑戰者

### uv

如果要以要一句話形容 uv，那就是完整且高效的一站式體驗，解決了時代眼淚組的所有問題。

uv 是 2024/2 才首發的新工具，簡單摘要幾個特點：

1. 由 rust 撰寫，標榜快速，比 Poetry 快十倍以上
2. 使用 PubGrub 演算法[解析套件依賴關係](https://docs.astral.sh/uv/reference/resolver-internals/)
3. **<u>取代 pyenv</u>**：支援 Python 版本管理[^global]
4. **<u>完美取代 pip/pip-tools</u>**：支援 lockfile 鎖定套件版本
5. **<u>完美取代 pipx</u>**：支援全域套件安裝
6. 發展快速，發布不到一年已經有 26k 星星

把特點 2\~4 加起來就是我們的最終目標了，有更好的套件解析演算法，不只支援 lockfile 管理套件，也支援 Python 版本管理，還沒有 pipenv 速度緩慢且更新停滯的問題，是目前虛擬環境管理工具的首選。

為何選擇 uv？我會給出這個結論：「一個工具完整取代 pyenv/pipx，幾乎包含 poetry 的所有功能，速度又快」，這麼多優點是我可以一次擁有的嗎，太夢幻了吧。

身為新穎又備受矚目的套件，目前的更新速度非常快，[兩個月就把問題解決了](https://www.loopwerk.io/articles/2024/python-uv-revisited/)，不單單只是這個問題，uv 目前基本上維持每個月兩個 release 的更新週期，而且不是因為功能少才頻繁更新，而是不斷的新增功能。

[^global]: 只剩下等效於 `pyenv global` 的設定全局 Python 功能還不支援但[已經在規劃中](https://github.com/astral-sh/uv/issues/6265)，加上 `--preview --default` 參數即可使用，目前實測還很早期，連 venv 都不能跑。

:::tip 使用心得

uv 內建的 Python 版本管理非常方便，不再需要 pyenv 多記一套指令，還支援安裝全局套件以取代 pipx，支援使用 `uv build` 和 `uv publish` 功能可以方便的構建和發布套件，還做了 `uv pip` 方便以往的 `pip` 用戶輕鬆上手（網路上那些只知道 uv 速度很快，但是真正實用的功能完全沒用到的文章也是這樣來的），除此之外還有最重要的 `uv run` 功能提供了非常優秀的開發便利性，再加上[超快的安裝和解析速度](https://astral.sh/blog/uv-unified-python-packaging)錦上添花，每個專案都使用各自的 `.venv` 虛擬環境方便管理，虛擬環境的套件使用 hard link 避免重複下載降低容量，筆者認為目前虛擬環境管理工具首選就是他了。

有兩個小缺點，第一是使用 rust 撰寫，所以 Python 開發者不好進行貢獻，第二是太新，連英文都沒有幾篇文章說明如何使用，不過別擔心筆者寫了使用較學，從安裝到發布套件一應俱全。
:::

:::tip 開始使用 uv

- 可以參考筆者寫的 [uv 使用教學](./uv-quick-guide)。  
- 非常優秀的開發便利性？請見 uv 使用教學的[這個段落](./python-uv-complete-guide/#uv-run)。  

:::

### Poetry

雖然文章開頭說「新工具的出現就是要解決舊有工具的痛點」，但前面好像都沒什麼提到，因為不是缺少功能就是問題很大根本沒得比。Poetry 的出現主要解決了以下問題：

- 使用高度成熟的依賴解析器解決依賴解析問題
- 內建套件發布工具解決套件發布流程複雜問題

並且有以下特點：

- 支援 pyproject.toml
- 支援 poetry.lock 鎖定套件版本
- 內建虛擬環境管理
- 和 uv 相比已經是穩定版本，經得起考驗

Poetry 最大的優勢是支援 pyproject.toml，可以設定從開發到發布的所有項目，使用自行開發的依賴解析器，支援 poetry.lock 完成依賴鎖定管理，背景使用 pip 讓兼容過往工具，看起來很美好，而且真的很美好。對比 pipenv 有更快的套件下載和更簡易的指令操作，以及更好的開發環境（整合各種開發工具的設定）；對比 uv 則原生支援套件打包發布。使用體驗上，用戶人數足夠多，相關資訊比 PDM/hatch 更充足，cli 提示做得很不錯，以往筆者用過的工具都需要不斷切換視窗搜尋指令，使用 Poetry 可以明顯感受到頻率降低很多，唯一美中不足的是不包含 Python 版本管理，需要結合 pyenv 使用。

> 網路上會說 Poetry 不支援 PEP 621，其實已計畫將於 [2.0 版本](https://github.com/orgs/python-poetry/discussions/5833)開始支援。

Poetry 的使用指令可以觀看這兩篇文章：簡短的 [\[note\] Python Poetry](https://pjchender.dev/python/note-python-poetry/) 和更豐富的 [Pyenv + Poetry 相關指令](https://hackmd.io/@OpenAIDocuments/ByIK7hTEa)，新手用戶不用管太多只要會 `poetry init` + `poetry add` 就可以了。

:::tip 和 uv 比較
hatch 由於其自身的特殊性（沒有 lock 檔案，沒有內建虛擬環境管理，主要用於跨平台而不是虛擬環境管理和依賴解析）所以沒得比較，對比 uv 在功能上則更為相似。

在這個部落格中寫了五篇關於 uv 的[文章](https://www.loopwerk.io/articles/tag/uv/)，我們可以看到 uv 的發展之快以及齊全的功能。該作者原本使用 poetry，五篇文章一路從 uv 還不夠好寫到全面遷移 uv，並且在他自己的 dotfiles 裡面[把 pyenv 和 poetry 都移除了](https://github.com/search?q=repo%3Akevinrenskers%2Fdotfiles+uv&type=commits)，可見 uv 有多強大。

兩者的選擇和開頭說的一樣，需要更成熟穩定且經過多數用戶驗證的工具，選擇 Poetry，想擁有一站式的良好使用體驗選擇 uv，只是 uv 仍然在早期階段有些細節還需要優化。
:::

### Pixi

Pixi 做的最正確的事情就是站在巨人的肩膀上。

對於科學開發所需要的 C 語言、R 語言套件，他支援[直接使用 Conda 的套件庫](https://prefix.dev/blog/pixi_global)；對於套件版本解析，Pixi 使用現今 Python 世界裡面[最強大的 uv 作為依賴解析工具](https://github.com/prefix-dev/pixi/pull/863)；對於 lockfile 鎖定文件原生支援，連 conda-lock 作者都說 Pixi 就是 Conda 加上鎖文件的[正確實踐結果](https://prefix.dev/blog/pixi_a_fast_conda_alternative#reason-7-native-lockfiles)，基本上我們可以把他理解為支援 Conda 庫的 uv。

缺點當然有，資源少基本上必須得翻文檔，另外就是還在開發階段會有些問題，例如 [Safouane Chergui](https://ericmjl.github.io/blog/2024/8/16/its-time-to-try-out-pixi/) 說的版本問題就需要特別處理，最後一個是我最擔心的問題，他的願景很大但是知名度不夠，未來發展令人擔憂。

- [Let's stop dependency hell](https://prefix.dev/blog/launching_pixi)
- [It's time to try out pixi!](https://ericmjl.github.io/blog/2024/8/16/its-time-to-try-out-pixi/)
- [Pixi Global: Declarative Tool Installation](https://prefix.dev/blog/pixi_global)
- [Towards a Vendor-Lock-In-Free conda Experience](https://prefix.dev/blog/towards_a_vendor_lock_in_free_conda_experience)
- [7 Reasons to Switch from Conda to Pixi](https://prefix.dev/blog/pixi_a_fast_conda_alternative#reason-7-native-lockfiles)

### PDM/hatch

由於是開發管理工具所以不只關注環境管理本身，而是更加注重對於 PEP 的支援，但筆者沒這麼厲害所以不知道確切的影響是什麼，以下大致整理一下網路資訊。

- hatch: 專注在開發的工具
  1. 旨在建立跨平台間能具有相同環境和操作的套件（例如解決 Windows 不支援 Pyenv）
  2. 不支援 lock 鎖定套件版本（[未來會支援](https://github.com/pypa/hatch/issues/749)）
  3. 允許一個專案使用多個環境
  4. 優秀的[開發版本管理](https://myapollo.com.tw/post/python-hatch/)

- PDM: [基於 PEP 582 而生的套件](https://pdm-project.org/zh-cn/latest/usage/pep582/)
  1. PEP 582 用於免虛擬環境進行 Python 管理，然而此提案最終被拒絕
  2. 不支援 Python 版本管理，速度[比 poetry 慢](https://astral.sh/blog/uv)，社群比 poetry 小，找了老半天好像找不到什麼優點

這兩個工具最大的問題是使用人數少，開發社群更小，問題能不能解決基本上要看開發者心情好不好，因為只有一個人在開發。

:::note
Chris Warrick: [PDM is a one-man-show, like Hatch.](https://chriswarrick.com/blog/2024/01/15/python-packaging-one-year-later/)
:::

## 總結

Poetry/uv 最好，兩者選擇取決於你喜歡更成熟的架構還是需要稍微自己摸索但是更好的整體使用體驗。hatch 很好，但是著重於多版本並存的管理，套件依賴問題需要其他工具幫忙，如果專案夠複雜可以選擇，但是需要用到的人不會來看這篇文章。Conda 很糟糕，就算需要 Conda 生態系也是選使用 C++ 重寫的 Mamba，或者乾脆換成 Pixi。如果你的是科學開發專案並且剛好沒有依賴問題可以選擇 Conda 系列，然後別忘了祈禱自己不要遇到依賴解析問題。

總結來說

1. 其餘一律使用 Poetry/uv
2. 科學開發使用 Pixi 或 Mamba
3. 需要用到 hatch 的人不會看這篇文章

本文解決了現有網路文章缺少的選擇分析，透過完整說明功能和套件現況，讀者可以清楚知道自己應該選擇哪個虛擬環境管理套件。

## 參考文章

- [Python Packaging, One Year Later: A Look Back at 2023 in Python Packaging](https://chriswarrick.com/blog/2024/01/15/python-packaging-one-year-later/)
- [Pipenv: promises a lot, delivers very little](https://chriswarrick.com/blog/2018/07/17/pipenv-promises-a-lot-delivers-very-little/)
- [An unbiased evaluation of environment management and packaging tools](https://alpopkes.com/posts/python/packaging_tools)，文章討論時把套件管理器分為五個象限：虛擬環境管理、套件依賴解析、Python 版本管理、套件打包和套件發布
- [Poetry versus uv](https://www.loopwerk.io/articles/2024/python-poetry-vs-uv/)
- [Comparing the best Python project managers](https://medium.com/@digitalpower/comparing-the-best-python-project-managers-46061072bc3f)
- [Hatch or uv for a new project?](https://www.reddit.com/r/Python/comments/1gaz3tm/hatch_or_uv_for_a_new_project/)

一些資料來源

- [套件發布日期](https://www.warp.dev/blog/prose-about-poetry)
- [pyenv 發布日期](https://github.com/pyenv/pyenv/blob/6393a4dfce90615792751c6567c07a118a961ff9/CHANGELOG.md?plain=1#L1366)
- [pipenv 發布日期](https://chriswarrick.com/blog/2018/07/17/pipenv-promises-a-lot-delivers-very-little/)
- [conda 發布日期](https://docs.anaconda.com/anaconda/release-notes/#anaconda-0-8-0-jul-17-2012)

## 後話

其實看星星數就知道該怎麼選了，工程師又不是笨蛋，好用的星星數量自然多。pipenv 在 2017 橫空出世試圖解決一次三個問題，看似是明日之星但是在隔一年就迅速殞落。隨之而來的是 Poetry/PDM/hatch，從現在回頭看 PDM 其基於 PEP 582 的優勢隨著該提案被拒絕已經不復存在，速度又[比 Poetry 還慢](https://astral.sh/blog/uv)，社群又小，已經無明顯優勢。hatch 網路上的中文文章可以說是幾乎為零，需要用他的估計都是大型專案。這樣回頭看就很合理了，通用工具套件 pyenv 星星最多，最全面的套件 Poetry 第二高，pipenv 在 pyenv 發布五年後登高一呼騙完讚馬上倒地，全面又高效的 uv 不到一年馬上登上第三名寶座，已經逼宮 Poetry。

撰文整理時發現套件數量越寫越多，正好看到這篇文章：[How to improve Python packaging, or why fourteen tools are at least twelve too many](https://chriswarrick.com/blog/2023/01/15/how-to-improve-python-packaging/)，平常看到可能無感，寫的時候心理偷笑了一下，因為真的越寫越多寫不完。裡面還大罵 PyPA，因為這麼多套件都是 PyPA 相關的，不整合四散的套件，而且功能最豐富的 Poetry/PDM 反而都不是由 PyPA 維護的，最後的結尾是 `Furthermore, I consider that the PyPA must be destroyed`。

本文針對虛擬環境管理套件之間應該如何選擇進行介紹，網路上一堆文章只會說明指令操作，同樣的文章已經存在，再多十篇也沒有意義。其實有點像是做研究，一堆人都做過了，那做這個幹嘛？還是你的效能（文筆）特別好？都沒有的話重複寫文章只是浪費自己和讀者的時間而已。

這篇文章寫給研究所時的自己，筆者研究訊號處理，整間實驗室只會算數學沒人會套件管理，當初沒多想學長用 Conda 就跟著用 Conda 了，不知道有這麼多替代品。
