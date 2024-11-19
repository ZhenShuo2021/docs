---
title: 初嘗 Python 工作流自動化
description: 初嘗 Python 工作流自動化
tags:
  - Programming
  - Python
keywords:
  - Programming
  - Python
  - pyproject
  - pre-commit
  - ruff
  - isort
  - mypy
last_update:
  date: 2024-11-19T14:22:30+08:00
  author: zsl0621
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 初嘗 Python 工作流自動化

看了很多 [*Code and Me*](https://blog.kyomind.tw/pyproject-toml/) 的文章，想說自己也來試試工作流自動化看好了，所以這篇是使用紀錄。

他的工作流自動化文章數量非常多，隨便點沒有特別找就有以下幾篇：

- [Python 開發：pyproject.toml 介紹 + 使用教學](https://blog.kyomind.tw/pyproject-toml/)
- [Python 專案從 Flake8、Black 遷移至 Ruff 指南](https://blog.kyomind.tw/migrate-to-ruff/)
- [Python 開發：Ruff Linter、Formatter 介紹 + 設定教學](https://blog.kyomind.tw/ruff/)
- [確保 isort 正確排序本地模組：pyproject.toml 與 pre-commit 設定](https://blog.kyomind.tw/isort-local-module-sorting/)
- [Python 開發：pre-commit 設定 Git Hooks 教學](https://blog.kyomind.tw/pre-commit/)

但是很可惜的沒有統整的文章，細節也太多，於是本文簡短紀錄自己的設定，包含 pyproject.toml + pre-commit。在搜尋資料的時候有發現資訊很少，查到後面才想到一整個專案只要有一個人負責設定就好了，那當然沒什麼人知道怎麼設定。

目前我已知的工作流自動化包含了以下幾點：

1. <u>靜態型別檢查 Type Checking</u>: 檢查變數型別有沒有使用錯誤。
2. <u>程式碼風格檢查 Linter</u>: 檢查程式碼是否違反 PEP 中的建議，也可以自行設定使用哪種風格。
3. <u>程式碼格式化工具 Formatter</u>: 自動格式化程式碼，讓所有程式碼維持相同格式。
4. <u>單元測試 Unittest</u>: 測試你的程式碼是否在各種邊界情況中正常運作。
5. <u>覆蓋率測試 Coverage</u>: 檢查有哪些程式碼沒有被單元測試覆蓋。
6. <u>安全檢查</u>: 例如有沒有[放到秘密](https://github.com/pre-commit/pre-commit-hooks/blob/main/pre_commit_hooks/detect_aws_credentials.py)。

經過一段時間的研究，最後我選擇的是 mypy (type check)，ruff (linter+formatter)，pytest (unittest)，沒測試覆蓋率 (因為測試只寫一個測試不想自暴其短)。

選擇 mypy 的原因是 pyright 整天在跟我說檢查不到套件，用 mypy 一行 disable import-not-found 就好了；ruff 功能強大且高效，pytest 則是從內建的 unittest 轉過來，語法確實比較簡潔，也整合 pdb 偵錯，coverage 我就只是玩過而已了。

> ruff 官網介紹自己核心理念不是創造新功能，而是在現有功能下達到更快的執行速度（這點非常好，一堆套件都不寫自己到底跟別人有什麼差別，浪費大家時間，要自己研究或踩坑才知道）。就我的爛 code 一個檔案不到一千行用 black 都會卡一下了，大型專案還得了。yapf 沒有試過，PEP8 則是修改太少，沒有按下去修正一堆的那種愉悅感。

說在前面，我失業菜雞菜到不行，這只是自己一個人找東西玩的紀錄，別問我不同設定差在哪，窩不知道。

## pyproject.toml
參考[官方文檔](https://python-poetry.org/docs/pyproject/)。

### 設定建構系統 build-system

必填，設定使用哪種打包工具。

```toml
# 例如以往 setup.py 中的常見的 setuptools
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

# 或者新穎的 poetry
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

# 或者更新的 hatch
requires = ["hatchling>=1.24.2", "hatch-vcs>=0.3.0"]
build-backend = "hatchling.build"
```

### 專案元資料 metedata

從這裡開始每個專案都不太一樣，由於我自己使用 poetry，所以只介紹 poetry，首先是必填項目。

```toml
[tool.poetry]
name = "發布到PyPI的名字"
version = "套件版本"
description = "簡短描述，在PyPI的列表簡介顯示"
authors = ["Your Name <your.email@example.com>"]
```

這些選填基本上都會附上
```toml
maintainers = ["Maintainer Name <maintainer@example.com>"]
repository = "儲存庫網址"
homepage = "專案網站"
license = "MIT"
readme = "README.md"
```

以下是 PyPI 的標籤，方便搜尋和分類使用
```toml
classifiers = [
    "Topic :: Multimedia :: Video",
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.9",
]
keywords = ["python", "cli", "scraper"]
```

### 依賴
設定專案依賴的套件們。

```toml
[tool.poetry.dependencies]
python = "^3.10"
colorama = "^0.4.6"
DrissionPage = "^4.1.0.9"
```

- ^：允許向上兼容的版本
- ~：允許補丁版本更新
- ==：精確版本
- Optional dependencies：可選依賴

以及開發者使用的依賴：

```toml
[tool.poetry.group.dev.dependencies]
ruff = "^0.7.1"
mypy = "^1.13.0"
pre-commit = "^4.0.0"
```

### 入口點
如果是 cli 工具可以設定套件入口點，在 your_package 資料夾中的 cli.py，裡面的 main 函式。

```toml
[tool.poetry.scripts]
mycli = "your_package.cli:main"
```

### 開發工具設定
此處開始才是 pyproject.toml 的核心優勢，可以設定完整的工具設定保持專案開發的一致性。我自己的設定因為 ruff 包含了 code style 檢查和 linter ，所以只有設定他：

```toml

[tool.ruff]
line-length = 100
exclude = [".git", "build", ".tox", ".eggs"]
preview = true

[tool.ruff.lint]
explicit-preview-rules = true
allowed-confusables = ["，"]
select = [
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "D",   # pydocstyle
    "E",   # pycodestyle
]
```

### 完整結果
ruff.lint 和 ignore 都是從大專案抄來的，例如 yt-dlp/pytorch/numpy/matplotlib 等等，如果想要自行設定可以[在 ruff 官網中查看所有規則](https://docs.astral.sh/ruff/rules/)，相信我看完之後你就會想直接抄作業了。在工作區根目錄設定完成後 VSCode 似乎能自動偵測，我沒有任何 IDE 就自動顯示檢查結果了。

```toml
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "v2dl"
version = "0.0.5"
description = "V2PH downloader"
authors = ["ZhenShuo2021 <leo01412123@gmail.com>"]
repository = "https://github.com/ZhenShuo2021/V2PH-Downloader"
homepage = "https://github.com/ZhenShuo2021/V2PH-Downloader"
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
colorama = "^0.4.6"
DrissionPage = "^4.1.0.9"
python-dotenv = "^1.0.1"
selenium = "^4.25.0"
lxml = "^5.3.0"
PyYAML = "^6.0.2"

[tool.poetry.group.dev.dependencies]
ruff = "^0.7.1"
mypy = "^1.13.0"
pre-commit = "^4.0.0"
pytest = "^8.3.3"
tox = "^4.23.0"

[tool.poetry.scripts]
v2dl = "v2dl.v2dl:main"

[tool.ruff]
line-length = 100
exclude = [".git", "build", ".tox", ".eggs"]
preview = true

[tool.ruff.lint]
explicit-preview-rules = true
allowed-confusables = ["，"]
select = [
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "D",   # pydocstyle
    "E",   # pycodestyle
    "EXE", # flake8-executable
    "F",   # pyflakes
    "G",   # flake8-logging-format
    "I",   # isort
    "LOG", # flake8-logging
    "N",   # pep8-naming
    "NPY", # NumPy-specific rules
    "PIE", # flake8-pie
    "PLC", # pylint
    "PLE",
    "PLR",
    "PLW",
    "PT",  # flake8-pytest-style
    "PYI", # flake8-pyi
    "Q",   # flake8-quotes
    "RSE", # flake8-raise
    "RUF", # ruff-specific rules
    "SIM", # flake8-simplify
    "TCH", # flake8-type-checking
    # "TRY",
    "UP", # pyupgrade
    "W",  # Warning
]
ignore = [
    "D100",    # Missing docstring in public module
    "D101",    # Missing docstring in public class
    "D102",    # Missing docstring in public method
    "D103",    # Missing docstring in public function
    "D104",    # Missing docstring in public package
    "D105",    # Missing docstring in magic method
    "D106",    # Missing docstring in public nested class
    "D107",    # Missing docstring in `__init__`RuffD107
    "D200",
    "D201",
    "D202",
    "D203",
    "D204",
    "D205",
    "D213",
    "D301",
    "D400",
    "D401",
    "D403",
    "D404",
    "D413",    # Missing blank line after last section
    "E741",
    "F841",
    "B007",
    "B008",
    "B017",
    "B018",
    "B023",
    "B028",
    "E402",
    "C408",
    "E501",
    "E721",
    "E731",
    "E741",
    "EXE001",
    "F405",
    "F841",
    "G101",
    "NPY002",
    "PERF203",
    "PERF401",
    "PERF403",
    "PYI024",
    "PYI036",
    "PYI041",
    "PYI056",
    "SIM102",
    "SIM103",
    "SIM112",
    "SIM113",
    "SIM105",
    "SIM108",
    "SIM110",
    "SIM114",
    "SIM115",
    "SIM116",
    "SIM117",
    "SIM118",
    "UP006",
    "UP007",
    "W292",
]
```

## .pre-commit-config.yaml
自動化幫你在 commit 前進行檢查，我只能說他是個神器，相見恨晚，從此再也不需要 `mypy ...` `ruff ...` `black ...` `isort ...`，所有檢查一行指令完成。

### 指令
先安裝套件 `pip install pre-commit` `pre-commit install` `pre-commit install --install-hooks`，之後使用以下指令使用 pre-commit：

```sh
# 執行所有檢查
pre-commit run -a 
# 提交前跳過檢查
git commit --no-verify
```

### 完整結果
設定也非常簡單，直接上結果，設定完成後每次提交都會自動執行以下檢查，從此之後這些指令只要使用 `pre-commit run -a` 就直接完成，不用一個一個打也不用記參數，甚至有信心這次提交全對的話連 pre-commit 都不用打，直接 git commit 也會幫你檢查，全都不用記，超讚。

```sh
isort --line-length=100
ruff check file --fix
mypy file --disable-error-code=import-untyped --disable-error-code=no-untyped-def --check-untyped-defs
pytest -s -v
```

```yaml
repos:
  # 此項是基本的 commit hook 設定
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
    - id: end-of-file-fixer
    - id: check-yaml
    - id: check-added-large-files

  # 設定 isort 自動排序 import 語法
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        exclude: safe_house/
        args:
          - --line-length=100

  # 設定 ruff code formatter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.1
    hooks:
    - id: ruff
      exclude: safe_house/
      args: [--fix, --exit-non-zero-on-fix]
    - id: ruff-format

  # 設定 mypy type check
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
    - id: mypy
      exclude: safe_house/
      args:
        - "--disable-error-code=import-untyped"
        - "--disable-error-code=import-not-found"
        - "--check-untyped-defs"

  # 設定 pytest
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: "pytest -s -v"
        language: system
        pass_filenames: false
        always_run: true
```


## 使用範例
這既是我第一次設定這些文件，也是我第一次使用程式碼品質工具，第一次使用的錯誤多到炸裂：

> 第一次檢查 ruff 的警告數量高達 113 個
![第一次檢查 ruff 的警告數量](ruff-pre.png "第一次檢查 ruff 的警告數量")

<br/>

> ruff 也有自動修復功能，對於一些小錯誤可以自行修復，自行修復 29 個錯誤後的結果，還剩下 84 個：
![自行修復後的結果](ruff-post.png "自行修復後的結果")

<br/>

> mypy 初次檢查也是嚇死人
![mypy 初次檢查也是嚇死人](mypy.png "mypy 初次檢查也是嚇死人")

<br/>

> 經過好一段努力後終於修復所有問題
![經過好一段努力後終於修復所有問題](finally.png "經過好一段努力後終於修復所有問題")

不得不說這有點像是玩遊戲的通關獎勵，送你一堆綠色 pass。

## 文章更新：新的設定檔案

> 2024/11/19

最近又對這兩個文件的設定進行更新，主要是照抄 yt-dlp 裡面的 ruff 設定，使用更嚴格的 mypy 檢查，加上 bandit 檢查程式漏洞，還有在 pyproject.toml 改變時自動匯出套件依賴到 requirements.txt，並且把主要設定移動到 pyproject.toml，.pre-commit-config.yaml 只保留要運行哪些 hooks 的設定。

<details>

<summary>更新結果</summary>

```toml
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "v2dl"
version = "0.1.4"
description = "V2PH downloader"
authors = ["ZhenShuo2021 <leo01412123@gmail.com>"]
repository = "https://github.com/ZhenShuo2021/V2PH-Downloader"
homepage = "https://github.com/ZhenShuo2021/V2PH-Downloader"
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
colorama = "^0.4.6"
DrissionPage = "^4.1.0.9"
python-dotenv = "^1.0.1"
selenium = "^4.25.0"
lxml = "^5.3.0"
PyYAML = "^6.0.2"
pynacl = "^1.5.0"
questionary = "^2.0.1"
httpx = "^0.27.2"
pathvalidate = "^3.2.1"

[tool.poetry.group.dev.dependencies]
ruff = "^0.7.1"
mypy = "^1.13.0"
pre-commit = "^4.0.0"
pytest = "^8.3.3"
tox = "^4.23.0"
isort = "^5.13.2"

[tool.poetry.scripts]
v2dl = "v2dl:main"

[tool.mypy]
ignore_missing_imports = true
strict = true
check_untyped_defs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
warn_redundant_casts = true
warn_unreachable = true
warn_unused_ignores = true
follow_imports = "silent"
disable_error_code = [
    "import-untyped",
    "import-not-found",
    "no-untyped-call",
    "no-any-return",
    "unused-ignore",
]
enable_error_code = ["attr-defined", "name-defined"]
# enable_error_code = [
#   "ignore-without-code",
#   "redundant-expr",
#   "truthy-bool",
# ]


[tool.ruff]
line-length = 100
exclude = [".git", "build", ".tox", ".eggs", "safe_house"]
preview = true
target-version = "py310"

[tool.ruff.format]
docstring-code-format = true
quote-style = "double"

[tool.ruff.per-file-ignores]
"v2dl/cli/account_cli.py" = ["T201"]
"v2dl/utils/security.py" = ["T201"]

[tool.ruff.lint]
explicit-preview-rules = true
allowed-confusables = ["，", "。", "（", "）"]
ignore = [
    "E402",    # module-import-not-at-top-of-file
    "E501",    # line-too-long
    "E731",    # lambda-assignment
    "E741",    # ambiguous-variable-name
    "UP036",   # outdated-version-block
    "B006",    # mutable-argument-default
    "B008",    # function-call-in-default-argument
    "B011",    # assert-false
    "B017",    # assert-raises-exception
    "B023",    # function-uses-loop-variable (false positives)
    "B028",    # no-explicit-stacklevel
    "B904",    # raise-without-from-inside-except
    "C401",    # unnecessary-generator-set
    "C402",    # unnecessary-generator-dict
    "PIE790",  # unnecessary-placeholder
    "SIM102",  # collapsible-if
    "SIM108",  # if-else-block-instead-of-if-exp
    "SIM112",  # uncapitalized-environment-variables
    "SIM113",  # enumerate-for-loop
    "SIM114",  # if-with-same-arms
    "SIM115",  # open-file-with-context-handler
    "SIM117",  # multiple-with-statements
    "SIM223",  # expr-and-false
    "SIM300",  # yoda-conditions
    "TD001",   # invalid-todo-tag
    "TD002",   # missing-todo-author
    "TD003",   # missing-todo-link
    "PLE0604", # invalid-all-object (false positives)
    "PLE0643", # potential-index-error (false positives)
    "PLW0603", # global-statement
    "PLW1510", # subprocess-run-without-check
    "PLW2901", # redefined-loop-name
    "RUF001",  # ambiguous-unicode-character-string
    "RUF012",  # mutable-class-default
    "RUF100",  # unused-noqa (flake8 has slightly different behavior)
]
select = [
    "E",      # pycodestyle Error
    "W",      # pycodestyle Warning
    "F",      # Pyflakes
    "I",      # isort
    "Q",      # flake8-quotes
    "N803",   # invalid-argument-name
    "N804",   # invalid-first-argument-name-for-class-method
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "A",      # flake8-builtins
    "COM",    # flake8-commas
    "C4",     # flake8-comprehensions
    "FA",     # flake8-future-annotations
    "ISC",    # flake8-implicit-str-concat
    "ICN003", # banned-import-from
    "PIE",    # flake8-pie
    "T20",    # flake8-print
    "RSE",    # flake8-raise
    "RET504", # unnecessary-assign
    "SIM",    # flake8-simplify
    "TID251", # banned-api
    "TD",     # flake8-todos
    "PLC",    # Pylint Convention
    "PLE",    # Pylint Error
    "PLW",    # Pylint Warning
    "RUF",    # Ruff-specific rules
]

[tool.ruff.lint.isort]
force-single-line = false
combine-as-imports = true
length-sort-straight = true
relative-imports-order = "closest-to-furthest"
```

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: detect-private-key
      - id: debug-statements
      - id: check-case-conflict

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.10
    hooks:
      - id: bandit
        args: ["-ll"]

  - repo: https://github.com/asottile/pyupgrade
    rev: v3.19.0
    hooks:
      - id: pyupgrade

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        args: ["--config-file=pyproject.toml"]
        exclude: ^(safe_house/|tests/)
        additional_dependencies:
          - types-PyYAML

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: "pytest -s -v"
        language: system
        pass_filenames: false
        always_run: true

  - repo: local
    hooks:
      - id: poetry-export
        name: poetry-export
        entry: bash -c 'poetry export -f requirements.txt --output requirements.txt --without-hashes'
        language: system
        pass_filenames: false
        files: ^(pyproject.toml|poetry.lock)$
        stages: [commit]
        always_run: false
```

</details>

## 心得
潮～爽～DER～ pre-commit 一行指令完成所有工作。

好啦正經一點，基本的 pre-commit-hooks 可以檢查是否提交大檔案，也真的讓我發現有一兩次不小心把圖片也 stage 了。

這些工具也可以幫我們多學一些平常不會碰到的知識，比如說 ruff 警告 try-except 後面的 logger 加上 exc_info=True 時，會提醒你[直接使用 logger.exception](https://docs.astral.sh/ruff/rules/logging-redundant-exc-info/#why-is-this-bad) 簡寫，還有 logger 使用 f-string 會造成[效能問題](https://docs.astral.sh/ruff/rules/logging-f-string/)，以及自動檢查程式碼中的 [magic number](https://docs.astral.sh/ruff/rules/magic-value-comparison/) 等等，還滿方便的。

isort 會自動幫我們排列所有 import 語句，這方面見仁見智，至少避免了強迫症每次都想手動排列浪費的時間。

mypy 的話則是地獄，為了改他多學了泛型，直接多寫了一篇文章（[Type Hint 教學：從入門到進階的 Python 型別註釋](/docs/python/type-hint-typing-complete-guide)），不過這就是初期成本，無法避免。

pytest 則可以在每次提交時自動執行，讓你想懶也不行，大家都知道打字會死，我相信沒有任何人想在提交時多打 --no-verify，寧願不打讓他自己跑測試假裝自己有在做事。

coverage 則是痛苦面具，程式碼寫完就夠累了還要寫測試，還好我沒用（？）

:::tip 2024/11/19 更新

隨著使用時間拉長也檢查到了更多問題，這裡提供這些工具實際上幫助到我的案例。

最大宗的是 mypy，檢查到兩大問題：忘記把字串加上 `""` 變成物件名稱，這個物件名稱剛好存在，又剛好和比較的方式永遠相等，被 mypy 發現該處永遠為 True；另一個是永遠不會進入的 if 語句，是在修改某處後另一處會造成的變化，沒有全部修正完成也被 mypy 提醒。還有幾個比較小的問題不記得了，總之滿有用的，尤其是程式越來越長之後。

bandit 則教了一些安全技巧，例如使用 `subprocess.Popen` 中有一個 shell 參數，使用 `shell=True` 代表輸入會直接丟到殼層解析處理，這時如果有人輸入惡意指令，例如 `ls /home/user; rm -rf /important_folder`，只使用一個基本的 `ls` 逃脫開頭 `rm -rf` 檢查，於是 bandit 報錯，尤其是我們知道[文字檢查是檢查不完的](https://www.youtube.com/watch?v=cUTYk-LyWQY)，永遠都有想不到的狀況，與其擔心出現問題不如一開始就不要用。

新加入的 pyupgrade 則是偶爾會提醒有更簡潔語法，避免使用舊版語法。

先前沒有提到的 PyPI 發布也很好用，只要設定好 Github Workflow 之後，每次 push tag 都會觸發版本發布，誰還要記得 poetry build 等落落長的指令。

:::
