---
title: 命名規範和輔助工具
description: 遵循命名規範非常重要，可以幫助你一眼就知道這個變數的類型，光是命名就可以包含更多資訊增加可讀性，減少無謂的思考開銷，說這些不影響的人完全就是亂講，這是重中之重，也是基礎中的基礎。
sidebar_label: 命名規範和輔助工具
slug: /naming-conventions
tags:
  - Python
keywords:
  - Python
last_update:
  date: 2025-03-27T18:52:00+08:00
  author: zsl0621
first_publish:
  date: 2025-03-26T23:27:00+08:00
---

>重要程度：5/5

遵循命名規範非常重要，可以幫助你一眼就知道這個變數的類型，光是命名就可以包含更多資訊增加可讀性，減少無謂的思考開銷，說這些不影響的人完全就是亂講，這是重中之重，也是基礎中的基礎。

因為網路上已經非常多類似教學所以本文不會深入細節，這篇文章主要目的是作為一個入口點，讓讀者能在同一個系列文章中系統性學習，詳細說明請見我老大的影片：[和python开发人员用同一套命名系统，一期视频就学会！](https://www.youtube.com/watch?v=x6I8x-40w6k)

## 命名規範

1. **常數**：大寫 `SNAKE_CASE`
   - 全大寫
   - 單詞之間用下劃線分隔
   - 範例：`MAX_CONNECTIONS`、`DEFAULT_TIMEOUT`

2. **變數、函數名稱**：小寫 `snake_case`
   - 全小寫
   - 單詞之間用下劃線分隔
   - 範例：`user_name`、`calculate_total`

3. **類別名稱**：`PascalCase`
   - 每個單詞首字母大寫
   - 單詞之間不使用下劃線
   - 範例：`UserProfile`、`DatabaseConnection`

4. **私有/受保護變數**：
   - 單下劃線前綴：`_private_var`（約定俗成的私有，還是可以訪問他）
   - 雙下劃線前綴：`__very_private`（觸發 name mangling）

5. **魔術方式**：`__xxx__`
   - 需要此文章的讀者就不要用他

### 範例

```python
class UserAuthentication:  # 類別名 PascalCase
    MAX_LOGIN_ATTEMPTS = 3  # 常數 SNAKE_CASE

    def __init__(self, username: str):  # 型別提示：輸入是 str
        self._current_user = username  # 弱私有，僅提醒
        self.__session_token = None   # 強私有，觸發 name mangling

    def validate_credentials(self, password: str) -> bool:  # 型別提示：輸入是 str，輸出 bool
        login_attempt = 0  # Variable names use snake_case
        return self._check_password(password)

    def _check_password(self, password: str) -> bool:  # 型別提示：輸入是 str，輸出 bool
        pass
```

設定完型別提示之後 IDE 就會自動啟用補全功能，mypy 等檢查工具也可以檢查是否有錯誤的變數使用。

## 其他風格細節

1. **縮排**：固定使用 4 個空格
2. **長度**：最大 79 字元，這對現代專案過少，通常會改成 100 或者 112
3. **匯入順序**：
   - 內建標準庫
   - 第三方庫
   - 本地模組
   - 所有庫都依照字母順序排列
4. **docstring風格**：分成 [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), [Numpy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) 以及 [Sphinx style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)，個人推薦 Google style，Sphinx 密密麻麻讀起來很彆扭。

Google docstring 範例如下

```python
class UserAuthentication:
    """Handles user authentication, including validation of credentials.

    This class allows for user authentication by validating the credentials provided
    and checking the password against stored credentials. It also manages login attempts
    and session token generation.

    Attributes:
        MAX_LOGIN_ATTEMPTS (int): The maximum number of login attempts allowed.
        _current_user (str): The username of the currently authenticated user.
        __session_token (str or None): The session token for the authenticated user.
    """
    MAX_LOGIN_ATTEMPTS = 3

    def __init__(self, username: str):
        self._current_user = username
        self.__session_token = None

    def validate_credentials(self, password: str) -> bool:
        """Validates user credentials.

        This function checks the provided password against the stored credentials.

        Args:
            password (str): The password to be validated.

        Returns:
            bool: True if the password is correct, False otherwise.
        """
        login_attempt = 0
        return self._check_password(password)

    def _check_password(self, password: str) -> bool:
        """Private method to check the password.

        This method is intended to be used internally to verify the password.

        Args:
            password (str): The password to be checked.

        Returns:
            bool: True if the password is correct, False otherwise.
        """
        pass
```

完成後只要游標移動到名稱上，IDE 就會顯示對應的 docstring

![docstring-example](https://cdn.zsl0621.cc/2025/docs/docstring---2025-04-27T16-18-34.webp)

## 型別提示

型別提示是 Python **非常重要**的一個功能，Python 是動態語言，型別提示幫他**無痛的**帶來了靜態語言的優點，同時又保持原本動態語言的靈活性。

初學者應該是看不懂動態靜態在講什麼，說人話的解釋是：

1. **提高可讀性**：開發者可以清楚知道函數或變數預期的資料型別，不需要額外翻閱文件或原始碼。
2. **減少錯誤**：型別提示可以搭配 mypy 等工具檢查型別不同的錯誤。
3. **工具整合**：現代 IDE 可以利用型別提示提供**自動完成和建議**（如 PyCharm、VS Code，Spyder 的型別提示可以當作沒有）。

如何使用型別提示請見[型別提示 - 基礎篇](type-hint)。

## 相關工具

格式化工具 (formatter) 可以幫助我們完成符合規範的程式風格，不必再手動慢慢排。這裡也順帶介紹其他檢查工具，這些工具能極大幅度減少程式錯誤，一定要使用這些工具，包含

1. **格式化工具 formatter**: 自動調整程式碼格式，統一規範。
2. **程式風格檢查 linter**: 確保程式碼符合風格指南，減少可讀性問題與潛在錯誤。  
3. **靜態檢查 static analysis**: 在不執行程式的情況下分析程式碼結構，檢查類型錯誤、未使用變數等問題。  
4. **（可選）安全性檢查**: 掃描程式碼中的潛在安全漏洞，例如 SQL 注入、XSS 或其他常見攻擊風險。
5. **（必選）提交鉤子 commit hook**: 在每次提交時自動執行指定任務。

### linter 和 formatter{#linter-formatter}

以前 linter 和 formatter 工具是[兵荒馬亂什麼都有](https://blog.kyomind.tw/ruff/)，在 ruff 推出之後建議一律使用 ruff，原因是 ruff 開宗明義說了**目標是優化速度而不是新增功能**，所以別人有的功能他全都有，還不是只有一兩個工具，pylint, pyflakes, pycodestyle 等等 ruff 全部支援而且速度更快。

上面的都屬於 linter，除此之外 ruff 也可以設定 formatter，建議風格為 isort + Black 格式。

:::tip
Black formatter 是最嚴格死板也是最不用煩惱的格式，使用 autopep8（PEP 8）太寬鬆基本上不會改什麼，yapf 個人研究不多不多做評論。
:::

### Static Analysis

靜態檢查工具只有三大家 [mypy](https://github.com/python/mypy)、[Pyright](https://github.com/microsoft/Pyright) 和 [basedpyright](https://github.com/DetachHead/basedpyright)，mypy 是元老，Pyright 是微軟出的工具，basedpyright 是 [DetachHead 在 pyright 發 issue 結果被 core dev 嗆](https://github.com/microsoft/pyright/issues/8065#issuecomment-2146352290)之後不爽自己出來開的分支，這幾個任意選擇都可以，最大不同是解析理念的差異和速度，mypy vs pyright 兩個系列之間的比較請見 [Mypy comparison](https://docs.basedpyright.com/latest/usage/mypy-comparison/)，兩個 pyright 之間的比較請見 [Benefits over pyright](https://docs.basedpyright.com/latest/benefits-over-pyright/baseline/)。

![star-history-python-static-analysis](https://cdn.zsl0621.cc/2025/docs/star-history-python-static-analysis---2025-04-27T16-18-34.webp)

:::tip
basedpyright 原本還直接在[官網第一頁就開嗆 Pyright core dev](https://docs.basedpyright.com/v1.18.1/)，現在他們好像和好了，有點可惜少了一個 drama 可以看XD
:::

### commit hook

commit hook 也是絕對必要的東西，因為上述工具沒有整合，我們就得一個一個執行或者自己寫腳本非常麻煩，commit hook 則是幫助你在每次提交的時候自動執行指定任務。使用方式很簡單，設定完 `.pre-commit-config.yaml` 後執行 `pre-commit install` 就完成安裝，之後每次提交都會自動執行指定任務，常用指令有

- `pre-commit run -a` 手動執行全部指定任務
- `pre-commit run <hook_id>` 只執行指定任務
- `SKIP=<hook_id> pre-commit run -a` 跳過指定任務
- `pre-commit autoupdate` 自動升級所有 hook

除了前面說到的工具以外，也可以設定執行程式測試、覆蓋率測試、自動檢查空白鍵、檢查是否有金鑰密碼等敏感訊息意外被提交，非常方便。

:::info
提交是 Git 版本管理的動作，筆者有寫[完整的 Git 版本管理教學](/git/introduction)教你如何使用。
:::

## 結論

這些規則讓你寫出「好讀」的程式碼而不是「好」的程式碼，「好」的程式碼是非常複雜的課題，而且每個人標準不一樣，但是「好讀」在公認的基礎上絕對不會分歧，程式碼符合規範是必須且無庸置疑的。

## 參考資料

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
