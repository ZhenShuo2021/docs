---
title: 理解 Python 的導入機制和順序
sidebar_label: 導入機制和順序
description: 這篇文章專注說明 Python 的導入順序 (import order)，說明 Python 是如何解析導入語句的，雖然不是非常重要，但是能讓你理解這些東西到底是怎麼運作的，避免似懂非懂問題。
slug: /import-system
tags:
  - Python
keywords:
  - Python
last_update:
  date: 2025-03-27T18:52:00+08:00
  author: zsl0621
first_publish:
  date: 2025-03-26T02:30:00+08:00
---

> 重要程度：2/5

這篇文章專注說明 Python 的導入順序 (import order)，說明 Python 是如何解析導入語句的，雖然不是非常重要，但是能讓你理解這些東西到底是怎麼運作的，避免似懂非懂。

:::info

在 Python 文檔 3.10 後 import 就從匯入改為引入了，簡中則是一直都用導入。

:::

## Python 的導入機制

以下是 Python 導入的基本步驟：

1. 檢查 `sys.modules`  
Python 首先檢查 sys.modules，這是一個儲存已導入模組的字典。如果模組已經被導入，就直接使用快取的版本，避免重複載入。

:::info
Python 不會重複導入同一個模組。
:::

2. 搜尋模組路徑  
如果模組不在 `sys.modules` 中，Python 會根據 `sys.path` 的順序尋找模組。`sys.path` 是一個列表，包含以下來源（按順序）：
目前腳本所在的目錄（或當前工作目錄）。

   - PYTHONPATH 環境變數中指定的路徑。
   - Python 安裝時的標準庫路徑。
   - 虛擬環境中的 site-packages 路徑。
   - 自定的路徑（例如透過 `sys.path.append` 手動添加）。

3. 載入模組或套件  
找到模組後，Python 會執行該模組的程式碼。如果是套件，則會先**執行**該套件的 \_\_init\_\_.py 檔案。

:::info
別忘了 \_\_init\_\_.py 的檔案會被完整執行。
:::

4. 綁定名稱  
最後，導入的模組或套件會被[綁定](https://docs.python.org/zh-cn/3.13/reference/executionmodel.html)到指定的名稱空間中，例如 import package_A 會將 package_A 加入當前名稱空間。

:::info
可以用 import sys; print(sys.path) 檢查當前 Python 的搜尋路徑順序。
:::

具體使用範例如下，複製貼上後直接執行即可

```py
import sys
import os

# 檢查 sys.modules
print("=== Check sys.modules ===")
print("Before import 'os':", "os" in sys.modules)  # True
print("Before import 'math':", "math" in sys.modules)  # False

import math  # 導入 math
print("After import 'math':", "math" in sys.modules)  # True

# 搜尋模組路徑 (sys.path)
print("\n=== Search sys.path ===")
print("Current sys.path:")
for i, path in enumerate(sys.path[:3], 1):  # 只顯示前三個路徑
    print(f"  {i}. {path}")
print("  ... (and more)")

# 增加自定路徑
sys.path.append("/custom/path")
print("After appending custom path:", sys.path[-1])

# sys.path 第一個路徑應該是當前目錄
print("\n=== Check sys.path with import sys ===")
print("First path in sys.path:", sys.path[0])
```

:::tip
除此之外你也可以在前一篇文章的每個檔案上方加上 `print("代號")` 觀察載入順序，例如 `uv run -m src.package_A` 和 `uv run calculation-cli` 會發現兩個的載入不完全一樣。
:::

## 循環導入

循環導入（Circular Import）是 Python 專案中常見的問題，例如 `package_A.calculator` 導入 `package_B.data_processor`，而 `package_B.data_processor` 又導入 `package_A.calculator`。這會導致 Python 在載入模組時報錯：
`ImportError: cannot import name 'xxx' from partially initialized module 'xxx'`

解決方法：

- 重構程式碼：基本解決方式，把共用邏輯抽取到第三個模組
- 延遲導入：把導入語法放到函式中而不是模組頂層。例如：

```py
# src/package_A/calculator.py
from typing import List

def advanced_calculation(numbers: List[float]) -> float:
    from package_B.data_processor import process_numbers  # 延遲導入
    processed_numbers = process_numbers(numbers)
    return sum(processed_numbers)
```
