---
title: 建立符合現代標準的專案結構
sidebar_label: 專案結構
description: 初學 Python 時大家的專案可能都是直接放在專案根目錄互相 import，不只有 import 關係混亂的問題，也不是 Python 官方建議的專案架構。本文介紹如何建立符合現代標準的 Python 專案，並且說明 package、module、__init__.py 到底是什麼。
slug: /src-layout-vs-flat-layout
tags:
  - Python
  - 專案結構
keywords:
  - Python
  - 專案結構
last_update:
  date: 2025-03-27T18:52:00+08:00
  author: zsl0621
first_publish:
  date: 2025-03-25T21:24:00+08:00
---


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

>重要程度：5/5

初學 Python 時大家的專案可能都是直接放在專案根目錄互相 import，不只有 import 關係混亂的問題，也不是 Python 官方建議的專案架構。本文介紹如何建立符合現代標準的 Python 專案，並且說明 `package`、`module`、`__init__.py` 到底是什麼。

## src-layout vs flat-layout

Python 專案佈局分成 src layout 和 flat layout 兩種佈局，多說無益，看的更好理解，結構如下所示

<Tabs>
  <TabItem value="flat layout" label="flat layout">

```tree
.
├── README.md
├── pyproject.toml
└── awesome_package/
    ├── __init__.py
    └── module.py
```

  </TabItem>

  <TabItem value="src layout" label="src layout">

```tree
.
├── README.md
├── pyproject.toml
└── src/
     └── awesome_package/
        ├── __init__.py
        └── module.py
```

  </TabItem>
</Tabs>

這兩種佈局是 Python 負責封裝的管理機構 [PyPA](https://packaging.python.org/en/latest/) 建議的[專案佈局方式](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)，也只應該使用這兩種佈局。兩者的差異為

1. 執行需求：src 佈局強制要求專案必須先安裝才能運行程式碼，而扁平佈局無需安裝即可直接執行。
2. 匯入安全性：src 佈局在可編輯安裝時，只允許導入 (import) 可導入的檔案，避免意外匯入問題。
3. 避免意外使用開發中程式碼：同上，最重要的是防止在可編輯模式下可用，正常安裝卻不可用的問題。

src 佈局是新的方式，雖然簡單專案兩者並無差異，但是新專案還是建議使用 src 佈局，大部分沒用 src 佈局的專案都是因為原本就已經使用扁平佈局，修改成本太大。

## 建立 src 佈局專案

先不學理論，能動再說。建立專案請使用專案管理工具完成，不要再看網路上的害人教學，2025 還在用過時的 venv 跟 conda，請用目前[最好的專案管理工具](best-python-project-manager) uv，安裝方式請見 [uv 介紹文章](uv-project-manager-1#安裝-uv)。使用 uv 建立 src 佈局專案的方式是

```sh
uv init --package --python=3.10 calculation-data-project
```

這會建立一個 src 佈局、使用 Python 3.10 的專案，並且自動建立好 pyproject.toml 和初始化 Git 版本記錄倉庫。接下來我們建立一個基本專案，建立這些檔案並且貼上程式碼

<Tabs>
  <TabItem value="pyproject.toml">

```toml
[project]
name = "calculation-data-project"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "ZhenShuo Leo", email = "your_git_email@gmail.com" }
]
requires-python = ">=3.10"
# 套件依賴（目前無）
dependencies = []

[tool.hatch.build.targets.wheel]
# 指定要打包的 Python 套件路徑，需要建立此段落才可以執行命令行入口，這個段落不會自動建立需要手動設定
packages = ["src/package_A", "src/package_B"]

[project.scripts]
# 專案命令行入口，會在 src/module/cli.py 中執行 main 函式
calculation-cli = "package_A.cli:main"

[build-system]
# 建置系統所需的工具
requires = ["hatchling"]
# 指定建置後端，前端是 uv，呼叫 hatchling 後端進行 build
build-backend = "hatchling.build"
```

  </TabItem>
  <TabItem value="calculator.py">

```py
# src/package_A/calculator.py
from typing import List
from package_B.data_processor import process_numbers

def advanced_calculation(numbers: List[float]) -> float:
    """
    Perform an advanced calculation using processed numbers.
    
    Args:
        numbers (List[float]): Input numbers to calculate
    
    Returns:
        float: Result of advanced calculation
    """
    processed_numbers = process_numbers(numbers)
    return sum(processed_numbers)
```

  </TabItem>
  <TabItem value="cli.py">

```py
# src/package_A/cli.py
import sys
from package_A.calculator import advanced_calculation

def main():
    """
    CLI entry point for calculation operations.
    
    Usage: calculation-cli 1 2 3 4 5
    """
    try:
        # Convert command-line arguments to float
        numbers = [float(arg) for arg in sys.argv[1:]]
        
        # Perform calculation
        result = advanced_calculation(numbers)
        
        # Print result
        print(f"Calculation Result: {result}")
    
    except ValueError:
        print("Error: Please provide valid numeric arguments")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

  </TabItem>
  <TabItem value="data_processor.py">

```py
# src/package_B/data_processor.py
from typing import List
from package_B.utils import filter_positive_numbers

def process_numbers(numbers: List[float]) -> List[float]:
    """
    Process input numbers using utility functions.
    
    Args:
        numbers (List[float]): Input numbers to process
    
    Returns:
        List[float]: Processed numbers
    """
    # Filter positive numbers and apply additional processing
    filtered_numbers = filter_positive_numbers(numbers)
    return [num * 1.5 for num in filtered_numbers]
```

  </TabItem>
  <TabItem value="utils.py">

```py
# src/package_B/utils.py
from typing import List

def filter_positive_numbers(numbers: List[float]) -> List[float]:
    """
    Filter out non-positive numbers from the input list.
    
    Args:
        numbers (List[float]): Input numbers to filter
    
    Returns:
        List[float]: List of positive numbers
    """
    return [num for num in numbers if num > 0]
```

  </TabItem>
  <TabItem value="package_A/__init__.py">

```py
# package_A/__init__.py
# 建立空白的檔案即可
```

  </TabItem>
  <TabItem value="package_A/__main__.py">

```py
# package_A/__main__.py
from package_A import main

if __name__ == "__main__":
    main()
```

  </TabItem>
  <TabItem value="package_B/__init__.py">

```py
# package_B/__init__.py
# 建立空白的檔案即可
```

  </TabItem>
</Tabs>

---

現在我們已經建立好有兩個 package 的專案，並且 package_A 還有 \_\_main\_\_.py 作為 package_A 自己的入口點，接著我們嘗試執行此專案，由於 src 佈局的關係我們一定要安裝專案才能執行，

```sh
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -e .
```

其中 `-e` 代表可編輯安裝，使用此選項後所有對程式碼的改動都會立刻影響到安裝的套件，如果不使用你就需要每次修改程式碼後手動重新安裝裁可刷新。完成這些步驟後我們終於可以執行腳本

```sh
calculation-cli 5 6
```

這就是傳統運行 Python 專案的流程。但是我們現在有先進的專案管理工具其實不需要這麼麻煩，我們退出虛擬環境並且刪除 `.venv` 目錄重置環境後，只需要執行此指令 uv 就會自動建立虛擬環境、安裝套件並且執行專案：

```sh
uv run calculation-cli 5 6
# 輸出 Calculation Result: 16.5
```

這裡的邏輯是 uv 會自動尋找 pyproject.toml 並且根據裡面的設定建立環境（如果還沒有環境），再找到 `project.scripts` 設定的專案的命令行入口點，也就是說會執行 src/package_A/cli.py 裡面的 main 函式。

:::info
`uv run` 不需進入虛擬環境就可以直接執行虛擬環境中的指令，uv 會自動尋找對應的虛擬環境或是 pyproject.toml。
:::

:::tip
你可以嘗試在每個檔案前面加上 print 測試他們的載入順序。
:::

<br/>

### 什麼是 \_\_init\_\_.py{#\_\_init\_\_.py}

\_\_init\_\_.py 是一個 package 的必要文件，有這個文件 Python 才會把該目錄作為一個 package 解析，這個文件空白的也可以，每次 import 一個 package 第一步就是讀取這個檔案，需要他的原因很簡單，因為 Python 只知道文件在哪個目錄中，不知道他在哪個 package 中，所以需要 \_\_init\_\_.py 告訴 Python 這是一個 package[^1]。

[^1]: [Relative imports for the billionth time](https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time), stackoverflow

- 通常我們會在這個檔案把該 package 的函式 import 進來，這樣別人使用時就不用再 `from package_A.cli import main`，直接使用 `from package_A import main` 就可以了。
- 如果加上 `__all__` 就代表在其他地方使用 `from package_A import *` 的時候會包含在 `__all__` 裡面定義的函式。
- 有些命令行工具也會喜歡在 `__init__.py` 裡面直接設定入口點，例如最知名的影片下載工具 [yt-dlp](https://github.com/yt-dlp/yt-dlp/blob/336b33e72fe0105a33c40c8e5afdff720e17afdb/yt_dlp/__init__.py#L1097) 和圖片下載工具 [gallery-dl](https://github.com/mikf/gallery-dl/blob/0ffef5877958399b8bf4c933ab53a2f6d22e3b8c/gallery_dl/__init__.py#L21) 都是這樣設定。
- 我們也可以把元資料放在這裡面，例如 `__author__` `__license__` 等等，設定範例如下。

```py
# package_A/__init__.py
from package_A.cli import main

__all__ = ["main"]
__author__ = "author name"
__license__ = "MIT License"
```

<br/>

### 什麼是 \_\_main\_\_.py{#\_\_main\_\_.py}

他不是必要的文件，他是可選的 package 入口點，使用此指令會把 package_A 作為腳本執行：

```sh
uv run -m src.package_A 5 6
```

並且尋找 \_\_main\_\_.py 作為主腳本。[-m 代表 module](https://docs.python.org/zh-tw/3.13/using/cmdline.html#cmdoption-m)，後面需要跟著一個 module，在此是 src 目錄中的 package_A。

:::info
此指令等效於進入虛擬環境後使用 `python3 -m src.package_A`，通常沒必要這樣用直接使用 `uv run` 即可。
:::

:::tip
在 \_\_init\_\_.py 加上一個 print，試試看 `uv run -m src.package_A 5 6` 和 `uv run calculation-cli 5 6`。
:::

<br/>

### 什麼是 package 和 module{#package-and-module}

- module: 任何副檔名是 py 的 Python 檔案
- package: 一堆 module 的集合，也就是說有一堆 py 檔案的目錄，但是裡面**必須包含 \_\_init\_\_.py**
- \_\_init\_\_.py: 告訴 Python 這是一個 package，可以是空白檔案
- \_\_main\_\_.py: 如果直接執行 package ，會把他作為主函式執行（使用 `python -m src.package_A` 時會尋找此檔案）

也就是說在剛才的範例中，我們有兩個 package，並且必須使用 \_\_init\_\_.py 才能讓 Python 將他辨識為 package，每個 package 裡面有兩個 module。

在 Python 導入 package 時，該 package 的 \_\_init\_\_.py 也會被執行。

<br/>

### 相對導入還是絕對導入{#import}

```py
# 相對導入
from . import module1
from .utils import util

# 絕對導入
from package import module1
from package.utils import function1
```

只建議絕對導入，可避免同名問題並且對 CI 友好，dask 在[討論](https://github.com/dask/distributed/issues/5889)後也改為絕對導入了。

## 什麼是 \_\_name\_\_ == "\_\_main\_\_"{#\_\_name\_\_-\_\_main\_\_}

常常會看到這種語法，例如 `cli.py` 就有出現這個，究竟是什麼意思呢？

```py
if __name__ == '__main__':
    main()
```

簡短說明：目的是避免特定程式在 import 時被執行。

長一點的說明：目的是區分一個 Python 檔案（模組）是被「直接執行」還是被「作為模組導入」到其他程式中，Python 執行時會為每個文件建立 `__name__` 屬性，當文件是頂層腳本，並且被直接執行時會設定為 `__main__`，否則為模組的名稱，以此區別主程式和被引入的程式。

## 延伸閱讀

- [《面试官一个小时逼疯面试者》之聊聊Python Import System？](https://zhuanlan.zhihu.com/p/348559778)
- import 在 Python 文檔 3.10 後就從匯入改為引入了，簡中則是一直都用導入
- 關於專案的知識網路上好像沒什麼文章寫，可能是知道的沒時間寫，不知道的寫不出來，所以又是我來寫啦。
