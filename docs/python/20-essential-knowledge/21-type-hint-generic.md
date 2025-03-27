---
title: Python Type Hint 型別註解教學 - 泛型
sidebar_label: 型別註解 - 泛型
slug: /type-hint-generic
tags:
  - Python
  - typing
  - type hint
keywords:
  - Python
  - typing
  - type hint
last_update:
  date: 2025-03-27T18:52:00+08:00
  author: zsl0621
first_publish:
  date: 2024-11-03T18:00:33+08:00
---

本文介紹如何在 Python 中使用泛型，前一篇文章的[高階使用](type-hint#高階使用)已經很少用了，這裡更用不到。

## 泛型

重點終於來了！最開始原本只想寫這裡的，不過現在反而變成可跳過了，因為連 mypy 這個專門做靜態檢查的工具都表示[跳過泛型完全沒差](https://mypy.readthedocs.io/en/stable/generics.html)。

:::tip 什麼是泛型？

用於創建可以操作多種型別的函式或類別，無需針對每種特定型別重複實作。這使得程式碼更具彈性、可重複利用，同時保證型別安全。當使用泛型時，可以讓類別或函式在實例化或調用時根據指定的型別參數進行型別推斷，從而在編譯或靜態檢查階段檢測不符合型別的操作。

一句話解釋：實例化後才決定使用的變數，避免重複撰寫相似程式，同時確保型別安全。

:::

### TypeVar

Python 中的泛型，用於標示尚未決定的型別，實例化該「函式」後就可以限制 `T` 只能使用相同的型別。

```py
from typing import List, TypeVar

T = TypeVar("T")


def concat(a: List[T], b: T) -> List[T]:
    return a + [b]  # 把 b 合併進 list a


# 故意混用不同型別
numbers: List[int] = [1, 2, 3]
text: str = "hello"

# mypy 會在這行報錯，因為 T 不能同時是 int 和 str
# highlight-next-line
result = concat(numbers, text)  # error: Cannot infer type argument 1 of "concatenate"  [misc]
print(result)   # [1, 2, 3, 'hello']
```

TypeVar 也有參數，使用方式為

#### bound

```py
T = TypeVar("T", int, str)  # 限制只能使用 int/str
T = TypeVar("T", int, str, bound=MyClass)  # 限制只能使用 int/str 和 MyClass 的子類別
```

#### covariant

設定使否使用協變，協變代表允許子類型替代父類型

```py
T_co = TypeVar('T_co', covariant=True)
```

#### contravariant

設定使否使用逆變，逆變代表允許父類型替代子類型

```py
T_co = TypeVar('T_co', contravariant=True)
```

<details>
<summary>由於協變逆變筆者也是第一次接觸，所以請語言模型生成了一個範例</summary>

> ChatGPT 4o#2024/11

協變與逆變在型別系統中決定了如何處理繼承關係中的類型約束，特別在泛型和繼承類別中的參數傳遞上起著至關重要的作用。Python 透過 `typing` 模組支持協變（covariant）與逆變（contravariant），特別用於類型參數的方向性控制，使我們能更靈活地進行類型檢查和靜態分析。這裡提供一個清晰的範例，展示設定協變與逆變如何影響程式的行為和安全性。

<h5>協變與逆變的概念</h5>
1. **協變（Covariant）**：允許子類型替代父類型，例如，`List[Dog]` 可以視為 `List[Animal]`。
2. **逆變（Contravariant）**：允許父類型替代子類型，例如，一個接受 `Animal` 的函數可以用來處理 `Dog`。
3. **不變（Invariant）**：完全相同類型，無法替代。

<h5>範例概述</h5>
假設我們有一個 `Animal` 類別和它的子類 `Dog`。`Animal` 是所有動物的基類，`Dog` 是 `Animal` 的子類。這裡展示協變與逆變的設定如何影響類別之間的兼容性。

<h5>1. 基本設置</h5>
首先，定義基類和子類：

```python
from typing import TypeVar, Generic, Protocol

class Animal:
    def speak(self) -> str:
        return "Generic animal sound"

class Dog(Animal):
    def speak(self) -> str:
        return "Woof"
```

<h5>2. 使用協變的範例</h5>
假設我們有一個 `Cage` 類別，用來裝動物，並希望設置它為協變類型，這樣可以接受 `Animal` 或任何 `Animal` 的子類。

```python
# 協變型別變數
T_co = TypeVar("T_co", bound=Animal, covariant=True)

class Cage(Generic[T_co]):
    def __init__(self, animal: T_co):
        self.animal = animal

    def get_animal(self) -> T_co:
        return self.animal
```

這樣設計使 `Cage[Dog]` 被視為 `Cage[Animal]` 的一個子類，因此可以安全地傳遞 `Dog`：

```python
dog_cage: Cage[Animal] = Cage(Dog())  # 這是允許的，因為 Cage 是協變的
print(dog_cage.get_animal().speak())  # 輸出 "Woof"
```

<h5>3. 使用逆變的範例</h5>
假設我們有一個 `Trainer` 類別，它可以訓練 `Animal` 類別或任何 `Animal` 的子類。為了允許更泛化的訓練對象，我們設置 `Trainer` 類別的型別變數為逆變。

```python
# 逆變型別變數
T_contra = TypeVar("T_contra", bound=Animal, contravariant=True)

class Trainer(Generic[T_contra]):
    def train(self, animal: T_contra) -> str:
        return f"Training {animal.speak()}"
```

這樣設計使得 `Trainer[Animal]` 可以被用於 `Dog`，但不能相反。這允許我們將更廣義的 `Animal` 訓練員用於特定類型的動物。

```python
trainer: Trainer[Dog] = Trainer[Animal]()  # 允許，因為 Trainer 是逆變的
print(trainer.train(Dog()))  # 輸出 "Training Woof"
```

<h5>小結</h5>
- **協變（covariant）**：允許更具體的類別被用在廣義類型的位置，通常適用於輸出情境（例如 `get_animal`）。
- **逆變（contravariant）**：允許更廣義的類別被用在具體類型的位置，通常適用於輸入情境（例如 `train`）。

此範例展示了協變和逆變在泛型類型設計中對型別約束的控制：協變允許子類的替代，而逆變允許父類的替代。
</details>

### Generic

Python 中的泛型，用於標示尚未決定的型別，實例化該「類別」後就可以限制只能使用相同的型別。

```py
from typing import Generic, TypeVar, List

T = TypeVar("T")


# Python 3.12: `class Box[T]`
class Box(Generic[T]):
    def __init__(self):
        self.items: List[T] = []

    def add(self, item: T) -> None:
        self.items.append(item)


box = Box[int]()  # 實例化
box.add(1)  # 正確
box.add("string")  # 錯誤
print(box.items)
```

## 實戰泛型：複寫抽象方法

前面都是使用「設定型別 -> 實例化變數」作為範例方便快速理解，這裡我們考慮一個實際情況：當我們使用 <u>**父類別設定模版**</u>，但是 <u>**子類別的實作輸出卻不同型別**</u>。

以下程式碼中，我們想要限制 `LinkType` 是某幾種特定的型別，使用抽象方法並且讓子類別繼承父類別，子類別可以選擇父類別中的任何一種 Link 作為變數型別。第一次嘗試時沒想這麼多，直接宣告

1. 父類別使用 `LinkType` 限制子類別的變數型別
2. 子類別輸出為 `AlbumLink` `ImageLink` ，與父類別型別不同

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

AlbumLink: TypeAlias = str
ImageLink: TypeAlias = tuple[str, str]
LinkType = TypeVar("LinkType", AlbumLink, ImageLink)


class BaseScraper(ABC):
    @abstractmethod
    # highlight-next-line
    def process_page_links(self, page_links: list[str]) -> list[LinkType]:
        """在父類別中使用包含兩種型別的型別變數LinkType"""


class AlbumScraper(ScrapingStrategy):
    # highlight-next-line
    def process_page_links(self, page_links: list[str]) -> list[AlbumLink]:
        """第一個子類別會輸出其中一種型別"""
        page_result = []
        for link in page_links:
            page_result.append(link)
        return page_result


class ImageScraper(ScrapingStrategy):
    # highlight-next-line
    def process_page_links(self, page_links: list[str]) -> list[ImageLink]:
        """第二個子類別輸出另外一種型別"""
        page_result = []
        for link in page_links:
            page_result.append((link, "after_some_process"))
        return page_result


links = ["http://example.com/1"]

a = AlbumListStrategy().process_page_links(links)
b = AlbumImageStrategy().process_page_links(links)
print(a)
print(b)

"""
# python test.py
['http://example.com/1']
[('http://example.com/1', 'after_some_process')]

# mypy --strict test.py
error: Return type "list[str]" of "process_page_links" incompatible with return type "list[LinkType]" in supertype "ScrapingStrategy"  [override]
error: Return type "list[tuple[str, str]]" of "process_page_links" incompatible with return type "list[LinkType]" in supertype "ScrapingStrategy"  [override]
"""
```

這個錯誤的原因有兩個。第一，父類別沒有使用 Generic，導致 `page_result` 的 type hint 的 scope 只存在於該 method 而不是整個 class，mypy 無法透過繼承追蹤型別，所以警告我們出現 override 錯誤；第二，當我們將 `Generic[LinkType]` 加入父類別後再執行檢查會成功，不過此時如果我們使用更嚴格的 --strict 參數進行檢查，會出現 `error: Missing type parameters for generic type "ScrapingStrategy"  [type-arg]`，解決此問題的方式是在繼承時指定使用哪種型別即可解決問題。

修正結果如下，只需在 class 繼承時額外指定該 class 的 type。

```py
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

AlbumLink: TypeAlias = str
ImageLink: TypeAlias = tuple[str, str]
LinkType = TypeVar("LinkType", AlbumLink, ImageLink)


# highlight-next-line
class BaseScraper(Generic[LinkType], ABC):
    """Abstract base class for different scraping strategies."""

    @abstractmethod
    def process_page_links(self, page_links: list[str]) -> list[LinkType]:
        """Process links found on the page."""


# highlight-next-line
class AlbumScraper(BaseScraper[AlbumLink]):
    def process_page_links(self, page_links: list[str]) -> list[AlbumLink]:
        page_result = []
        for link in page_links:
            page_result.append(link)
        return page_result


# highlight-next-line
class ImageScraper(BaseScraper[ImageLinkAndALT]):
    def process_page_links(self, page_links: list[str]) -> list[ImageLink]:
        page_result = []
        for link in page_links:
            page_result.append((link, "after_some_process"))
        return page_result


links = ["http://example.com/1"]

a = AlbumListStrategy().process_page_links(links)
b = AlbumImageStrategy().process_page_links(links)
print(a)
print(b)

"""
# mypy --strict test.py
Success: no issues found in 1 source file
"""
```

筆者也是遇到這個問題才去看 Generic 的。

### 還有其他方法嗎？

當我們想到一個解決方法後，下一步就是問自己有沒有更好的解決方法，這裡筆者自行檢討了幾種不同的 type hint 方式：

1. Generic: 原本的 Generic 提供了繼承的功能，這是其他方式做不到的。
2. Protocol: 只提供類似 ABC 抽象類別的功能，並沒有針對輸出輸入型別限制，功能完全不同。
3. Union: 更鬆散的 type hint，使用輸出的變數會被 IDE 提醒型別不一致。
4. Override: 可以用，但是不符合直覺，邏輯上 abstractmethod 的子類應該遵守父類定義，可是子類的實現卻又 override 父類。
5. Overload: 用途不同，在同一個 scope 中有多個同名函式或方法才使用此裝飾器

綜合以上幾點，直接使用 Generic 在此情境中是最佳解，而且使用 override 仍需要在父類繼承 Generic，所以直接使用 Generic 是更直觀且方便的方式。

:::info 請 Claude 3.5 Sonnet#2024/11 做的總結：

結論：

**Override 的問題：**

1. AI 直接沒寫這段所以我自己寫，缺點就是上面寫的不符合直覺，邏輯上 abstractmethod 的子類應該遵守父類定義，可是子類的實現卻又 override 父類。

**Overload 的問題：**

1. Code Organization：
   - 需要為每種可能的型別組合寫一個 overload
   - 維護成本高，且容易出錯

2. Design Intent：
   - Overload 主要用於同一個函數處理不同型別參數的情況
   - 不適合用於表達 class hierarchy 中的型別關係

**Generic 是此案例的最佳解決方案**

  1. Type Safety：提供完整的型別安全性
  2. Design Clarity：清楚表達設計意圖
  3. Maintainability：容易維護和擴展
  4. Compile-time Checking：提供編譯時期的型別檢查

使用 Generic 能最好地表達：「這是一個可以處理不同型別的策略，但每個具體策略實現都需要指定並遵守其處理的特定型別」。
:::

## 相關工具

雖然只有兩行但是很重要，所以獨立一個段落。

如果要最大化發揮 type hint 功效則需要結合 [mypy/pylint 等套件](https://www.trackawesomelist.com/typeddjango/awesome-python-typing/readme/#static-type-checkers) 使用，建議直接整合 pre-commit hooks 使用，請參考文章 [初嘗 Python 工作流自動化](/memo/python/pre-commit-first-try)。

## 結語

本文解釋了沒什麼人講過的 Generic，使用 type hint 時需要自行衡量標注的完整程度和程式開發的方便程度，寫的太完整會導致開發中需要不斷處理各種型別，失去 Python 快速開發的意義。

## 參考資料

- [用代码打点酱油的chaofa - Python 类型体操训练](https://bruceyuan.com/post/python-type-challenge-basic.html)  
- [the maintainer of pyright closes valid issues for no reason and lashes out at users](https://docs.basedpyright.com/latest/) 超好笑，我從來沒在 Github 上看過[兩百個倒讚](https://github.com/microsoft/pyright/issues/8065#issuecomment-2146352290)  
