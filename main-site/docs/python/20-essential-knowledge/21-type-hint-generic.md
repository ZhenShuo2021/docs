---
title: Python Type Hint 型別提示教學 - 泛型
sidebar_label: 型別提示 - 泛型
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

重點終於來了！最開始只想寫泛型，不過現在反而變成可跳過了，因為連 mypy 這個專門做靜態檢查的工具都表示[跳過泛型完全沒差](https://mypy.readthedocs.io/en/stable/generics.html)。本文介紹如何在 Python 中使用泛型並且提供範例，你可以複製後使用 `mypy <檔案名稱>` 檢查型別。

## 泛型

:::tip 什麼是泛型？

用於創建可以操作多種型別的函式或類別，無需針對每種特定型別重複實作。這使得程式碼更具彈性、可重複利用，同時保證型別安全。當使用泛型時，可以讓類別或函式在實例化或調用時根據指定的型別參數進行型別推斷，從而在編譯或靜態檢查階段檢測不符合型別的操作。

一句話解釋：**允許在實例化後才決定使用的變數型別**，避免重複撰寫相似程式，同時確保型別安全。

:::

也就是說，當某個函式或類別可以處理不同型別的資料，且**每次使用需要保持一致的型別不允許混用時**，就需要用到泛型。

### TypeVar

泛型「函式」的型別提示，實例化該函式後就可以限制 `T` 只能使用相同的型別。

```py
from typing import List, TypeVar

T = TypeVar("T")


# 把 b 合併進 list a
def concat(a: List[T], b: T) -> List[T]:
    return a + [b]


# 故意混用不同型別
numbers: List[int] = [1, 2, 3]
text: str = "hello"

# mypy 會在這行報錯，因為 T 不能同時是 int 和 str
# error: Cannot infer type argument 1 of "concatenate"  [misc]
result = concat(numbers, text)
print(result)  # [1, 2, 3, 'hello']
```

### Generic

Python 中的泛型，用於標示尚未決定的型別，實例化該「類別」後就可以限制只能使用相同的型別。

```py
from typing import Generic, TypeVar, List

T = TypeVar("T")


# Python 3.12 新的等效語法: `class Box[T]`
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

#### bound

bound 參數限制泛型只能給指定型別而不是任意型別。

```py
T = TypeVar("T", int, str)  # 限制只能使用 int/str
T = TypeVar("T", int, str, bound=MyClass)  # 限制只能使用 int/str 和 MyClass 的子類別
```

#### covariant

設定使用協變，允許**子類替代父類**。例如動物園的餵食系統中，定義一個泛型類別 `Feeder`，用來餵食某種動物。現在有 `Animal` 父類和 `Cat` 子類，協變告訴型別系統子類 `Feeder[Cat]`（專門餵貓的餵食器）可以被當作父類 `Feeder[Animal]`（餵動物的餵食器）使用。

```py
from typing import Generic, TypeVar

"""
covariant: 設定子類別可替代父類別，不使用 covariant 會造成
           子類無法替代父類，造成 Incompatible types error

bound: 限制只能使用 Animal 這個型別，不使用 bound 會造成
       所有類型都可輸入，造成 make_sound no attribute error
"""
T_co = TypeVar("T_co", bound="Animal", covariant=True)


class Animal:
    def make_sound(self) -> str:
        return ""


class Cat(Animal):
    def make_sound(self) -> str:
        return "Meow"


class Feeder(Generic[T_co]):
    def __init__(self, animal: T_co):
        self.animal = animal

    def feed(self) -> str:
        return f"Feeding {self.animal.make_sound()}"


cat_feeder: Feeder[Cat] = Feeder(Cat())
animal_feeder: Feeder[Animal] = cat_feeder
```

在這個範例裡面你可以測試把 TypeVar 裡面的 bound 或 covariant 移除掉，都會出現錯誤，原因是沒有 bound 他就不確定輸入是不是有 `make_sound` 這個方法，沒有 covariant 他就沒辦法替代父類。

#### contravariant

設定使用逆變，允許父類替代子類。比如說一個動物訓練系統，定義一個泛型類別 `Trainer` 用來訓練動物，因為能訓練所有動物的訓練師自然也能訓練狗，所以假設有 `Animal` 父類和 `Dog` 子類，逆變允許一個 `Trainer[Animal]`（能訓練所有動物的訓練師）被當作 `Trainer[Dog]`（訓練狗的訓練師）使用。

```py
from typing import Generic, TypeVar

T_contra = TypeVar("T_contra", bound="Animal", contravariant=True)


class Animal:
    def perform(self) -> str:
        return "Parent animal class"


class Dog(Animal):
    def perform(self) -> str:
        return "Bark"


class Trainer(Generic[T_contra]):
    def __init__(self, animal: T_contra):
        self.animal = animal

    def train(self) -> str:
        return f"Training {self.animal.perform()}"


# 使用範例，建立一個父類
animal_trainer: Trainer[Animal] = Trainer(Animal())  # 用 Animal 實例初始化

# 逆變，子類型別被父類賦值 (變數設定型別 Dog, 被變數 Animal 賦值)
dog_trainer: Trainer[Dog] = animal_trainer

# 測試
print(dog_trainer.train())
```

## 實戰泛型：抽象方法 abstractmethod

考慮一個常見的實際情況：當我們使用 <u>**父類別設定模版，但是子類別的實作輸出允許輸出不同型別**</u>，這時應該如何加上型別提示呢？下方的範例中 `BaseScraper` 是模板父類，所有子類都必須實現 `process_page_links` 方法，而且允許每個子類的輸出型別不同。

### 第一次的錯誤嘗試

以下程式碼中，我們想要限制 `LinkType` 是某幾種特定的型別，使用抽象方法並且讓子類別繼承父類別，子類別可以選擇父類別中的任何一種 Link 作為變數型別。第一次嘗試時很直覺的這樣宣告

1. 父類別使用 `LinkType` 限制子類別的變數型別
2. 子類別輸出為 `AlbumLink` `ImageLink` ，與父類別型別不同

```python title="第一次嘗試將抽象方法加上型別提示"
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

AlbumLink: TypeAlias = str
ImageLink: TypeAlias = tuple[str, str]

# LinkType 限制只有 AlbumLink 和 ImageLink 兩種類型
# highlight-next-line
LinkType = TypeVar("LinkType", AlbumLink, ImageLink)


class BaseScraper(ABC):
    # 父類別設定 LinkType
    @abstractmethod
    # highlight-next-line
    def process_page_links(self, page_links: list[str]) -> list[LinkType]:
        """在父類別中使用包含兩種型別的型別變數LinkType"""


class AlbumScraper(BaseScraper):
    # 子類別設定對應型別
    # highlight-next-line
    def process_page_links(self, page_links: list[str]) -> list[AlbumLink]:
        """第一個子類別會輸出其中一種型別"""
        page_result = []
        for link in page_links:
            page_result.append(link)
        return page_result


class ImageScraper(BaseScraper):
    # 子類別設定對應型別
    # highlight-next-line
    def process_page_links(self, page_links: list[str]) -> list[ImageLink]:
        """第二個子類別輸出另外一種型別"""
        page_result = []
        for link in page_links:
            page_result.append((link, "after_some_process"))
        return page_result


links = ["http://example.com/1"]

a = AlbumScraper().process_page_links(links)
b = ImageScraper().process_page_links(links)
print(a)
print(b)


"""
# 輸出如下
# python test.py
['http://example.com/1']
[('http://example.com/1', 'after_some_process')]

# mypy --strict test.py
error: Return type "list[str]" of "process_page_links" incompatible with return type "list[LinkType]" in supertype "BaseScraper"  [override]
error: Return type "list[tuple[str, str]]" of "process_page_links" incompatible with return type "list[LinkType]" in supertype "BaseScraper"  [override]
"""
```

<br/>

這裡的錯誤是父類沒有使用 Generic，導致 `page_result` 的 type hint 的 scope 只存在於該 method 而不是整個 class，mypy 無法透過繼承追蹤型別，所以警告我們出現 override 錯誤，除了在父類加上 Generic，子類也需要在宣告時指定使用哪種型別。

### 正確方式

修正結果如下，只需在 class 繼承時額外指定該 class 的 type。

```python title="修正後使用泛型"
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

AlbumLink: TypeAlias = str
ImageLink: TypeAlias = tuple[str, str]
LinkType = TypeVar("LinkType", AlbumLink, ImageLink)

# 加上 Generic
# highlight-next-line
class BaseScraper(Generic[LinkType], ABC):
    """Abstract base class for different scraping strategies."""

    @abstractmethod
    def process_page_links(self, page_links: list[str]) -> list[LinkType]:
        """Process links found on the page."""


# 指定 AlbumLink
# highlight-next-line
class AlbumScraper(BaseScraper[AlbumLink]):
    def process_page_links(self, page_links: list[str]) -> list[AlbumLink]:
        page_result = []
        for link in page_links:
            page_result.append(link)
        return page_result


# 指定 ImageLink
# highlight-next-line
class ImageScraper(BaseScraper[ImageLink]):
    def process_page_links(self, page_links: list[str]) -> list[ImageLink]:
        page_result = []
        for link in page_links:
            page_result.append((link, "after_some_process"))
        return page_result


links = ["http://example.com/1"]

a = AlbumScraper().process_page_links(links)
b = ImageScraper().process_page_links(links)
print(a)
print(b)
```

`LinkType = TypeVar("LinkType", AlbumLink, ImageLink)` 不一定要設定 `AlbumLink, ImageLink`，設定了就限制只能用這兩種型別。

### 還有其他方法嗎？

當我們想到一個解決方法後，下一步就是問自己有沒有更好的解決方法，這裡筆者自行檢討了幾種不同的 type hint 方式：

1. Generic: 提供了繼承的功能，這是其他方式做不到的。
2. Protocol: 只提供類似 ABC 抽象類別的功能，並沒有針對輸出輸入型別限制，功能完全不同。
3. Union: 更鬆散的 type hint，後續處理輸出的變數會被 IDE 提醒型別問題。
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

如果要最大化發揮 type hint 功效則需要結合 [靜態檢查工具](https://www.trackawesomelist.com/typeddjango/awesome-python-typing/readme/#static-type-checkers) 使用，建議直接整合 pre-commit hooks 使用，請參考文章 [初嘗 Python 工作流自動化](/memo/python/pre-commit-first-try)。

## 結語

本文解釋了沒什麼人講過的 Generic，使用 type hint 時需要自行衡量標注的完整程度和程式開發的方便程度，寫的太完整會導致開發中需要不斷處理各種型別問題，失去 Python 快速開發的意義。

## 參考資料

- [用代码打点酱油的chaofa - Python 类型体操训练](https://bruceyuan.com/post/python-type-challenge-basic.html)  
- [the maintainer of pyright closes valid issues for no reason and lashes out at users](https://docs.basedpyright.com/v1.18.1/) 超好笑，我從來沒在 Github 上看過[兩百個倒讚](https://github.com/microsoft/pyright/issues/8065#issuecomment-2146352290)  
