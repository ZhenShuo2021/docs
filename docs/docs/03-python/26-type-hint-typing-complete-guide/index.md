---
title: Type Hint 教學：從入門到進階的 Python 型別註釋
description: Type Hint 教學：從入門到進階的 Python 型別註釋，掌握 Protocol、泛型等進階技巧 | 2024 Python 3.8-3.14 版本更新整理，最新語法特性 + 實戰技巧 | 一篇精通 Type Hints
tags:
  - Programming
  - Python
  - typing
  - type hint
keywords:
  - Programming
  - Python
  - typing
  - type hint
last_update:
  date: 2024-10-30T04:44:33+08:00
  author: zsl0621
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Type Hint 教學：從入門到進階的 Python 型別註釋

型別註釋功能可以讓身為動態語言的 Python 在執行前就進行檢查，兩大優點分別是 **「讓使用者馬上知道該函式應該輸入哪種類型的變數」** ，不用點進函式閱讀程式碼；並且 **「整合 IDE，在程式撰寫時就可以警告」** ，不用等到執行才知道使用錯誤。

本文主要著重在進階的泛型，對於初階使用網路上已經有非常多文章就不重複撰寫，附上筆者搜尋後覺得最好的資源作為初階介紹，講的非常好，不必再看其他文章：

- [【python】Type Hint入门与初探，好好的python写什么类型标注？](https://www.youtube.com/watch?v=HYE85bqNoGw)
- [【python】Type Hint的进阶知识，这下总该有你没学过的内容了吧？](https://www.youtube.com/watch?v=6rgBwA7TRfE)

如果喜歡文字版本，請看 [用代码打点酱油的chaofa - Python 类型体操训练](https://bruceyuan.com/post/python-type-challenge-basic.html) ，該文章包含完整的語法範例，並且不要看任何其他中文文章以節省各位的時間。


## 基礎關鍵字

本章節紀錄基礎關鍵字，方便讀者快速查找

- list/set/tuple/dict: 列表/集合/元組/字典
- Union: 接受 Union 中的所有類型
- Optional: 接受 Optional 中的所有類型或者 None
- Literal: 限制只能使用指定輸入
- Callable: 可以呼叫的對象，使用方式是 Callable[[input_type1, input_type2], output_type]
- Iterable: 可以迭代的對象（該對象存在 \_\_iter\_\_ 方法，例如 list）

## 中階關鍵字

### TypedDict
用於限制字典的 key-value pair 的變數類型，就是軟限制版本的 @dataclass。  

[Python 类型体操训练（二）-- 中级篇](https://bruceyuan.com/post/python-type-challenge-intermediate.html#typeddict-%E5%9F%BA%E7%A1%80%E7%94%A8%E6%B3%95)



### NoReturn/Literal/NewType
- NoReturn: 告訴程式碼這裡出錯，連預設的 None 都不會返回。  
- Literal: 限制只能使用指定輸入
- NewType: 新增一個型別，如新增 `UserId = NewType('UserId', int)` 此類別會和 `int` 型別不同。  

NoReturn: [【python】Type Hint入门与初探，好好的python写什么类型标注？@198s](https://youtu.be/6rgBwA7TRfE?si=G3uRQeGNjXqPC1jJ&t=198)  
Literal/NewType: [【python】Type Hint入门与初探，好好的python写什么类型标注？@418s](https://youtu.be/6rgBwA7TRfE?si=ae1xcBlXEydsYknj&t=418)

### TypeAlias
TypeAlias 和 NewType 的差異是前者用於建立別名，後者用於新建一個「不同的」類型。建立別名的目的僅是方便記憶和開發管理。

```py
from typing import NewType, TypeAlias

# 分別定義 NewType, TypeAlias 作為範例
UserId = NewType('UserId', int)  
Age: TypeAlias = int


def get_user_age(user_id: UserId, age: Age) -> str:
    return f"User {user_id} is {age} years old"  # 正確


# 正確
user_id = UserId(1234)  # 正確
age: Age = 25  # 正確
print(get_user_age(user_id, age))  # 正確

# 錯誤
user_id_wrong = 1234  # 沒有使用 UserId
age_wrong: Age = "25"  # 錯誤：Age 應為 int
print(get_user_age(user_id_wrong, age_wrong))  # 錯誤：int 和 UserId 是不同類型變數
```



## 高級關鍵字
本章節開始是本篇重點。


### Protocol
檢查該類別是否都實作相同的方法，軟性限制需要實作相同方法，和抽象方法 (@abstractmethod) 的差異是抽象方法是硬性限制，前者還是可以執行（畢竟只是 hint，後者無法執行）。

> 符合 Python 鴨子型別的 typing，長得像就好，其他隨便你

```py
from typing import Protocol


# 定義一個 Protocol，指定必須有 `speak` 方法
class Speaker(Protocol):
    def speak(self) -> str: ...


class Dog:
    def speak(self) -> str:
        return "Woof!"


class Robot:
    def not_speak(self) -> str:
        return "Beep!"


# 指定輸入符合 Speaker 這種類型的 Protocol
def test(entity: Speaker) -> None:
    pass


dog = Dog()
robot = Robot()
print(test(dog))  # 正確
print(test(robot))  # 錯誤

```

### ClassVar

告訴 Python 這是類型變數而不是實例變數，不能在實例化後修改

> 類別變數是所有實例共享的，實例變數則是每個實例獨立擁有的

```py
from typing import ClassVar


class MyClass:
    CLS_VAR: ClassVar[str] = "My Class"


a = MyClass
a.CLS_VAR = "newvalue"  # 正確
b = MyClass()
b.CLS_VAR = "newvalue"  # 錯誤


# 使用字典要小心
class MyClass:
    URL_MAPPINGS: ClassVar[dict[str, str]] = {
        "album": "scrape_album",
        "actor": "scrape_actor",
    }


a = MyClass
a.URL_MAPPINGS["new key"] = "newvalue"  # 正確
b = MyClass()
b.URL_MAPPINGS["new key"] = "newvalue"  # 正確
b.URL_MAPPINGS = {"new album": "value"}  # 錯誤
```

### 泛型

本篇重點終於來了！最開始原本只想寫這裡的不過現在反而變成可跳過章節了。連 mypy 專門做靜態檢查的套件，其官方文檔都表示 [跳過沒差以後再回來看](https://mypy.readthedocs.io/en/stable/generics.html)。

:::tip 泛型

> ChatGPT 4o#2024/10

用於創建可以操作多種型別的函式或類別，無需針對每種特定型別重複實作。這使得程式碼更具彈性、可重複利用，同時保證型別安全。當使用泛型時，可以讓類別或函式在實例化或調用時根據指定的型別參數進行型別推斷，從而在編譯或靜態檢查階段檢測不符合型別的操作。

一句話解釋：實例化後才決定使用的變數，避免重複撰寫相似代碼，同時確保型別安全。

:::

#### TypeVar
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

```py
T = TypeVar("T", int, str)  # 限制只能使用 int/str
T = TypeVar("T", int, str, bound=MyClass)  # 限制只能使用 int/str 和 MyClass 的子類別
```

#### Generic
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


## **實戰泛型：複寫抽象方法**
前面都是使用「設定型別 -> 實例化變數」作為範例方便快速理解，這裡我們考慮一個實際情況：當我們使用 **父類別設定模版**，但是 **子類別的實作輸出卻不同型別**。

以下程式碼中，我們想要限制 `LinkType` 是某幾種限定類別的參數，使用抽象方法並且讓子類別繼承父類別，子類別可以選擇父類別中的任何一種 Link 作為變數型別。第一次嘗試時沒想這麼多，直接宣告

1. 父類別使用 `LinkType` 限制子類別的變數型別
2. 子類別輸出為 `AlbumLink` `ImageLink` ，與父類別型別不同

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

AlbumLink: TypeAlias = str
ImageLink: TypeAlias = tuple[str, str]
LinkType = TypeVar("LinkType", AlbumLink, ImageLink)


class ScrapingStrategy(ABC):
    """Abstract base class for different scraping strategies."""

    @abstractmethod
    def process_page_links(
        self,
        page_links: list[str],
        # highlight-next-line
    ) -> list[LinkType]:
        """Process links found on the page."""


class AlbumListStrategy(ScrapingStrategy):
    def process_page_links(
        self,
        page_links: list[str],
        # highlight-next-line
    ) -> list[AlbumLink]:
        page_result = []
        for link in page_links:
            page_result.append(link)
        return page_result


class AlbumImageStrategy(ScrapingStrategy):
    def process_page_links(
        self,
        page_links: list[str],
        # highlight-next-line
    ) -> list[ImageLink]:
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

這個錯誤的原因有兩個。第一，父類別沒有使用 Generic，導致 `page_result` 的 type hint 的 scope 只存在於該 method 而不是整個 class，mypy 無法透過繼承追蹤型別，所以警告我們出現 override 錯誤；第二，當我們將 `Generic[LinkType]` 加入父類別後再執行檢查會成功，不過此時如果我們使用更嚴格的 --strict 參數進行檢查，會出現 `error: Missing type parameters for generic type "ScrapingStrategy"  [type-arg]`，此時再將選擇子類別應該使用哪種 `AlbumLink` 或是 `ImageLink` 即可解決問題。

修正結果如下，只需在 class 繼承時額外指定該 class 的 type。

```py
from abc import ABC, abstractmethod
from typing import Generic, TypeAlias, TypeVar

AlbumLink: TypeAlias = str
ImageLink: TypeAlias = tuple[str, str]
LinkType = TypeVar("LinkType", AlbumLink, ImageLink)


# highlight-next-line
class ScrapingStrategy(ABC, Generic[LinkType]):
    """Abstract base class for different scraping strategies."""

    @abstractmethod
    def process_page_links(
        self,
        page_links: list[str],
    ) -> list[LinkType]:
        """Process links found on the page."""


# highlight-next-line
class AlbumListStrategy(ScrapingStrategy[AlbumLink]):
    def process_page_links(
        self,
        page_links: list[str],
    ) -> list[AlbumLink]:
        page_result = []
        for link in page_links:
            page_result.append(link)
        return page_result


# highlight-next-line
class AlbumImageStrategy(ScrapingStrategy[ImageLink]):
    def process_page_links(
        self,
        page_links: list[str],
    ) -> list[ImageLink]:
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

當我們想到一個解決方法後，下一步就是問自己有沒有更好的解決方法，不過現在有語言模型，所以這是語言模型的回答：

:::tip 問題（prompt）
1. 使用generic是最好方式嗎？
2. 有沒有更好方式解決？
3. override/overload可以解決這個問題嗎？
:::

> Claude 3.5 sonnet#2024/10

1. 使用 Generic 是否最佳？
- 在這個案例中，Generic 確實是最佳解決方案
- 原因：
  1. Type Safety：提供編譯時期的型別檢查
  2. Code Reuse：允許型別安全的程式碼重用
  3. Interface Consistency：確保 inheritance hierarchy 中的型別一致性

2. 其他可能的解決方案：

```python
# Solution 1: Protocol (Structural Subtyping)
from typing import Protocol

class ScrapingProtocol(Protocol):
    def process_page_links(self, page_links: list[str], 
                          page_result: list[Any]) -> None: ...

# Solution 2: Type Union
class ScrapingStrategy:
    def process_page_links(self, page_links: list[str],
                          page_result: list[Union[AlbumLink, ImageLink]]) -> None:
        pass
```

但這些替代方案的缺點：
- Protocol：失去特定型別的安全性
- Union：無法保證型別的特定性，可能混用不同型別

3. Override/Overload 是否可行：

```python
# Using Overload - Not a good solution
from typing import overload

class ScrapingStrategy:
    @overload
    def process_page_links(self, page_links: list[str],
                          page_result: list[AlbumLink]) -> None: ...
    
    @overload
    def process_page_links(self, page_links: list[str],
                          page_result: list[ImageLink]) -> None: ...
```

Overload 的問題：
1. Static Type Safety：
   - Overload 只提供型別提示，不提供實際的型別約束
   - 無法強制子類別遵守特定的型別規範

2. Code Organization：
   - 需要為每種可能的型別組合寫一個 overload
   - 維護成本高，且容易出錯

3. Design Intent：
   - Overload 主要用於同一個函數處理不同型別參數的情況
   - 不適合用於表達 class hierarchy 中的型別關係

結論：
- Generic 是此案例的最佳解決方案
- 原因：
  1. Type Safety：提供完整的型別安全性
  2. Design Clarity：清楚表達設計意圖
  3. Maintainability：容易維護和擴展
  4. Compile-time Checking：提供編譯時期的型別檢查

使用 Generic 能最好地表達："這是一個可以處理不同型別的策略，但每個具體策略實現都需要指定並遵守其處理的特定型別"。

## 名詞解析

經過這些努力，我們的變數會從抽象型別 (abstract type) 變成具體型別 (concrete type)。

> https://course.khoury.northeastern.edu/cs5010f17/InterfacesClasses2/concrete1.html
>
>
> Concrete and Abstract Data Types  
> A concrete data type is a data type whose representation is known and relied upon by the programmers who use the data type.
> 
> If you know the representation of a data type and are allowed to rely upon that knowledge, then the data type is concrete.
> 
> If you do not know the representation of a data type and are not allowed to rely upon its representation, then the data type is abstract.

## 版本紀錄

本章節紀錄 Python 各個版本新增的 type hint 功能，方便快速查找

- 3.8: 新增 `Protocol`
- 3.9: 內建 `list/set/tuple/dict`，不再需要從 typing 載入，和原本 typing 版本的功能一模一樣。
- 3.10: `Union` 關鍵字可以用管道符號 | 代替。
- 3.12: `Generic` 新增了語法 class MyClass[T]，舊版語法是 class MyClass(Generic[T])。
- 3.12: `TypeAlias` 支援語法[^1] type Vector = list[float]，舊版語法是 Vector: TypeAlias = list[float]
- 3.14: typing 中的 `List/Set/Tuple/Dict` 將被標記為 deprecated。

[^1]: 筆者認為舊版語法和 Python 原本語法一致，不需多記一種比較好讀。


## 結語
其實原本只想寫 Generic，但是想想還是稍微整理一下資訊，結果就是最想寫的反而變成最後一項了。

本文貢獻除了整理真正有用的資訊，也解釋了沒什麼人講過的 Generic。

使用 typing 時需要自行衡量標注的完整程度和程式開發的方便程度，寫的太完整會導致開發中需要不斷處理各種型別，失去 Python 快速開發的意義。

## 參考資料

[用代码打点酱油的chaofa - Python 类型体操训练](https://bruceyuan.com/post/python-type-challenge-basic.html)