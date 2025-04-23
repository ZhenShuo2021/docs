---
title: Python Type Hint 型別提示教學 - 基礎篇
sidebar_label: 型別提示 - 基礎篇
slug: /type-hint
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

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

型別提示功能可以讓身為動態語言的 Python **無痛的**帶來了靜態語言的優點，同時又保持原本動態語言的靈活性，如果看不懂，講人話就是

1. **提高可讀性**：開發者可以清楚知道函數或變數預期的資料型別，不需要額外翻閱文件或原始碼。
2. **減少錯誤**：型別提示可以搭配 mypy 等工具檢查型別不同的錯誤。
3. **工具整合**：現代 IDE 可以利用型別提示提供**自動完成和建議**。

初階使用網路上已經有非常多文章就不重複撰寫，附上筆者整理後覺得最好的資源，講的非常好：

- [【python】Type Hint入门与初探，好好的python写什么类型标注？](https://www.youtube.com/watch?v=HYE85bqNoGw)
- [【python】Type Hint的进阶知识，这下总该有你没学过的内容了吧？](https://www.youtube.com/watch?v=6rgBwA7TRfE)
- [Python 类型体操训练（一）-- 基础篇](https://bruceyuan.com/post/python-type-challenge-basic.html)
- [Python 类型体操训练（二）-- 中级篇](https://bruceyuan.com/post/python-type-challenge-intermediate.html)
- [Python 类型体操训练（三）-- 高级篇](https://bruceyuan.com/post/python-type-challenge-advanced.html)

建議看幾個資源就好，筆者發現大部分的中文文章品質很差不建議閱讀。

## 注意事項

這些是我覺得實用但不常見的型別提示知識，獨一無二只有我寫，所以放在教學最前面。

**1. Type hint 不是越多越好**

Python 的優勢是快速迭代與靈活開發，過度型別✍反而會拖慢流程，尤其是複雜型別如果沒有明顯效益，過度優化型別提示只會增加維護成本。

**2. 少用 Union，除非有穩健的 type narrowing**

Union 會造成 IDE 與靜態檢查工具一直抱怨型別問題，如果沒有明確的 type narrowing 機制那不如直接用 `Any` 簡化開發流程。

**3. 用 `reveal_type` / `reveal_locals` 偵錯型別**

遇到不確定的變數型別時，可以利用 mypy 內建的偵錯工具 `reveal_type` 和 `reveal_locals`，不需 import 直接寫在程式中即可。在終端機執行 `mypy example.py` 就會顯示推論結果。使用教學請見 [How to Debug Types with reveal_type](https://adamj.eu/tech/2021/05/14/python-type-hints-how-to-debug-types-with-reveal-type/)。

**4. `if TYPE_CHECKING` 的正確用途**

有說法是「盡量避免使用」的東西，這是錯的，正確用途是**需要導入只用於型別檢查的情境**，可以避免不必要的 import，又能保持型別提示完整，想用就用無須煩惱。

**5. 延遲解析型別：使用 `"ClassName"`**

型別還沒宣告而我們通常又希望在檔案開頭集中定義所有型別提示時使用，以字串表示型別名稱（如 `"MyClass"`）來延遲解析，在我的[泛型教學](type-hint-generic#covariant)有使用範例。

**6. `List`, `Dict` 不需再從 `typing` 匯入**

Python 3.9 之後可以直接使用內建的小寫泛型語法（如 `list[int]`、`dict[str, Any]`），不必再使用 `List`, `Dict` 等大寫版本。因為 Python 3.8 已經 EOL，除非有兼容性需求否則應該使用新語法。

**7. Python 3.10 沒有預設啟用 `from __future__ import annotations`**

告訴型別提示會延遲解析，避免循環引用等問題，如果不加上這行就要把型別改為字串才能延遲解析，然而此提案被退回，沒有正式實施！

## 初階使用

只記錄基礎關鍵字，方便讀者快速查找，基本上只用這些關鍵字也能完成九成以上的 type hint。

- [list/dict/tuple/set](https://bruceyuan.com/post/python-type-challenge-basic.html#%E7%AE%80%E5%8D%95%E5%8F%98%E9%87%8F): 列表/集合/元組/字典
- [Union](https://bruceyuan.com/post/python-type-challenge-basic.html#union): 接受 Union 中的所有類型，可直接用 `list | dict | tuple` 替代
- [Optional](https://bruceyuan.com/post/python-type-challenge-basic.html#optional): 接受 Optional 中的所有類型或者 None，可直接用 `list | None` 替代
- [Literal](https://bruceyuan.com/post/python-type-challenge-intermediate.html#literal): 限制只能使用指定輸入，通常用於常數
- [Callable](https://bruceyuan.com/post/python-type-challenge-intermediate.html#callable): 可以呼叫的對象，使用方式是 `Callable[[input_type1, input_type2], output_type]`
- Iterable: 可以迭代的對象（該對象存在 `__iter__` 方法，例如 list）
- [Final](https://bruceyuan.com/post/python-type-challenge-basic.html#final): 最終結果，不應該被覆寫
- `if TYPE_CHECKING`: 只有在型別檢查時才會啟用此區塊，通常用於啟用型別提示系統，避免真的 import

接下來是稍微複雜一點的型別提示。

### NoReturn

[NoReturn](https://mypy.readthedocs.io/en/latest/more_types.html#the-noreturn-type) 告訴型別提示系統這個函式應該要出錯 (raise exception)，連 None 都不會返回。  

NoReturn: [【python】Type Hint入门与初探，好好的python写什么类型标注？@198s](https://youtu.be/6rgBwA7TRfE?si=G3uRQeGNjXqPC1jJ&t=198)  

### NewType/TypeAlias

- NewType: 新增一個型別，如新增 `UserId = NewType('UserId', int)` 此類別會和 `int` 型別不同。  
- TypeAlias: 建立別名，和 NewType 的差異是前者用於建立別名，後者用於新建一個「不同的」類型。建立別名的目的僅是方便記憶和開發管理。

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

NewType: [【python】Type Hint入门与初探，好好的python写什么类型标注？@418s](https://youtu.be/6rgBwA7TRfE?si=ae1xcBlXEydsYknj&t=418)

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

## 高階使用

可以跳過這裡也沒關係，完全不用這裡的 type hint 也不太會對型別提示系統帶來問題或是降低功能。

### overload/override

用於提示 mypy 輸入輸出型別的多載的裝飾器，和 C++ 真正意義上的多載不同，只用於提示 mypy/IDE 而已。overload 用於函式或方法之間，override 用於繼承之間。

寫一寫會不小心忘記這些只是提示，[就像這篇文章一樣](https://stackoverflow.com/questions/57222412/cannot-guess-why-overloaded-function-implementation-does-not-accept-all-possible)，請記得 type hint 完全不影響 Python 實際運作。

### Protocol

檢查該類別是否都實作相同的方法，軟性限制需要實作相同方法，和抽象方法 (abstractmethod) 的差異是抽象方法是硬性限制，前者還是可以執行（畢竟只是 hint，後者無法執行）。

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

### TypedDict

用於限制字典的 key-value pair 的變數類型，就是軟限制版本的 dataclass，個人覺得滿難用的，除非 dict 資料型別絕對不會變動才使用他。

[Python 类型体操训练（二）-- 中级篇](https://bruceyuan.com/post/python-type-challenge-intermediate.html#typeddict-%E5%9F%BA%E7%A1%80%E7%94%A8%E6%B3%95)

### 不常用關鍵字

- Annotated: 用於[附註變數](https://stackoverflow.com/questions/71898644/how-to-use-python-typing-annotated)。
- Self: 回傳類別本身。
- typeguard: 用於 [type narrowing](https://rednafi.com/python/typeguard_vs_typeis/)。

## 相關工具

雖然只有兩行但是很重要，所以獨立一個段落。

如果要最大化發揮 type hint 功效則需要結合 [靜態檢查工具](https://www.trackawesomelist.com/typeddjango/awesome-python-typing/readme/#static-type-checkers) 使用，建議直接整合 pre-commit hooks 使用，請參考文章 [初嘗 Python 工作流自動化](/memo/python/pre-commit-first-try)。

## 版本紀錄

紀錄 Python 各個版本新增的 type hint 功能方便快速查找

- 3.8: 新增 `Protocol`
- 3.9: 內建 `list/set/tuple/dict`，不再需要從 typing 載入，還有很多 collections [不再建議從 typing 載入](https://stackoverflow.com/questions/65120501/typing-any-in-python-3-9-and-pep-585-type-hinting-generics-in-standard-collect)
- 3.9: 新增 `Annotated` 功能
- 3.10: `Union` 關鍵字可以用管道符號 | 代替
- 3.10: 預設使用 `from __future__ import annotations`，此功能允許延遲型別提示，允許定義類別時使用類別自身作為型別提示
- 3.12: `Generic` 新增了語法 `class MyClass[T]`，舊版語法是 `class MyClass(Generic[T])`
- 3.12: `TypeAlias` 支援語法 `type Vector = list[float]`，舊版語法是 `Vector: TypeAlias = list[float]`
- 3.12: 新增 `Override`
- 3.14: typing 中的 `List/Set/Tuple/Dict` 將被標記為 deprecated
