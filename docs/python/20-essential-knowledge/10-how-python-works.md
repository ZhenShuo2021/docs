---
title: Python 核心機制與執行流程
sidebar_label: 核心機制與執行流程
description: 介紹 Python 的核心機制解析，包括萬物皆物件、鴨子類型、魔術方法、閉包、元類與垃圾回收。深入探討 Python 執行流程、直譯器與 C 語言的差異，並解析 Byte Code 優化與 .pyc 快取機制，提升對 Python 底層運作的理解。
slug: /how-python-works
tags:
  - Python
keywords:
  - Python
last_update:
  date: 2025-03-27T18:52:00+08:00
  author: zsl0621
first_publish:
  date: 2025-03-27T03:57:00+08:00
---

> 重要程度：1/5

這些知識初學者不需要馬上知道。

## Python 核心機制

### 萬物皆物件

你一定有聽過這句話，但是我覺得網路教學全在講廢話，用 CPython 原始碼和實際行為做範例還更容易理解。在 CPython 中，所有 Python 物件都基於 C 語言的 `PyObject` 結構，這是一個包含型別資訊和引用計數的通用容器。無論是整數、函數還是類別，都使用相同模型：

```c
// 簡化的 PyObject
typedef struct _object {
    Py_ssize_t ob_refcnt;         // 引用計數
    struct _typeobject *ob_type;  // 型別指標
} PyObject;
```

這代表在 Python 中：

- 每個值都有型別資訊
- 每個值都可以被檢查和操作
- 每個值都有自己的方法和屬性

範例如下所示：

```python
# 整數是物件
x = 42
print(x.__class__)  # <class 'int'>
print(x.__sizeof__())  # 可以調用方法

# 函數也是物件
def func():
    pass

print(func.__class__)  # 可以查看名稱
print(func.__sizeof__())  # 可以調用方法
func.custom_attr = "hello"  # 可以動態添加屬性
```

底層的 `PyObject` 結構保證了 Python 中**萬物皆物件**的統一性，帶來靈活和動態特性。

### 鴨子類型 Duck Typing

**如果它走起路來像隻鴨子，叫起來也像隻鴨子，那麼它就是一隻鴨子**，不同於 Java 或 C++ 等靜態型別語言，Python 不關心一個物件的具體型別，而是關心它是否具有特定的方法或行為。

例如一個物件只要實現了 `__len__()` 方法就可以被 `len()` 函數調用，不管它是 list、tuple、字串還是自定義的類別。

### 魔術方法 Magic Method

使用前後雙底線的變數通常是魔術方法，可以自定義對象的內部行為，以一個常見的上下文管理器來說，我們可以自訂他進出和被調用的方式：

```py
class ResourceManager:
    def __init__(self, resource_name):
        self.resource_name = resource_name
        self.call_count = 0
    
    def __enter__(self):
        print(f"【進入】開啟 {self.resource_name} 資源")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"【退出】關閉 {self.resource_name} 資源")
        return False
    
    def __call__(self, *args):
        self.call_count += 1
        print(f"第 {self.call_count} 次調用資源")
        return sum(args)

with ResourceManager("網路連接") as rm:
    result = rm(1, 2, 3)   # 呼叫 __call__
    print(result)          # 輸出 6
```

輸出如下

```
【進入】開啟 網路連接 資源
第 1 次調用資源
6
【退出】關閉 網路連接 資源
```

with 語句會自動幫我們呼叫 `__enter__` 和 `__exit__`，中間使用 `rm(1, 2, 3)` 則是調用了自定義的 `__call__` 方法。

另一個常用的魔術方法是運算符重載自定義標準運算符的行為，以一個座標向量為範例：

```py
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(3, 4)
print(v1 + v2)
```

最後會輸出 `Vector(5, 7)`。

### 閉包

閉包主要在講變數作用域的問題，已經有[優質文章](https://steam.oxxostudio.tw/category/python/basic/closure.html)，不再重複介紹，老大也有解釋閉包的影片[【python】闭包的实现机制。嵌套函数怎么共享变量的？](https://www.youtube.com/watch?v=Flce9y5Qn38)

### 元類 Metaclass

類別本身也是物件，負責創建類別的東西就是元類，簡單來說，元類是「類別的類別」，用來控制類別的創建過程。Python 預設使用 `type` 作為所有類別的元類，但你可以自定義元類來改變類別的行為，詳情請見老大的影片[【python】metaclass理解加入门，看完就知道什么是元类了。](https://www.youtube.com/watch?v=E_-Nl3My3mo)

### Python 的垃圾回收

我們都知道在 C 語言記憶體管理**完全由開發者負責**，使用 `malloc()` 來分配記憶體，在不需要時呼叫 `free()` 釋放，忘記釋放記憶體會導致記憶體洩漏 (Memory Leak) 或是野指標 (Dangling Pointer) 等問題，2024 造成全球電腦大當機的 CrowdStrike 事件就是存取到野指標造成的，[epcdiy 對此有詳細介紹](https://www.youtube.com/watch?v=05Vgrq_DfS8)。

Python 則沒有此問題，因為他採用**自動垃圾回收（Garbage Collection, GC）**，減少手動管理記憶體的負擔。Python 的記憶體管理核心機制是**引用計數(Reference Counting)**，當物件的引用數降為零時，Python 會立即回收該物件。然而引用計數無法處理循環引用 (Cyclic References)，因此 Python 內建的 GC 模組會額外執行標記-清除 (Mark and Sweep) 來移除無用的循環物件，GC 也使用分代回收 (Generational GC)，將物件分為不同世代，較舊且存活時間較長的物件會被較少檢查，以降低 GC 開銷。

老大也拍了 GC 影片 [Unreachable的对象咋回收的？generation又是啥？](https://www.youtube.com/watch?v=8cOJwmvcl90)

## Python 執行流程

CPU 只看得懂 Machine Code 看不懂我們寫的程式碼，那 Python 是怎麼讓 CPU 讀懂程式碼並運行呢？

包含三個步驟：**Interpreter** 解析 `.py` 檔，轉換為 **Byte Code**；**Virtual Machine** 處理 Byte Code，轉為 **Machine Code**；最後 **CPU** 執行 Machine Code 完成任務。  

- **Interpreter（直譯器）**：將 Python 原始碼轉換為 Byte Code，讓後續步驟處理。  
- **Byte Code（位元組碼）**：Interpreter 產生的中間二進制程式碼，用於讓虛擬機運行，使用 `dis` 函數可以看到編譯的 Byte Code。  
- **Virtual Machine（虛擬機）**：執行 Byte Code 將其為硬體可讀的 Machine Code。  
- **Machine Code（機器碼）**：CPU 可直接執行的低階指令。  

### Python 主要 Interpreter

Python 常見 **Interpreter** 包括 **CPython**（最普及）、**PyPy**（效能優化）與 **Jython**（與 Java 整合）。不同 **Interpreter** 影響 **Byte Code** 生成方式，但整體執行機制相同。  

- **CPython**：官方標準版，用 C 語言實作，最多人使用，支援最多功能。
- **PyPy**：使用 JIT（即時編譯）技術，加速執行效能。
- **Jython**：使用 Java 實作，最近兩年不到一百個提交，最近半年完全沒提交，基本上死了。

在 uv 中可以使用 `uv python list --all-versions` 列出他支援的 Python 版本，並且使用 `uv venv --python=<name>` 建立該直譯器的虛擬環境。

### Python 與 C 的執行方式差異

C 在每次運行前需要經過編譯步驟，經過編譯器將程式碼編譯為 **Machine Code**，而 Python 經過 **Interpreter** 轉換成 **Byte Code** 作為中介，再透過 **Virtual Machine** 運作，讓他可以免編譯執行並且跨平台，代價就是非常高的效能損失。

- **編譯語言（Compiled Language）**：像 C 一樣，程式碼會被完整轉換為 Machine Code，執行效率高。  
- **直譯語言（Interpreted Language）**：像 Python，程式碼逐步轉換並執行，靈活但效能相對較低。  

:::tip
我們會把效能要求高的任務用高效率的 C/Rust/Go 寫，Python 只作為呼叫的介面，這樣就可以有方便的程式碼和高效率的程式執行，例如 Pybind11 就是 Python 和 C 之間的呼叫橋樑，甚至 Numba 還可以自動完成平行化和 SIMD 等加速方式而不需撰寫任何一個 C 程式碼，深度學習開發也同理，實際上在呼叫 C 和 CUDA。
:::

:::tip
本站也有鉅細靡遺的 [Numba 教學](numba-tutorial-1)！
:::

### 如何優化 Byte Code

先講結論，花 80% 的力氣優化 5% 的效能和記憶體，完全沒必要。

舉例來說使用 comprehension 等技巧可以加速執行和降低記憶體使用，因為使用 comprehension 技巧就可以完全讓直譯器掌握你的迴圈在做什麼，從而編譯出更好的 Byte Code，但是不值得花時間心力搞這東西，要存空間大到值得花時間解決的超大數據也不會拿內建 list 來存，我們有 Numpy，數據庫，甚至是 generator 都可以優化記憶體問題，更重要的是可讀性問題，所以結論是毫無意義，無須關注。

相關內容請見 [Python byte-code and micro-optimization](https://medium.com/@chipiga86/python-byte-code-and-micro-optimization-1c0acb902c9) 還有我老大的影片 [原来我可以少写这么多for loop！学会之后代码都pythonic了起来](https://www.youtube.com/watch?v=8DJ6M3tvnwY)。

### `.pyc` 與 `__pycache__`

Python 會將 **Byte Code** 快取成 `.pyc` 檔案，存放於 `__pycache__` 目錄，加快未來執行速度。  

- **.pyc（已編譯 Python 檔案）**：Python 轉換為 Byte Code 後的二進位檔案，避免每次執行時重複解析 `.py` 原始碼。  
- **__pycache__（快取目錄）**：存放 `.pyc` 檔，通常依 Python 版本區分，如 `script.cpython-39.pyc` 代表 Python 3.9 產生的 Byte Code。
