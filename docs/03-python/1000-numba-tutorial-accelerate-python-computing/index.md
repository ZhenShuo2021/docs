---
title: 最快、最好、最完整的 Numba 教學：使用 Numba 加速科學計算
description: 最快、最好、最完整的 Numba 教學：使用 Numba 加速科學計算。坑筆者都踩過了只要照做可以得到最好性能，不會漏掉任何優化可能；除此之外本文第一不廢話，第二上手極快，第三介紹如何除錯和優化，第四補充進階使用方式，第五給出「精選有用的延伸閱讀」，不是給沒用文章，第六也是最重要，筆者自認為本文是中文圈最詳細教學。
link: test
tags:
  - Programming
  - Python
  - Numba
  - Tutorial
keywords:
  - Programming
  - Python
  - Numba
  - Tutorial
last_update:
  date: 2024-10-03 GMT+8
  author: zsl0621
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 最快、最好、最完整的 Numba 教學：使用 Numba 加速科學計算
鑑於繁體中文資訊少，最近剛好又重看了一下文檔，於是整理資訊分享給大家。本篇的目標讀者是沒學過計算機的初階用戶到中階用戶都可以讀，筆者自認本篇教學已經覆蓋到絕大部分使用場景，只有 CUDA 沒有覆蓋到，但也提供相關教學連結。

- **重要**  
    以免看了整篇才發現沒用：
    > Q: [哪些計算適合 Numba](https://numba.pydata.org/numba-doc/dev/user/5minguide.html#will-numba-work-for-my-code)  
    > A: If your code is numerically orientated (does a lot of math), uses NumPy a lot and/or has a lot of loops, then numba is often a good choice

    翻譯：大量包含迴圈的 Numpy 數值運算，且不涉及 IO 操作 (pandas)。
- **為甚麼選擇此教學**  
<u>最快、最好、最完整。</u> 坑筆者都踩過了只要照做可以得到最好性能，不會漏掉任何優化可能；除此之外本文第一不廢話，第二上手極快，第三介紹如何除錯和優化，第四補充進階使用方式，第五給出「精選有用的延伸閱讀」，不是給沒用文章，第六也是最重要，**筆者自認為本文是中文圈最詳細教學**。

- **如何閱讀本文**  
不用害怕文章看起來很長，初學者可以先不使用平行化，跳過章節 <u>自動平行化與競爭危害</u> 可以快速上手 Numba；對於大多數人而言 <u>進階使用</u> 都不用看；如果你急到不行，看完 <u>一分鐘學會如何使用</u> 後直接跳到 <u>如何除錯</u>。

:::info 寫在前面

不要看 [舊版，左上角版本號 0.52](https://numba.pydata.org/) 的文檔！缺失部分在新版文檔有補齊，偏偏舊版文檔 Google 搜尋在前面。

:::

## 原理
- 使用 JIT 編譯器取代 Python 字結碼轉譯，編譯成 machine code 優化效能（見[詞彙表](/docs/python/numba-tutorial-accelerate-python-computing#附錄詞彙表)）。
- 靜態類型推斷，避免 Python 中一堆類型檢查 overhead。
- 特別優化數值計算，如使用 SIMD。

## 基礎使用
說是基礎使用，但是已經包含九成的使用情境。

### 一分鐘學會如何使用

比官方的五分鐘教學又快五倍，夠狠吧。測試對陣列開根號後加總，比較每個方法的時間，分別是有沒有使用 numba 和有沒有使用陣列計算總共四個函式的運算時間。
```py
import numpy as np
import time
from numba import jit, prange


@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def sum_of_sqrt_numba(arr):
    # Numba + Loop
    total = 0
    for x in prange(len(arr)):
        total += x**0.5
    return total


def sum_of_sqrt_python(arr):
    # Python + Loop
    total = 0.0
    for x in arr:
        total += x**0.5
    return total


@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def sum_of_sqrt_numba_arr(arr):
    # Numba + Vector
    return np.sum(np.sqrt(arr))


def sum_of_sqrt_python_arr(arr):
    # Python + Vector
    return np.sum(np.sqrt(arr))


n = 10000000
arr = np.arange(n)

# 第一次運行的初始化，第二次以後才是單純的執行時間
sum_of_sqrt_numba(arr)
sum_of_sqrt_numba_arr(arr)

start = time.time()
result_python = sum_of_sqrt_python(arr)
end = time.time()
print(f"Python迴圈版本執行時間: {end - start} 秒")

start = time.time()
result_python_arr = sum_of_sqrt_python_arr(arr)
end = time.time()
print(f"Python陣列版本執行時間: {end - start} 秒")

start = time.time()
result_numba = sum_of_sqrt_numba(arr)
end = time.time()
print(f"Numba迴圈版本執行時間: {end - start} 秒")

start = time.time()
result_numba_arr = sum_of_sqrt_numba_arr(arr)
end = time.time()
print(f"Numba陣列版本執行時間: {end - start} 秒")

print("Are the outputs equal?", np.isclose(result_numba, result_python))
print("Are the outputs equal?", np.isclose(result_numba_arr, result_python_arr))

# 輸出結果

# Python迴圈版本執行時間: 12.916845798492432 秒
# Python陣列版本執行時間: 0.03603005409240723 秒
# Numba迴圈版本執行時間: 0.0018897056579589844 秒
# Numba陣列版本執行時間: 0.002209901809692383 秒
# Are the outputs equal? True
# Are the outputs equal? True
```

比對函式差異可以看到使用方式很簡單，在要優化的函式前加上 `@jit` 裝飾器，在要平行化處理的地方顯式的改為 prange 就完成了。裝飾器的選項[有以下幾個](https://numba.readthedocs.io/en/stable/user/jit.html#compilation-options)：

| 參數      | 說明                                                      |
|----------|-----------------------------------------------------------|
| nopython | 是否忽略 Python C API，產生最快的機器碼                       |
| parallel | 是否平行運算                                                |
| fastmath | 是否放寬 IEEE 754 的精度限制以獲得額外性能                     |
| cache    | 是否將編譯結果寫入快取，避免每次呼叫 Python 程式時都需要編譯      |
| nogil    | 是否關閉全局鎖                                              |

<br/>
<br/>

:::info 安全性提醒

1. fastmath: 測試時使用，離正式生產越近越該關閉。
2. 依筆者個人使用 fastmath/nogil 並沒有快多少，當然這是 case-specific，像上面的範例就有差。
3. numba 每次編譯後全域變數會變為常數，在程式中修改該變數不會被函式察覺。

:::


:::danger 安全性***警告***

**對於暫時不想處理競爭危害的用戶，請先不要使用 `parallel` `nogil` 方法。**
1. parallel/nogil: 小心[競爭危害](https://zh.wikipedia.org/zh-tw/%E7%AB%B6%E7%88%AD%E5%8D%B1%E5%AE%B3) (race condition)。簡單說明競爭危害，就是兩個線程一起處理一個運算 `x += 1`，兩個一起取值，結果分別寫回 x 的值都是 `x+1` 導致最終結果是 `x+1` 而不是預期的 `x+2`。
2. 雖然上面的範例顯示結果一致，但還是一定要 **避免任何可能的多線程問題！**

:::

<br/>

### 效能優化
基礎使用章節已經包含官方文檔中所有[效能優化技巧](https://numba.readthedocs.io/en/stable/user/performance-tips.html)，只是沒有每個選項各自對比，這裡補充其他效能優化方式。

1. 安裝 SVML (short vector math library), threading layers (平行計算, tbb/omp)，安裝後不需設定 numba 會自行調用
```sh
# conda
conda install intel-cmplr-lib-rt
conda install tbb
conda install anaconda::intel-openmp

# pip
pip install intel-cmplr-lib-rt
pip install tbb
pip install intel-openmp

# Troubleshooting: 沒有發現 SVML 的解決方式
# https://github.com/numba/numba/issues/4713#issuecomment-576015588
# numba -s   # 檢查是否偵測到 SVML
# sudo pip3 install icc_rt; sudo ldconfig
```

2. 使用 Numba 反而變慢
    - 記得扣掉首次運行需要消耗的編譯時間。
    - 檢查 IO 瓶頸，不要放任何需要 IO 的程式碼在函式中。
    - 輸入範圍太小。
    - 宣告後就不要修改矩陣維度或型別。
    - 盡量寫簡單的語法，不要各種包裝，因為你不知道 Numba 是否支援。
    - 記憶體問題 [The wrong way to speed up your code with numba](https://pythonspeed.com/articles/slow-numba/)。

3. 無關 Numba，有關本身程式撰寫
    - [How to Write Fast Numerical Code](https://people.inf.ethz.ch/markusp/teaching/263-2300-ETH-spring14/slides/06-locality-caches.pdf)

2. [threading layers 設定平行計算方式](https://numba.readthedocs.io/en/stable/user/threading-layer.html)
    - `default` provides no specific safety guarantee and is the default.
    - `safe` is both fork and thread safe, this requires the tbb package (Intel TBB libraries) to be installed.
    - `forksafe` provides a fork safe library.
    - `threadsafe` provides a thread safe library.

threading layers 官方範例
```py
from numba import config, njit, threading_layer, set_num_threads
import numpy as np

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'threadsafe'

# Setting NUMBA_NUM_THREADS=2, the parallel code will only execute on 2 threads.
# 在某些情形下應該設定為較低的值，以便 numba 可以與更高層級的平行性一起使用。 （但是文檔沒有說是哪些情形）
set_num_threads(2)



@njit(parallel=True)
def foo(a, b):
    return a + b

x = np.arange(10.)
y = x.copy()

# this will force the compilation of the function, select a threading layer
# and then execute in parallel
foo(x, y)

# demonstrate the threading layer chosen
print("Threading layer chosen: %s" % threading_layer())
```

### 自動平行化與競爭危害
本章節整理自文檔 [Automatic parallelization with @jit](https://numba.readthedocs.io/en/stable/user/parallel.html#)。

#### 自動平行化

> 設定 Numba 自動平行化的官方文檔，由於很精練，知識也很重要，所以翻譯完貼在這裡。  
> 簡單來說，網路上手刻平行化的人連文檔都沒看就開始亂寫文章了。Numba 支援自動平行化，並且快取優化更好，不需要你手刻。

在 `jit()` 函式中設置 `parallel` 選項，可以啟用 Numba 的轉換過程，嘗試自動平行化函式（或部分函式）以執行其他優化。目前此功能僅適用於CPU。

一些在用戶定義的函式中執行的操作（例如對陣列加上純量）已知具有平行語義。用戶的程式碼可能包含很多這種操作，雖然每個操作都可以單獨平行化，但這種方法通常會因為快取行為不佳而導致性能下降。相反地，通過自動平行化，Numba 會嘗試識別用戶程式碼中的這類操作並將相鄰的操作合併到一起，形成一個或多個自動平行執行的 kernels。這個過程是完全自動的，無需修改用戶程式碼，這與 Numba 的 `vectorize()` 或 `guvectorize()` 機制形成對比，後者需要手動創建並行 kernels。


#### 競爭危害
這裡展示競爭危害的範例和解決方式，顯示出競爭危害的存在，請不要錯誤的推斷為 scalar 運算可以避免而 vector 運算不行，**任何時候我們都應該避免競爭危害的可能**。那我們就不能寫 for 迴圈了嗎？其實有其他方法，例如這兩種方式都可以。

<!-- <details>
<summary>競爭危害範例</summary> -->

<Tabs>
  <TabItem value="1" label="發生競爭危害的範例">

```py
from numba import njit, prange
import numpy as np


@njit(parallel=True)
def prange_wrong_result_numba(x):
    n = x.shape[0]
    y = np.zeros(4)
    for i in prange(n):
        # 一般的矩陣相加，會造成競爭危害
        y[:] += x[i]
    return y


@njit(parallel=True)
def prange_wrong_result_mod_numba(x):
    n = x.shape[0]
    y = np.zeros(4)
    for i in prange(n):
        # 優化後的矩陣相加，嘗試利用不同的 i 取餘數避免競爭危害，仍舊失敗
        y[i % 4] += x[i]
    return y


# 沒有加上裝飾器的版本
def prange_wrong_result_python(x):
    n = x.shape[0]
    y = np.zeros(4)
    for i in range(n):
        y[:] += x[i]
    return y


# 沒有加上裝飾器的版本
def prange_wrong_result_mod_python(x):
    n = x.shape[0]
    y = np.zeros(4)
    for i in range(n):
        y[i % 4] += x[i]
    return y


# 提醒：使用浮點數測試時因浮點數特性每次計算結果都會有誤差，所以比較時應該使用整數測試
x = np.random.randint(-10, 100, size=1000000)

result_numba = prange_wrong_result_numba(x)
result_python = prange_wrong_result_python(x)
print("Are the outputs equal?", np.array_equal(result_numba, result_python))

result_numba_mod = prange_wrong_result_mod_numba(x)
result_python_mod = prange_wrong_result_mod_python(x)
print("Are the outputs equal?", np.array_equal(result_numba_mod, result_python_mod))

# 輸出

# Are the outputs equal? False
# Are the outputs equal? False
```
</TabItem>

  <TabItem value="2" label="解決方式">

```py
from numba import njit, prange
import numpy as np


@njit(parallel=True)
def prange_ok_result_whole_arr(x):
    n = x.shape[0]
    y = np.zeros(4)
    for i in prange(n):
        # "whereas performing a whole array reduction is fine" 節錄自官方文檔
        y += x[i]
    return y


@njit(parallel=True)
def prange_ok_result_outer_slice(x):
    n = x.shape[0]
    y = np.zeros(4)
    z = y[:]
    for i in prange(n):
        # "as is creating a slice reference outside of the parallel reduction loop" 節錄自官方文檔
        z += x[i]
    return y


# 沒有加上裝飾器的版本
def prange_ok_result_whole_arr_python(x):
    n = x.shape[0]
    y = np.zeros(4)
    for i in prange(n):
        y += x[i]
    return y


# 沒有加上裝飾器的版本
def prange_ok_result_outer_slice_python(x):
    n = x.shape[0]
    y = np.zeros(4)
    z = y[:]
    for i in prange(n):
        z += x[i]
    return y


# 提醒：使用浮點數測試時因浮點數特性每次計算結果都會有誤差，所以比較時應該使用整數測試
x = np.random.randint(-10, 100, size=(1000000, 4))

result_numba_whole_arr = prange_ok_result_whole_arr(x)
result_python_whole_arr = prange_ok_result_whole_arr_python(x)
print("Are the outputs equal?", np.array_equal(result_numba_whole_arr, result_python_whole_arr))

result_numba_outer_slice = prange_ok_result_outer_slice(x)
result_python_outer_slice = prange_ok_result_outer_slice_python(x)
print("Are the outputs equal?", np.array_equal(result_numba_outer_slice, result_python_outer_slice))

# 輸出

# Are the outputs equal? True
# Are the outputs equal? True
```

</TabItem>


  <TabItem value="3" label="正確使用範例">

節錄自官方文檔，此運算基本上和基礎使用的範例相同，英文 reduction 代表運算完成後降低維度
> "The example below demonstrates a parallel loop with a reduction (A is a one-dimensional Numpy array)":

```py
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def prange_test(A):
    s = 0
    # Without "parallel=True" in the jit-decorator
    # the prange statement is equivalent to range
    for i in prange(A.shape[0]):
        s += A[i]
    return s

@njit(parallel=True)
def two_d_array_reduction_prod(n):
    shp = (13, 17)
    result1 = 2 * np.ones(shp, np.int_)
    tmp = 2 * np.ones_like(result1)

    for i in prange(n):
        result1 *= tmp

    return result1


n = 100000
A = np.random.randint(-5, 10, size=n)
result_numba_test = prange_test(A)
result_python_test = sum(A)
print("Are the outputs equal?", np.array_equal(result_numba_test, result_python_test))

result_numba_prod = two_d_array_reduction_prod(n)
result_python_prod = np.power(2, n) * np.ones((13, 17), dtype=np.int_)
print("Are the outputs equal?", np.array_equal(result_numba_prod, result_python_prod))

# 輸出

# Are the outputs equal? True
# Are the outputs equal? True
```

</TabItem>


</Tabs>

<!-- </details> -->

:::info 迴圈出口

`prange` 不支援多個出口的迴圈，例如中間包含 `assert`。

:::

:::warning 迴圈變數隱性轉型

關閉平行化處理時迴圈變數 (induction variable) 沒有問題，和 Python 預設一樣使用有號整數。然而如果開啟平行化且範圍可被識別為嚴格正數，則會被自動轉型為 `uint64`，而 `uint64` 和其他變數計算時**有機會不經意的返回一個浮點數**。

:::



#### 平行化的優化技巧

官方文檔介紹平行化迴圈的優化技巧。

1. **迴圈融合 (Loop Fusion)：** 將相同迴圈邊界的迴圈合併成一個大迴圈，提高資料局部性進而提升效能。
2. **迴圈序列化 (Loop Serialization)：** Numba 不支援巢狀平行化，當多個 `prange` 迴圈嵌套時只有最外層的 `prange` 迴圈會被平行化，內層的 `prange` 迴圈會被視為普通的 `range` 執行。
3. **提出不變的程式碼 (Loop Invariant Code Motion)：** 將不影響迴圈結果的語句移到迴圈外。
4. **分配外提 (Allocation Hoisting)**：範例是拆分 `np.zeros` 成 `np.empty` 和 `temp[:] = 0` 避免重複初始化分配。

進一步優化: 請見 [Diagnostics your parallel optimization](https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics)。

### 如何除錯
當年在寫 Numba 在出現錯誤時 Numba 的報錯資訊不明確，現在不知道有沒有改進，那時的方法是「找到錯誤行數的方式是二分法直接刪程式碼到 Numba 不報錯」

錯誤通常是變數型別錯誤，例如誤用不支援相加的不同的變數型別，或者使用 Numba 不支援的函式，除錯請先看函式是否支援以免當冤大頭。

- [Supported Python features](https://numba.readthedocs.io/en/stable/reference/pysupported.html)
- [Supported NumPy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)

### 小結
1. Numba like loops 在心裡默念十次
2. Numba likes NumPy functions
3. Numba likes NumPy broadcasting
4. 不要在函式內修改數據結構 
5. 保持順序記憶體讀取
6. 函式中不要包含 IO 操作
7. ***還是 Numba like loops***

到這裡就結束基本使用了，建議先不要看進階使用，而是跳到 [See Also](/docs/python/numba-tutorial-accelerate-python-computing#see-also) 看延伸閱讀。

---

## 進階使用

```sh
# 這是用來阻止你繼續讀的 placeholder！
 _   _                       _             
| \ | |  _   _   _ __ ___   | |__     __ _ 
|  \| | | | | | | '_ ` _ \  | '_ \   / _` |
| |\  | | |_| | | | | | | | | |_) | | (_| |
|_| \_|  \__,_| |_| |_| |_| |_.__/   \__,_|

```
除非你是進階用戶，否則 **你不應該看進階使用章節！** 看了反而模糊焦點，你應該把握好如何基礎使用，基礎使用已經包含了九成以上的使用情景。

進階使用裡面就只有 [numba.typed.dict](/docs/python/numba-tutorial-accelerate-python-computing#numbatypeddict) 你可以稍微看一下。

### CUDA 運算
> 這就是剩下那一成。

優化 CUDA 不像優化 CPU 加上裝飾器那麼簡單，而是要針對 CUDA 特別寫函式，導致程式只能在 GPU 上跑，所以筆者目前還沒寫過，不過基本注意事項一樣是注意 IO、工作量太小的不適合 CUDA。那比較什麼函式適合 CPU 而不是 CUDA 呢？

1. **順序處理而不是平行處理**，影像處理以外的演算法大概都是這類
2. 記憶體超過顯卡記憶體上限（註：不應該寫出這種程式，Numba like loops）
3. 大量分支處理 (if-else)（註：不應該寫出這種程式，尤其在 Numba 中）
4. 顯卡雙精度浮點運算效能差，深度學習和遊戲都吃單精度，但是科學計算需要雙精度，我們又只能買到遊戲卡
5. 一些 library 只支援 CPU，這要試了才知道

如果你需要使用 CUDA，這裡也有好用的指南連結：

- [28000x speedup with numba.CUDA](https://curiouscoding.nl/posts/numba-cuda-speedup/)：使用 CUDA 加速並且有完整的對比。  
- [用 numba 學 CUDA! 從入門到精通 (上)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8A-ede7b381f6c7)
- [用 numba 學 CUDA! 從入門到精通 (下)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8B-770c11bffd37)

### 其他裝飾器
常見裝飾器有

- vectorize
- guvectorize
- jitclass
- stencil

#### vectorize
允許把 scalar 輸入的函式當作向量 [Numpy ufunc](http://docs.scipy.org/doc/numpy/reference/ufuncs.html) 使用。

這裡是範例，網路上說 vectorize 目的是平行處理還是向量化都是過時的，文檔寫的很清楚，vectorize 是用來讓函式能用作 Numpy ufunc 函式，於是你就可以把一個簡單的函式改成像 numpy 一樣使用，但是有著 numba 的速度優化。此方法不推薦使用，因為不好懂，你看著一個函式會想說這方法從哪來的，別人不好理解程式碼，IDE 也會跳警告。
```py
# Edit from: https://github.com/numba/numba/blob/main/numba/tests/doc_examples/test_examples.py
# test_vectorize_multiple_signatures
from numba import vectorize, int32, int64, float32, float64
import numpy as np

@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
def f(x, y):
    return x + y

a = np.arange(6)
result = f(a, a)
# result == array([ 0,  2,  4,  6,  8, 10])

correct = np.array([0, 2, 4, 6, 8, 10])
np.testing.assert_array_equal(result, correct)

a = np.linspace(0, 1, 6)
result = f(a, a)
# Now, result == array([0. , 0.4, 0.8, 1.2, 1.6, 2. ])

correct = np.array([0., 0.4, 0.8, 1.2, 1.6, 2. ])
np.testing.assert_allclose(result, correct)

a = np.arange(12).reshape(3, 4)
# a == array([[ 0,  1,  2,  3],
#             [ 4,  5,  6,  7],
#             [ 8,  9, 10, 11]])

result1 = f.reduce(a, axis=0)
print(result1)
# result1 == array([12, 15, 18, 21])

result2 = f.reduce(a, axis=1)
print(result2)
# result2 == array([ 6, 22, 38])

result3 = f.accumulate(a)
print(result3)
# result3 == array([[ 0,  1,  2,  3],
#                   [ 4,  6,  8, 10],
#                   [12, 15, 18, 21]])

result4 = f.accumulate(a, axis=1)
print(result4)
```

[官方文檔](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-vectorize-decorator)

#### guvectorize
generalized universal functions，強化版的 vectorize，接受任意形狀元素的輸入輸出。

[官方文檔](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-guvectorize-decorator)

#### jitclass
把 class 中所有 methods 都用 numba 優化，還在早期版本。使用 jit class 一定是 nopython 模式。

個人感覺不好用，因為你要給出 class 類所有成員的資料類型，所以不如直接在外面寫好函式，再到 class 中定義 method 直接呼叫寫好的函式。

[有使用到 jitclass 的教學](https://curiouscoding.nl/posts/numba-cuda-speedup/)  
[官方文檔](https://numba.readthedocs.io/en/stable/user/jitclass.html)  

#### stencil
用於簡化固定模式（stencil kernel）進行的操作以提升程式碼可讀性，例如對上下左右取平均，可以寫成如下方形式，可讀性高，效能也和 jit 一樣。

```py
import time
import numpy as np
from numba import stencil, njit

@stencil
def kernel1(a):
    # a 代表套用核心的輸入陣列
    return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])

@njit
def kernel1_jit(a):
    return kernel1(a)

def kernel1_python(a):
    result = np.zeros_like(a)
    for i in range(1, a.shape[0] - 1):
        for j in range(1, a.shape[1] - 1):
            result[i, j] = 0.25 * (a[i, j + 1] + a[i + 1, j] + a[i, j - 1] + a[i - 1, j])
    return result

@njit
def kernel1_python_jit(a):
    result = np.zeros_like(a)
    for i in range(1, a.shape[0] - 1):
        for j in range(1, a.shape[1] - 1):
            result[i, j] = 0.25 * (a[i, j + 1] + a[i + 1, j] + a[i, j - 1] + a[i - 1, j])
    return result


n_runs = 100
input_array = np.random.rand(5000, 5000)

# 第一次運行的初始化，第二次以後才是單純的執行時間
output_array_stencil = kernel1_jit(input_array)
output_array_python = kernel1_python_jit(input_array)

start = time.time()
for _ in range(n_runs):
    kernel1_jit(input_array)
end = time.time()
print(f"stencil: {(end - start)/n_runs} 秒")

start = time.time()
for _ in range(n_runs):
    kernel1_python_jit(input_array)
end = time.time()
print(f"pure jit: {(end - start)/n_runs} 秒")

# Compare the results
print("Are the outputs equal?", np.array_equal(output_array_stencil, output_array_python))

# 輸出

# stencil: 0.03909627914428711 秒
# pure jit: 0.038599507808685304 秒
# Are the outputs equal? True
```

### 提前編譯
numba 主要是即時編譯，但也支援像 C 語言一樣[提前編譯](https://numba.readthedocs.io/en/stable/user/pycc.html)完才能執行。

- 優點
    - 執行時不需 numba 套件
    - 沒有編譯時間開銷  
- 缺點
    - 不支援 ufuncs
    - 必須明確指定函式簽名 (signatures)
    - 導出的函式不會檢查傳遞的參數類型，調用者需提供正確的類型。
    - AOT 編譯生成針對 CPU 架構系列的通用程式碼（如 "x86-64"），而 JIT 編譯則生成針對特定 CPU 型號的優化程式碼。


### Signature
顯式的告訴 numba 型別，用於輸入維度可變，或者使用 AoT 編譯等，有無標示對效能不會有太大影響。[可用的 signature 列表](https://numba.pydata.org/numba-doc/dev/reference/types.html)。

- [輸入維度可變](https://stackoverflow.com/questions/66205186/python-signature-with-numba)，包含 guvectorize 和 [jitclass](https://numba.readthedocs.io/en/stable/user/jitclass.html)
- [AoT 編譯](https://numba.readthedocs.io/en/stable/user/pycc.html#limitations): 限制需要顯式指定 signature

### numba.typed.dict
作為數值模擬我們一定會遇到參數量超多的問題，numba 其實支援[用 dict 傳遞參數](https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict)。

### jit_module
開發者用，讓整個模組的函式都自動被 jit 裝飾。除了官方文檔，這裡節錄原始碼中的註解：

> Note that ``jit_module`` should only be called at the end of the module to be jitted. In addition, only functions which are defined in the module ``jit_module`` is called from are considered for automatic jit-wrapping.

https://numba.pydata.org/numba-doc/dev/user/jit-module.html

## 常見問題
1. SIMD 指令集？  
numba 不會用，但是 numba 使用的 LLVM 會用到。

2. 我要學會寫平行處理？  
不用，網路上在亂教，numba 會自動平行處理而且比手寫還好。

2. [可不可以把函式當參數給 numba 優化？](https://numba.readthedocs.io/en/stable/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function)  
可以，但是會造成額外開銷，請考慮工廠模式。

3. 提前編譯執行效率會變高嗎？  
根據提前編譯文檔，他會生成最泛用的函式而不是最符合當前 CPU/GPU 的函式，所以不會。

4. Numba JIT 和 Python JIT 一樣嗎？  
[不確定]根據這個影片說明 [CPython JIT](https://www.youtube.com/watch?v=SNXZPZA8PY8) 的核心理念是 JIT，而筆者在文檔或者 repo 中完全搜不到熱點分析，應該是不一樣。

5. numba 可能會產生和 Numpy 不一樣的結果  
根據[浮點陷阱](https://numba.readthedocs.io/en/stable/reference/fpsemantics.html)，我們應該避免對同一矩陣重複使用 numba 運算以免越錯越多。


## See Also
這裡放筆者覺得有用的文章。

- [官方使用範例](https://numba.readthedocs.io/en/stable/user/examples.html)
- [How to Write Fast Numerical Code](https://people.inf.ethz.ch/markusp/teaching/263-2300-ETH-spring14/slides/06-locality-caches.pdf)
- [Profiling your numba code](https://pythonspeed.com/articles/numba-profiling/) 
- [The wrong way to speed up your code with numba](https://pythonspeed.com/articles/slow-numba/)  
- [28000x speedup with numba.CUDA](https://curiouscoding.nl/posts/numba-cuda-speedup/) CUDA 加速並且有完整的對比。  
- [用 numba 學 CUDA! 從入門到精通 (上)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8A-ede7b381f6c7) 非常長。
- [用 numba 學 CUDA! 從入門到精通 (下)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8B-770c11bffd37) 非常長。
- 非官方[中文文檔](https://github.com/apachecn/numba-doc-zh) 只更新到 0.44，按需觀看，舊版缺乏使用警告可能導致意想不到的錯誤。


## 附錄：詞彙表
- AOT  
  Compilation of a function in a separate step before running the program code, producing an on-disk binary object which can be distributed independently. This is the traditional kind of compilation known in languages such as C, C++ or Fortran.

- Python bytecode （字結碼）  
  The original form in which Python functions are executed. Python bytecode describes a stack-machine executing abstract (untyped) operations using operands from both the function stack and the execution environment (e.g. global variables).

- JIT  
  Compilation of a function at execution time, as opposed to ahead-of-time compilation.

- ufunc  
  A NumPy universal function. numba can create new compiled ufuncs with the @vectorize decorator.


## 結語
目標讀者其實就是在說通訊系，也就是當年的自己。另外看到別篇文章結尾感謝部門其餘四個人，所以總共五個人討論出來才寫出 numba 文章，雖然晚了他一年，但筆者當年可是研究生，一個人自己學會用 numba...夭壽實驗室。

>內容基於 numba 文檔，作者：Anaconda, Inc.，授權：BSD 2-Clause。
>
>- GitHub: https://github.com/numba/numba
>- 文檔: https://numba.readthedocs.io/