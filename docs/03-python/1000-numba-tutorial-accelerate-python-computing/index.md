---
title: 使用 Numba 加速 Python 科學計算
description: 最快、最正確、最完整的 Numba 教學：使用 Numba 加速 Python 科學計算。坑筆者都踩過了只要照做可以得到最好性能，不會漏掉任何優化可能；除此之外本文第一不廢話，第二上手極快，第三介紹如何除錯和優化，第四補充進階使用方式，第五給出「精選有用的延伸閱讀」，不是給沒用文章，第六也是最重要，筆者可以很自信的說本文是中文圈最詳細教學。
tags:
  - Programming
  - Python
  - Numba
  - 教學
keywords:
  - Programming
  - Python
  - Numba
  - 教學
last_update:
  date: 2024-10-03 GMT+8
  author: zsl0621
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 使用 Numba 加速 Python 科學計算
鑑於繁體中文資訊少，最近剛好又重看了一下文檔，於是整理資訊分享給大家。本篇的目標讀者是沒學過計算機的初階用戶到中階用戶都可以讀，本篇教學已經覆蓋到絕大部分使用場景，只有 CUDA 沒有覆蓋到，但也提供相關教學連結。

- **為甚麼選擇此教學**  
    >最快、最正確、最完整   
    
    坑筆者都踩過了只要照做可以得到最好性能，不會漏掉任何優化可能；除此之外本文第一不廢話，第二上手極快，第三介紹如何除錯和優化，第四補充進階使用方式，第五給出「精選有用的延伸閱讀」，不是給沒用文章，第六也是最重要，**筆者可以很自信的說本文是中文圈最詳細教學**。

- **如何閱讀本文**  
不用害怕文章看似很長，初學者先不要使用平行計算功能，只需看到<u>進一步優化效能</u>即可快速上手 Numba；還要更快再看<u>自動平行化與競爭危害</u>；對於大多數人而言都不用看<u>進階使用</u>；如果你急到不行，看完<u>一分鐘學會 Numba</u> 後直接跳到<u>如何除錯</u>。

:::info 寫在前面

不要看 [**舊版**，左上角版本號 0.52](https://numba.pydata.org/) 的文檔！內容缺失，偏偏舊版文檔 Google 搜尋在前面，一不小心就點進去了。

:::

## 簡介：Numba 是什麼？
Numba 是一個將 Python 和 Numpy 程式碼轉換為快速的機器碼的即時編譯器 (Just-In-Time Compiler, JIT)。

Python 之所以慢的原因是身為動態語言，在運行時需要額外開銷來進行類型檢查，需經過字節碼轉譯和虛擬機執行，還有 GIL 進一步限制效能（見[附錄](/docs/python/numba-tutorial-accelerate-python-computing#附錄)），於是 Numba 就針對這些問題來解決，以下是他的主要特色：

- 靜態類型推斷：Numba 在編譯時分析程式碼推斷變數類型，避免額外的型別檢查。
- 即時編譯：將 Python 程式碼編譯成優化的機器碼。
- 平行化：Numba 支援平行計算。
- 向量化：使用 LLVM (LLVM 調用 SIMD) 指令集，將迴圈中操作向量化。

## 我是否該選擇 Numba？

> Q: [哪些程式適合 Numba](https://numba.pydata.org/numba-doc/dev/user/5minguide.html#will-numba-work-for-my-code)  

大量包含迴圈的 Numpy 數值運算，且不涉及 I/O 操作，例如 pandas。(If your code is numerically orientated (does a lot of math), uses NumPy a lot and/or has a lot of loops, then numba is often a good choice.)

> Q: Numba 有什麼特點？  

1. 簡單：只要一行裝飾器就可以加速原本程式碼。
2. 快速：專為科學計算而生，被設計成和 Numpy 協同工作（但也可以加速 Python 語法）。
3. 方便：支援 **自動** 平行化計算，效能甚至比一般人手寫的[更好](/docs/python/numba-tutorial-accelerate-python-computing#自動平行化)。
4. 強大：支援 **CUDA** 以顯示卡執行高度平行化的計算。
5. 通用：除了即時編譯也支援提前編譯，讓程式碼在沒有 Numba 或要求首次執行速度的場景應用。

> Q: 和競爭品如何選擇？  

常見的競爭選項有 Ray 和 Dask  
- Ray: 用於多台電腦的分布式計算，但是單機也可以優化平行計算。
- Dask: 用於巨量數據的平行處理，良好支援 pandas，也可用於分布式計算。
- Numba: 最大的差異是**從編譯的程式碼就更快**，並且有**平行處理優化**，但是著重在 Numpy 和單機操作。

經過這三個問題我們可以很清楚的知道，如果單純的想加速 Numpy 計算速度，**Numba 絕對是是第一選擇**，因為除了支援平行處理，連編譯的程式碼都更快。其實他們並不是互斥關係，比如說我們可以結合 Dask + Numba 打一套[組合拳](/docs/python/numba-tutorial-accelerate-python-computing#see-also)。

## 基礎使用
說是基礎使用，但是已經包含九成的使用情境。

### 一分鐘學會 Numba

比官方的五分鐘教學又快五倍，夠狠吧。測試對陣列開根號後加總，比較每個方法的時間，分別是有沒有使用 numba 和有沒有使用陣列計算，比較總共四個函式的運算時間差異。
```py
import numpy as np
import time
from numba import jit, prange


@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def numba_loop(arr):
    # Numba + Loop
    bias = 2
    total = 0
    for x in prange(len(arr)):   # Numba likes loops
        total += np.sqrt(x)      # Numba likes numpy
    return bias + total          # Numba likes broadcasting


def python_loop(arr):
    # Python + Loop
    bias = 2
    total = 0.0
    for x in arr:
        total += np.sqrt(x)
    return bias + total


@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def numba_arr(arr):
    # Numba + Vector
    bias = 2
    return bias + np.sum(np.sqrt(arr))


def python_arr(arr):
    # Python + Vector
    bias = 2
    return bias + np.sum(np.sqrt(arr))


n_runs = 1000
n = 10000000
arr = np.arange(n)

# 第一次運行的初始化，第二次以後才是單純的執行時間
result_python_arr = python_arr(arr)
result_numba_arr = numba_arr(arr)
result_numba = numba_loop(arr)

start = time.time()
result_python = python_loop(arr)
end = time.time()
print(f"Python迴圈版本執行時間: {end - start} 秒")

start = time.time()
result_python_arr = python_arr(arr)
end = time.time()
print(f"Python陣列版本執行時間: {end - start} 秒")

start = time.time()
for _ in range(n_runs):
    result_numba = numba_loop(arr)
end = time.time()
print(f"Numba迴圈版本執行時間: {(end - start)/n_runs} 秒")

start = time.time()
for _ in range(n_runs):
    result_numba_arr = numba_arr(arr)
end = time.time()
print(f"Numba陣列版本執行時間: {(end - start)/n_runs} 秒")

print("Are the outputs equal?", np.isclose(result_numba, result_python))
print("Are the outputs equal?", np.isclose(result_numba_arr, result_python_arr))

# Python迴圈版本執行時間: 9.418870210647583 秒
# Python陣列版本執行時間: 0.021904706954956055 秒
# Numba迴圈版本執行時間: 0.0013016948699951171 秒
# Numba陣列版本執行時間: 0.0024524447917938235 秒
# Are the outputs equal? True
# Are the outputs equal? True
```

這個實驗中我們可以看到使用 Numba 後速度還可以額外接近兩倍，但是也可以發現一個有趣的事實：「迴圈版本比陣列版本更快」，這引導我們到第一個重點 **Numba likes loops**，他還喜歡的另外兩個是 **Numpy** 和 **matrix broadcasting**。

比對函式差異可以看到使用方式很簡單，在要優化的函式前加上 `@jit` 裝飾器，接著在要平行化處理的地方顯式的改為 prange 就完成了。裝飾器的選項[有以下幾個](https://numba.readthedocs.io/en/stable/user/jit.html#compilation-options)：

| 參數      | 說明                                                      |
|----------|-----------------------------------------------------------|
| nopython | 是否嚴格忽略 Python C API，此參數影響速度最大                  |
| fastmath | 是否放寬 IEEE 754 的精度限制以獲得額外性能                     |
| cache    | 是否將編譯結果寫入快取，避免每次呼叫 Python 程式時都需要編譯      |
| parallel | 是否使用平行運算                                             |
| nogil    | 是否關閉全局鎖                                              |

<br/>
<br/>

:::info 安全性提醒

1. fastmath: 測試時使用，離正式生產越近越該關閉。
2. 依筆者個人使用 fastmath/nogil 並沒有快多少，當然這是 case-specific，像文檔的範例就有差。
3. numba 每次編譯後全域變數會變為常數，在程式中修改該變數不會被函式察覺。

:::


:::danger 安全性警告！

**對於暫時不想處理競爭危害的用戶，請先不要使用 `parallel` `nogil` 方法。**
1. parallel/nogil: 小心[競爭危害](https://zh.wikipedia.org/zh-tw/%E7%AB%B6%E7%88%AD%E5%8D%B1%E5%AE%B3) (race condition)。簡單說明競爭危害，就是兩個線程一起處理一個運算 `x += 1`，兩個一起取值，結果分別寫回 x 的值都是 `x+1` 導致最終結果是 `x+1` 而不是預期的 `x+2`。
2. 雖然上面的範例顯示結果一致，但還是一定要 **避免任何可能的多線程問題！**

:::

<br/>

### 進一步優化效能
基礎使用章節已經包含官方文檔中所有效能優化技巧，只是沒有每個選項[各自對比](https://numba.readthedocs.io/en/stable/user/performance-tips.html#intel-svml)，這裡補充其他效能優化方式。

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
    - 別忘了扣掉首次執行需要消耗的編譯時間。
    - 檢查 I/O 瓶頸，不要放任何需要 I/O 的程式碼在函式中。
    - 總計算量太小。
    - 宣告後就不要修改矩陣維度或型別。
    - 語法越簡單越好，不要使用任何各種包裝，因為你不知道 Numba 是否支援。
    - 記憶體問題 [The wrong way to speed up your code with numba](https://pythonspeed.com/articles/slow-numba/)。

4. [threading layers 設定平行計算方式](https://numba.readthedocs.io/en/stable/user/threading-layer.html)
    - `default` provides no specific safety guarantee and is the default.
    - `safe` is both fork and thread safe, this requires the tbb package (Intel TBB libraries) to be installed.
    - `forksafe` provides a fork safe library.
    - `threadsafe` provides a thread safe library.

    ```py
    # 設定只使用兩個線程執行，此指令等效於 NUMBA_NUM_THREADS=2
    # 在某些情形下應該設定為較低的值，以便 numba 可以與更高層級的平行性一起使用。
    # 但是文檔沒有說是哪些情形
    set_num_threads(2)
    sen: %s" % threading_layer())
    ```

5. 使用 @guvectorize  
    故意放最後因為很奇怪，請見[下方說明](/docs/python/numba-tutorial-accelerate-python-computing#guvectorize)。

讀到這裡你已經學會基礎的使用方式，能夠簡單的使用 Numba。如果有競爭危害的知識再開啟自動平行化功能，否則請跳到[如何除錯](/docs/python/numba-tutorial-accelerate-python-computing#如何除錯)。

### 自動平行化與競爭危害
本章節整理自文檔 [Automatic parallelization with @jit](https://numba.readthedocs.io/en/stable/user/parallel.html#)，閱讀本章節前請先確保你對競爭危害有一定程度的理解，否則請跳過本章節，並且**不要開啟 parallel 和 nogil 功能**。

#### 自動平行化

> 設定 Numba 自動平行化的官方文檔，由於很精練，知識也很重要，所以翻譯完貼在這裡。  
> 簡單來說，網路上手刻平行化的人連文檔都沒看就開始亂寫文章了。Numba 支援自動平行化，並且快取優化更好，不需要手刻。

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

`prange` 不支援多個出口的迴圈，例如迴圈中間包含 `assert`。

:::

:::warning 迴圈變數隱性轉型

關閉平行化處理時迴圈變數 (induction variable) 沒有問題，和 Python 預設一樣使用有號整數。然而如果開啟平行化且範圍可被識別為嚴格正數，則會被自動轉型為 `uint64`，而 `uint64` 和其他變數計算時**有機會不小心的返回一個浮點數**。

:::



#### 平行化的優化技巧

介紹如何撰寫迴圈才可使 Numba 加速最大化的技巧。

1. **迴圈融合 (Loop Fusion)：** 將相同迴圈邊界的迴圈合併成一個大迴圈，提高資料局部性進而提升效能。
2. **迴圈序列化 (Loop Serialization)：** Numba 不支援巢狀平行化，當多個 `prange` 迴圈嵌套時只有最外層的 `prange` 迴圈會被平行化，內層的 `prange` 迴圈會被視為普通的 `range` 執行。
3. **提出不變的程式碼 (Loop Invariant Code Motion)：** 將不影響迴圈結果的語句移到迴圈外。
4. **分配外提 (Allocation Hoisting)**：範例是拆分 `np.zeros` 成 `np.empty` 和 `temp[:] = 0` 避免重複初始化分配。

進一步優化：使用診斷功能，請見 [Diagnostics your parallel optimization](https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics)。

### 如何除錯
當年在寫 Numba 在出現錯誤時 Numba 的報錯資訊不明確，現在不知道有沒有改進，那時的方法是「找到錯誤行數的方式是二分法直接刪程式碼到 Numba 不報錯」

錯誤通常使用 Numba 不支援的函式，除錯請先看函式是否支援以免當冤大頭，再來就是檢查變數型別錯誤，例如誤用不支援相加的不同的變數型別。

- [Supported Python features](https://numba.readthedocs.io/en/stable/reference/pysupported.html)
- [Supported NumPy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)

除了本人的土砲方式，也可以用 `@jit(debug=True)`，詳情請見 [Troubleshooting and tips](https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html)。

### 小結
1. Numba likes loops 在心裡默念十次
2. Numba likes NumPy functions
3. Numba likes NumPy broadcasting
4. 不要在函式內修改數據結構 
5. 保持順序記憶體讀取
6. 函式中不要包含 I/O 操作
7. ***還是 Numba likes loops***

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
除非你是進階用戶，否則 **你不應該看進階使用章節！** 看了反而模糊焦點，你應該把握好如何基礎使用，基礎使用已經包含了八成以上的使用情景。

進階使用裡面就只有 [numba.typed.dict](/docs/python/numba-tutorial-accelerate-python-computing#numbatypeddict) 你可以稍微看一下。

### 使用 CUDA 加速運算
> 這就是剩下的其中一成。

優化 CUDA 不像優化 CPU 加上裝飾器那麼簡單，而是要針對 CUDA 特別寫函式，導致程式只能在 GPU 上跑，所以筆者目前還沒寫過，不過基本注意事項一樣是注意 IO、工作量太小的不適合 CUDA。那比較什麼函式適合 CPU 而不是 CUDA 呢？

1. **順序處理而不是平行處理**，影像處理以外的演算法大概都是這類
2. 記憶體超過顯卡記憶體上限（註：不應該寫出這種程式，Numba likes loops）
3. 大量分支處理 (if-else)（註：不應該寫出這種程式，尤其在 Numba 中）
4. 顯卡雙精度浮點運算效能差，深度學習和遊戲都吃單精度，但是科學計算需要雙精度，我們又只能買到遊戲卡
5. 一些 library 只支援 CPU，這要試了才知道

如果你需要使用 CUDA，這裡也有好用的指南連結：

- [28000x speedup with numba.CUDA](https://curiouscoding.nl/posts/numba-cuda-speedup/)：使用 CUDA 加速並且有完整的對比。  
- [用 numba 學 CUDA! 從入門到精通 (上)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8A-ede7b381f6c7)
- [用 numba 學 CUDA! 從入門到精通 (下)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8B-770c11bffd37)


### Signature
顯式的告訴 numba 型別，用於輸入維度可變，或者使用 AoT 編譯等，有標示對也不會比較快。[可用的 signature 列表](https://numba.pydata.org/numba-doc/dev/reference/types.html)。

- [輸入維度可變](https://stackoverflow.com/questions/66205186/python-signature-with-numba)，包含 guvectorize 和 [jitclass](https://numba.readthedocs.io/en/stable/user/jitclass.html)
- [AoT 編譯](https://numba.readthedocs.io/en/stable/user/pycc.html#limitations): 限制需要顯式指定 signature


<details>
<summary>簡單的 Numba signature 範例和效能測試</summary>

這裡的結果很奇怪，我預期應該是相差不多，結果反而顯式比較慢。測試在 apple M1 上運行，也有可能是 apple silicon 在搞鬼，x86 的用戶可以在自己電腦上執行看看結果如何。

```py
import numpy as np
import numba as nb
import time


# 使用顯式 signature
@nb.jit("float64[:](float64[:], float64[:])", nopython=True)
def add_and_sqrt(x, y):
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = np.sqrt(x[i] + y[i])
    return result


# 不使用顯式 signature
@nb.jit(nopython=True)
def add_and_sqrt_no_sig(x, y):
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = np.sqrt(x[i] + y[i])
    return result


# 測試函數
def test_function(func, x, y, num_runs=1000):
    start_time = time.time()
    for _ in range(num_runs):
        _ = func(x, y)
    end_time = time.time()
    return (end_time - start_time) / num_runs


if __name__ == "__main__":
    x = np.random.random(100000)
    y = np.random.random(100000)

    _ = add_and_sqrt(x, y)
    _ = add_and_sqrt_no_sig(x, y)

    time_with_sig = test_function(add_and_sqrt, x, y)
    time_without_sig = test_function(add_and_sqrt_no_sig, x, y)

    print(f"使用顯式 signature 的平均執行時間: {time_with_sig:.6f} 秒")
    print(f"不使用顯式 signature 的平均執行時間: {time_without_sig:.6f} 秒")
    print(f"性能差異: {abs(time_with_sig - time_without_sig) / time_without_sig * 100:.2f}%")

    result_with_sig = add_and_sqrt(x, y)
    result_without_sig = add_and_sqrt_no_sig(x, y)
    print(f"結果是否相同: {np.allclose(result_with_sig, result_without_sig)}")

# 使用顯式 signature 的平均執行時間: 0.000104 秒
# 不使用顯式 signature 的平均執行時間: 0.000052 秒
# 性能差異: 99.58%
# 結果是否相同: True
```

</details>

<details>
<summary>複雜的 Numba signature 範例</summary>

修改自 https://stackoverflow.com/questions/30363253/multiple-output-and-numba-signatures 

```py
import numpy as np
import numba as nb

# 輸入兩個一維浮點數陣列，返回兩個一維浮點數陣列
@nb.jit(nb.types.UniTuple(nb.float64[:], 2)(nb.float64[:], nb.float64[:]), nopython=True)
def homogeneous_output(a, b):
    return np.sqrt(a), np.sqrt(b)

# 字串表示
@nb.jit('UniTuple(float64[:], 2)(float64[:], float64[:])', nopython=True)
def homogeneous_output_str(a, b):
    return np.sqrt(a), np.sqrt(b)

# 異質類型返回值
@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:,:]))(nb.float64[:], nb.int64[:,:]), nopython=True)
def heterogeneous_output(a, b):
    return np.sqrt(a), b.astype(np.float64)

# 使用字串表示
@nb.jit('Tuple((float64[:], float64[:,:]))(float64[:], int64[:,:])', nopython=True)
def heterogeneous_output_str(a, b):
    return np.sqrt(a), b.astype(np.float64)

if __name__ == "__main__":
    a = np.array([1., 4., 9., 16.], dtype=np.float64)
    b = np.array([25., 36., 49., 64.], dtype=np.float64)
    c = np.array([[1, 2], [3, 4]], dtype=np.int64)

    # 測試同質類型輸出
    result1, result2 = homogeneous_output(a, b)
    print("同質類型輸出 (使用 nb.types):")
    print("結果 1:", result1)
    print("結果 2:", result2)

    result1, result2 = homogeneous_output_str(a, b)
    print("\n同質類型輸出 (使用字符串):")
    print("結果 1:", result1)
    print("結果 2:", result2)

    # 測試異質類型輸出
    result3, result4 = heterogeneous_output(a, c)
    print("\n異質類型輸出 (使用 nb.types):")
    print("結果 3:", result3)
    print("結果 4:\n", result4)

    result3, result4 = heterogeneous_output_str(a, c)
    print("\n異質類型輸出 (使用字符串):")
    print("結果 3:", result3)
    print("結果 4:\n", result4)
```

</details>

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
from numba import vectorize
import numpy as np

@vectorize(["int32(int32, int32)",
            "int64(int64, int64)",
            "float32(float32, float32)",
            "float64(float64, float64)"])
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
generalized universal functions，強化版的 vectorize，允許輸入是任意數量的 ufunc 元素，接受任意形狀輸入輸出的元素。雖然文檔沒有說明 guvectorize 有效能優化，但是我實際測試[這篇文章](https://medium.com/capital-one-tech/dask-numba-for-efficient-in-memory-model-scoring-dfc9b68ba6ce)的結果發現 guvectorize 速度確實比較快。

一樣是一個裝飾器就完成，這裡附上

<Tabs>
  <TabItem value="1" label="命名方式示範：矩陣相乘">
  
    ```py
    from numba import guvectorize, prange
    import numpy as np
    import time


    # vanilla guvectorize
    # tuple 第一項設定輸入
    # 輸入：在 list 中設定選項，這裡可接受四種類型的輸入
    # 輸出：只需定義維度
    # 輸入 "C"：guvectorize 不需要 return，而是把回傳值直接寫入輸入矩陣 C
    @guvectorize(
        [
            "float64[:,:], float64[:,:], float64[:,:]",
            "float64[:,:], float64[:,:], float64[:,:]",
            "int32[:,:], int32[:,:], int32[:,:]",
            "int64[:,:], int64[:,:], int64[:,:]",
        ],
        "(m,n),(n,p)->(m,p)",
    )
    def matrix_multiply(A, B, C):
        m, n = A.shape
        n, p = B.shape
        for i in range(m):
            for j in range(p):
                C[i, j] = 0
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]


    # guvectorize with prange
    # 測試 nopython, parallel
    @guvectorize(
        [
            "float64[:,:], float64[:,:], float64[:,:]",
            "float64[:,:], float64[:,:], float64[:,:]",
            "int32[:,:], int32[:,:], int32[:,:]",
            "int64[:,:], int64[:,:], int64[:,:]",
        ],
        "(m,n),(n,p)->(m,p)",
        nopython=True,
        target="parallel",
    )
    def matrix_multiply_prange(A, B, C):
        m, n = A.shape
        n, p = B.shape
        for i in prange(m):
            for j in range(p):
                C[i, j] = 0
                for k in range(n):
                    C[i, j] += A[i, k] * B[k, j]


    def run_benchmark():
        n_runs = 1
        N = 100
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)
        res_python = A @ B

        start = time.time()
        for _ in range(n_runs):
            C = np.empty((N, N), dtype=np.float64)
            matrix_multiply(A, B, C)
            print("Are the results the same?", np.allclose(C, res_python))
        end = time.time()
        print("Without prange: Total time for {} runs: {:.4f} seconds".format(n_runs, end - start))

        start = time.time()
        for _ in range(n_runs):
            C_prange = np.empty((N, N), dtype=np.float64)
            matrix_multiply_prange(A, B, C_prange)
            print("Are the results the same?", np.allclose(C_prange, res_python))
        end = time.time()
        print("With prange: Total time for {} runs: {:.4f} seconds".format(n_runs, end - start))

    run_benchmark()

    # Are the results the same? True
    # Without prange: Total time for 1 runs: 0.0012 seconds
    # Are the results the same? True
    # With prange: Total time for 1 runs: 0.0012 seconds
    ```

  </TabItem>

  <TabItem value="2" label="效能測試">

    ```py
    import numpy as np
    import time
    from numba import njit, guvectorize, prange

    n = 250000
    x = np.random.poisson(lam=5, size=n)
    y, z = np.random.normal(size=(n, 2)).T
    overlay = False
    n_runs = 100
    res = np.zeros((n, 15))


    # 原始Python版本
    def python_func(x, y, z, overlay=False):
        out = np.zeros((x.shape[0], 15))
        adj = 1.5 if overlay else 1.0
        for t in range(15):
            out[:, t] = t * x**2 + y - 2 * z - 2 * t
        return adj * out


    # @njit 優化版本
    @njit
    def jitted_func(x, y, z, overlay=False):
        out = np.zeros((x.shape[0], 15))
        adj = 1.5 if overlay else 1.0
        for t in range(15):
            out[:, t] = t * x**2 + y - 2 * z - 2 * t
        return adj * out


    # @njit + signature 優化版本
    @njit("float64[:,:](int64[:], float64[:], float64[:], boolean)")
    def jitted_func_with_signature(x, y, z, overlay=False):
        out = np.zeros((x.shape[0], 15))
        adj = 1.5 if overlay else 1.0
        for t in range(15):
            out[:, t] = t * x**2 + y - 2 * z - 2 * t
        return adj * out


    # @guvectorize 優化版本
    @guvectorize(
        "i8, f8, f8, b1, f8[:], f8[:]",
        "(), (), (), (), (specifySameDimension) -> (specifySameDimension)",
    )
    def fast_predict_over_time(x, y, z, overlay, _, out):
        adj = 1.5 if overlay else 1.0
        for t in range(len(out)):
            out[t] = adj * (t * x**2 + y - 2 * z - 2 * t)


    # 初始化編譯
    res_python = python_func(x, y, z, overlay)
    res_jitted = jitted_func(x, y, z, overlay)
    res_jitted_signature = jitted_func_with_signature(x, y, z, overlay)
    res_fast_pred = fast_predict_over_time(x, y, z, overlay, res)

    # 1. 測試原始Python版本
    start_time = time.time()
    for _ in range(n_runs):
        _ = python_func(x, y, z, overlay)
    end_time = time.time()
    print(f"Time: {(end_time - start_time) / n_runs:.6f} seconds, pure Python")

    # 2. 測試 @njit 優化版本
    start_time = time.time()
    for _ in range(n_runs):
        _ = jitted_func(x, y, z, overlay)
    end_time = time.time()
    print(f"Time: {(end_time - start_time) / n_runs:.6f} seconds, pure @njit")

    # 3. 測試 @njit + signature 優化版本
    start_time = time.time()
    for _ in range(n_runs):
        _ = jitted_func_with_signature(x, y, z, overlay)
    end_time = time.time()
    print(f"Time: {(end_time - start_time) / n_runs:.6f} seconds, @njit with signature")

    # 4. 測試 @guvectorize 優化版本
    start_time = time.time()
    for _ in range(n_runs):
        _ = fast_predict_over_time(x, y, z, overlay, res)
    end_time = time.time()
    print(f"Time: {(end_time - start_time) / n_runs:.6f} seconds, @guvectorize")

    print("Are the results the same?", np.array_equal(res_python, res_fast_pred))
    print("Are the results the same?", np.array_equal(res_python, res_jitted_signature))

    # Time: 0.039077 seconds, pure Python
    # Time: 0.027496 seconds, pure @njit
    # Time: 0.027633 seconds, @njit with signature
    # Time: 0.005587 seconds, @guvectorize
    # Are the results the same? True
    # Are the results the same? True
    ```

  </TabItem>
</Tabs>
  

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

### numba.typed.dict
作為數值模擬我們一定會遇到參數量超多的問題，numba 其實支援[用 dict 傳遞參數](https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict)。

### jit_module
開發者用，讓整個模組的函式都自動被 jit 裝飾。除了官方文檔，這裡節錄原始碼中的註解：

> Note that ``jit_module`` should only be called at the end of the module to be jitted. In addition, only functions which are defined in the module ``jit_module`` is called from are considered for automatic jit-wrapping.

https://numba.pydata.org/numba-doc/dev/user/jit-module.html

## 常見問題
1. 我要學會寫平行運算？  
不用，網路上在亂教，numba 會自動處理平行運算，而且效能比手寫還好。

2. [可不可以把函式當參數給 numba 優化？](https://numba.readthedocs.io/en/stable/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function)  
可以，但是會造成額外 call stack 開銷，請考慮工廠模式。

3. 提前編譯執行效率會變高嗎？  
不會。根據文檔，提前編譯會生成最泛用的函式而不是最符合當前 CPU/GPU 的函式。

4. Numba JIT 和 Python JIT 一樣嗎？  
[不確定]根據這個影片說明 [CPython JIT](https://www.youtube.com/watch?v=SNXZPZA8PY8) 的核心理念是 JIT，而筆者在文檔或者 Numba Github repo 中完全搜不到有關熱點分析的關鍵字，應該是不一樣。

5. numba 可能會產生和 Numpy 不一樣的結果  
根據[浮點陷阱](https://numba.readthedocs.io/en/stable/reference/fpsemantics.html)，我們應該避免對同一矩陣重複使用 numba 運算以免越錯越多。


## See Also
這裡放筆者覺得有用的文章。

- [官方使用範例](https://numba.readthedocs.io/en/stable/user/examples.html)
- 對 Numba 程式碼進行效能分析。  
[Profiling your numba code](https://pythonspeed.com/articles/numba-profiling/)
- 🔥 陣列運算降低 Numba 速度的範例  
[The wrong way to speed up your code with numba](https://pythonspeed.com/articles/slow-numba/)  
- 🔥 CUDA 加速並且有完整的對比，值得一看。  
[28000x speedup with numba.CUDA](https://curiouscoding.nl/posts/numba-cuda-speedup/)   
- 非常長的 CUDA 教學文章。  
[用 numba 學 CUDA! 從入門到精通 (上)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8A-ede7b381f6c7) 
- 非常長的 CUDA 教學文章。  
[用 numba 學 CUDA! 從入門到精通 (下)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8B-770c11bffd37)
- 🔥 使用 Dask + Numba 的簡單範例，其中包括 guvectoize 的使用，值得一看。  
[Dask + Numba for Efficient In-Memory Model Scoring](https://medium.com/capital-one-tech/dask-numba-for-efficient-in-memory-model-scoring-dfc9b68ba6ce) 
- 使用 Numba CUDA 功能加上 Dask 分散式加速運算並解決顯卡記憶體不足的問題。  
[Accelerated Portfolio Construction with Numba and Dask in Python](https://developer.nvidia.com/blog/accelerated-portfolio-construction-with-numba-and-dask-in-python/)
- 需要有計算機組織的知識才能讀懂得性能優化指南  
[How to Write Fast Numerical Code](https://people.inf.ethz.ch/markusp/teaching/263-2300-ETH-spring14/slides/06-locality-caches.pdf)

- 非官方[中文文檔](https://github.com/apachecn/numba-doc-zh) 只更新到 0.44，按需觀看，舊版缺乏使用警告可能導致意想不到的錯誤。


## 附錄
- [延伸閱讀] 全域直譯器鎖 GIL  
  用來限制同一時間內只能有一個執行緒執行 Python 字節碼的機制。Python 內建資料結構如字典等並非線程安全，所以需要 GIL 確保了多執行緒程式的安全性，避免競爭危害，然而也導致了多執行緒程式在工作中的效能低落。

- [[延伸閱讀](https://medium.com/citycoddee/python%E9%80%B2%E9%9A%8E%E6%8A%80%E5%B7%A7-5-python-%E5%88%B0%E5%BA%95%E6%80%8E%E9%BA%BC%E8%A2%AB%E5%9F%B7%E8%A1%8C-%E7%9B%B4%E8%AD%AF-%E7%B7%A8%E8%AD%AF-%E5%AD%97%E7%AF%80%E7%A2%BC-%E8%99%9B%E6%93%AC%E6%A9%9F%E7%9C%8B%E4%B8%8D%E6%87%82-553182101653)] Python 底層執行方式  
  Python 和 C/C++ 編譯成機器碼後執行不同，需要先直譯 (interprete) 成字節碼，再經由虛擬機作為介面執行每個字節碼的機器碼，再加上動態語言需要的型別檢查導致速度緩慢。
  
- AOT  
  Compilation of a function in a separate step before running the program code, producing an on-disk binary object which can be distributed independently. This is the traditional kind of compilation known in languages such as C, C++ or Fortran.

- Python bytecode （字節碼）  
  The original form in which Python functions are executed. Python bytecode describes a stack-machine executing abstract (untyped) operations using operands from both the function stack and the execution environment (e.g. global variables).

- JIT  
  Compilation of a function at execution time, as opposed to ahead-of-time compilation.

- ufunc  
  A NumPy universal function. numba can create new compiled ufuncs with the @vectorize decorator.

- jitted
  經過 JIT 編譯後的程式碼稱做 jitted。

- [延伸嘴砲] 甚至可以找到這樣的一篇論文：Dask & Numba: Simple libraries for optimizing scientific python code，恕我直言，這比熊文案還水。

## 結語
目標讀者其實就是在說通訊系，也就是當年的自己。另外看到別篇文章結尾感謝部門其餘四個人，所以總共五個人討論出來才寫出 numba 文章，當時雖然比他晚一年，但筆者當年可是研究生，一個人自己學會用 numba...夭壽實驗室。

開頭的最快、最正確和最完整，其實是自己看網路文章一直以來的不舒服感，完整的太詳細（跟讀文檔沒兩樣），快且正確的文章又不完整，好像永遠沒辦法兼顧。於是這篇和其他文章一樣，主要照顧初學者，讓初學者可以快速上手，讀起來又完整，而且內容還正確。

>內容基於 numba 文檔，作者：Anaconda, Inc.，授權：BSD 2-Clause。
>
>- GitHub: https://github.com/numba/numba
>- 文檔: https://numba.readthedocs.io/