---
title: Numba 教學：加速 Python 科學計算（上）
description: 你能找到最好的 Numba 教學！
sidebar_label: Numba 教學（上）
tags:
  - Python
  - Numba
  - Performance
  - 教學
keywords:
  - Python
  - Numba
  - Numpy
  - 教學
  - Speed-Up
  - Accelerate
  - Performance
last_update:
  date: 2024-10-18T00:00:00+08:00
  author: zsl0621
first_publish:
  date: 2024-10-18T00:00:00+08:00
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Numba 教學：加速 Python 科學計算（上）

> 你能找到最好的中文教學！

鑑於繁體中文資源匱乏，最近剛好又重新看了一下文檔，於是整理資訊分享給大家。本篇的目標讀者是沒學過計算機組織的初階用戶到中階用戶都可以讀，筆者能非常肯定的說這篇文章絕對是你能找到最好的教學。

:::info 寫在前面

**[舊版文檔](https://numba.pydata.org/numba-doc/dev/index.html)內容缺失！**，查看文檔時注意左上角版本號是否為 Stable，偏偏舊版文檔 Google 搜尋在前面，一不小心就點進去了。

:::

## Numba 簡介與比較

Numba 是一個針對 Python 數值和科學計算，使用 LLVM 函式庫優化效能的即時編譯器 (JIT compiler)，能顯著提升 Python 執行速度。

Python 速度慢的原因是身為動態語言，運行時需要額外開銷來進行類型檢查，轉譯成字節碼在虛擬機上執行[^python1]又多了一層開銷，還有 GIL 的限制進一步影響效能[^python2]，於是 Numba 就針對這些問題來解決，以下是它的優化機制：

- 靜態類型推斷：Numba 在編譯時分析程式碼推斷變數類型，避免型別檢查影響效能。
- 即時編譯：將函式編譯[^interpret]成針對當前 CPU 架構優化的機器碼，並且以 LLVM 優化效能。
- 向量化與平行化：透過 LLVM 使用 SIMD 進行向量化運算，並支援多核平行計算和 CUDA 運算。

[^python1]: [[延伸閱讀](https://medium.com/citycoddee/python%E9%80%B2%E9%9A%8E%E6%8A%80%E5%B7%A7-5-python-%E5%88%B0%E5%BA%95%E6%80%8E%E9%BA%BC%E8%A2%AB%E5%9F%B7%E8%A1%8C-%E7%9B%B4%E8%AD%AF-%E7%B7%A8%E8%AD%AF-%E5%AD%97%E7%AF%80%E7%A2%BC-%E8%99%9B%E6%93%AC%E6%A9%9F%E7%9C%8B%E4%B8%8D%E6%87%82-553182101653)] Python 底層執行方式  
  Python 和 C/C++ 編譯成機器碼後執行不同，需要先直譯 (interprete) 成字節碼 (.pyc)，再經由虛擬機作為介面執行每個字節碼的機器碼，再加上動態語言需要的型別檢查導致速度緩慢。
[^python2]: [延伸閱讀] 全域直譯器鎖 GIL  
  用來限制同一時間內只能有一個執行緒執行 Python 字節碼的機制。Python 內建資料結構如字典等並非線程安全，所以需要 GIL 確保了多執行緒程式的安全性，避免競爭危害，然而也導致了多執行緒程式在工作中的效能低落。

[^interpret]: 實際上 Numba 在首次執行時分析會 Python 的字節碼，並進行 JIT 編譯以了解函式的結構，再使用 LLVM 優化，最後轉換成經過 LLVM 最佳化的機器碼。與原生 Python 相比有以下優化：不使用 Python 字節碼/沒有 Python 內建型別檢查/沒有 Python 虛擬機開銷/多了 LLVM 最佳化的機器碼。首次執行前的分析就是 Numba 需要熱機的原因，官方文檔對 Numba 架構有[詳細說明](https://numba.readthedocs.io/en/stable/developer/architecture.html)，除此之外，很多 Numpy 函式 Numba 也使用[自己的實現](https://numba.discourse.group/t/how-can-i-make-this-function-jit-compatible/2631/2)。

Numba 適用於大量包含迴圈的 Numpy 數值運算，但不適合如 pandas 的 I/O 操作。除了 Numba 以外還有其他加速套件，那我們是否該選擇 Numba 呢？這裡列出常見的競爭選項，有 Cython、pybind11、Pythran 和 CuPy，我們從特點討論到性能，最後做出結論。

- **特點**
  - Numba：只支援 Numpy，並且有些方法不支援，如 [FFT](https://numba.discourse.group/t/rocket-fft-a-numba-extension-supporting-numpy-fft-and-scipy-fft/1657) 和[稀疏矩陣](https://numba-scipy.readthedocs.io/en/latest/reference/sparse.html)。
  - Pythran：和 Numba 相似，Numba 是即時編譯，Pythran 則是提前編譯。
  - Cython：需要學會他的獨特語法，該語法只能用在 Cython 是其最大缺點。
  - pybind11：就是寫 C++。
  - CuPy：為了 Numpy + Scipy 而生的 CUDA 計算套件。

- **效能**
    從 [Python 加速符文](https://stephlin.github.io/posts/Python/Python-speedup.html) 這篇文章中我們可以看到效能[^1]相差不大，除此之外，你能確定文章作者真的知道如何正確該套件嗎[^2]？因此，我們應該考量套件的限制和可維護性，而非單純追求效能極限，否則直接用 C 寫就可以了。

- **結論**  
    經過這些討論我們可以總結成以下
  - Numba：**簡單高效**，適合不熟悉程式優化技巧的用戶。缺點是因為太方便所以運作起來像是黑盒子，有時會感到不安心。
  - Pythran：搜尋結果只有一萬筆資料，不要折磨自己。
  - Cython：麻煩又不見得比較快。最大也是唯一的優點是支援更多 Python 語法，以及對程式行為有更多控制。
  - pybind11：適合極限性能要求，對程式行為有完全掌控的用戶。
  - CuPy：使用 CUDA，針對大量平行計算場景的最佳選擇。

[^1]: 效能測試跟測試者寫的程式有很大關係，以 [Python 加速符文](https://stephlin.github.io/posts/Python/Python-speedup.html) 為例，他測試 Numba 比 pybind11 慢，然而我在[下篇教學](numba-tutorial-accelerate-python-computing-2#guvectorize)中的 `測試計算弧長` 章節使用 guvectorize 功能就將速度提升了一個數量級，應該會比 pybind11 更快。除此之外，因為 Numba 支援 LLVM 所以他甚至可以[比普通的 C++ 還快](https://stackoverflow.com/questions/70297011/why-is-numba-so-fast)，所以當效能測試的程式碼碰巧對 LLVM 友善時速度就會變快，反之亦然。也就是說單一項的效能測試無法作為代表只能參考，尤其是當函式越簡單，Numba 越好優化，該效能測試的代表性就越低。

[^2]: 甚至連 geeksforgeeks 的文章 [Numba vs. Cython: A Technical Comparison](https://www.geeksforgeeks.org/numba-vs-cython-a-technical-comparison/) 都犯了一個最基本的錯誤：把 Numba 初次編譯的時間也算進去，該作者甚至都不覺得 Numba 比 Python 還久很奇怪，這麼大一個組織都錯了，我們還能期望網路上的文章多正確嗎？另外幫大家跑了他的程式，在 colab 上實際運行時間運行執行 1000 次取平均，兩者都是 `1.58ms`，因為他的程式碼簡單到即使 Numba 是自動優化的，也可以編譯出和 Cython 一樣速度的機器碼，除了證實前一個註腳，也說明該文章毫無參考價值。

## 安裝

安裝 Numba 以及相關的加速套件，包括 SVML (short vector math library) 向量化套件和 tbb/openmp 多線程套件，安裝後不需設定，Numba 會自行調用。

```sh
# conda
conda install numba
conda install intel-cmplr-lib-rt
conda install tbb
conda install anaconda::intel-openmp

# pip
pip install numba
pip install intel-cmplr-lib-rt
pip install tbb
pip install intel-openmp
```

安裝完成後重新啟動終端，使用 `numba -s | grep SVML` 檢查 SVML 是否成功被 Numba 偵測到，如果沒有，Linux 用戶可以用 `sudo ldconfig` 刷新 lib 連結。

## 基礎使用

說是基礎使用，但是已經包含七成的使用情境。

### 一分鐘學會 Numba

比官方的五分鐘教學又快五倍，夠狠吧。這個範例測試對陣列開根號後加總的速度，比較有沒有使用 Numba 和使用陣列/迴圈這四種方法的執行時間。

```py
import numpy as np
import time
from numba import jit, prange


# Numba Loop
@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def numba_loop(arr):
    bias = 2
    total = 0.0
    for x in prange(len(arr)):   # Numba likes loops
        total += np.sqrt(x)      # Numba likes numpy
    return bias + total          # Numba likes broadcasting


# Python Loop，沒有使用裝飾器
def python_loop(arr):
    bias = 2
    total = 0.0
    for x in arr:
        total += np.sqrt(x)
    return bias + total


# Numba Vector
@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def numba_arr(arr):
    bias = 2
    return bias + np.sum(np.sqrt(arr))


# Python Vector，沒有使用裝飾器
def python_arr(arr):
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

可以看到使用方式很簡單，僅需在想要優化的函式前加上 `@jit` 裝飾器，接著在要平行化處理的地方顯式的改為 `prange` 就完成了。裝飾器的選項有以下幾個[^compile]：

| 參數      | 說明                                                      |
|----------|-----------------------------------------------------------|
| nopython | 是否嚴格忽略 Python C API <br/>此參數是整篇文章中影響速度最大的因素，使用 @njit 等價於啟用此參數              |
| fastmath | 是否放寬 IEEE 754 的精度限制以獲得額外性能                     |
| parallel | 是否使用迴圈平行運算                                         |
| cache    | 是否將編譯結果寫入快取，避免每次呼叫 Python 程式時都需要編譯      |
| nogil    | 是否關閉全局鎖，關閉後允許在多線程中同時執行多個函式實例           |

[^compile]: 這是最常使用的五個參數，除此之外還有官方文檔沒有的隱藏選項，故意把參數名稱打錯後會顯示，裡面比較好用的是 `inline='Always'` 和 `looplift`。

效能方面，在這個測試中我們可以看到使用 Numba 後速度可以提升約兩倍，也發現一個有趣的事實：「迴圈版本比陣列版本更快」，這引導我們到第一個重點 **Numba likes loops**，另外兩個是 **Numpy** 和 **matrix broadcasting**。

<br/>
<br/>

:::info 提醒

1. 這些選項的效能差異依照函式而有所不同。
2. Numba 每次編譯後全域變數會變為常數，在程式中修改該變數不會被函式察覺。

:::

:::danger 競爭危害

**對於暫時不想處理競爭危害的用戶，請先不要使用 `parallel` 和 `nogil` 選項。**

1. 開啟 parallel/nogil 選項時必須小心[競爭危害](https://zh.wikipedia.org/zh-tw/%E7%AB%B6%E7%88%AD%E5%8D%B1%E5%AE%B3) (race condition)。  
簡單解釋競爭危害，兩個線程一起處理一個運算 `x += 1`，兩個一起取值，結果分別寫回 x 的值都是 `x+1` 導致最終結果是 `x+1` 而不是預期的 `x+2`。
2. 雖然上面的範例顯示結果一致，但還是一定要 **避免任何可能的多線程問題！**

:::

<br/>

### 進一步優化效能

基礎使用章節已經涵蓋[官方文檔](https://numba.readthedocs.io/en/stable/user/performance-tips.html#intel-svml)中的所有效能優化技巧，這裡補充進階的優化方式。

1. 使用 Numba 反而變慢
    - 別忘了扣掉首次執行需要消耗的編譯時間。
    - 檢查 I/O 瓶頸，不要放任何需要 I/O 的程式碼在函式中。
    - 總計算量太小。
    - 宣告後就不要修改矩陣維度或型別。
    - 語法越簡單越好，不要使用任何各種包裝，因為你不知道 Numba 是否支援。
    - 記憶體問題 [The wrong way to speed up your code with numba](https://pythonspeed.com/articles/slow-numba/)。
    - 分支預測問題 [Understanding CPUs can help speed up Numba and NumPy code](https://pythonspeed.com/articles/speeding-up-numba/)

2. 使用 `@vectorize` 或 `@guvectorize` 向量化  
    中文教學幾乎沒人提到向量化到底在做什麼。向量化裝飾器除了使函式支援 ufunc 以外還可以**大幅提升效能**，詳細說明請見[教學](numba-tutorial-accelerate-python-computing-2#vectorize)。

3. 使用[第三方套件](https://github.com/pythonspeed/profila)進行效能分析。
  
### fastmath

筆者在這裡簡單的討論一下 fastmath 選項。

雖然 fastmath 在文檔中只說他放寬了 IEEE 754 的精度限制，沒有說到的是他和 SVML 掛勾，但筆者以此 [Github issue](https://github.com/numba/numba/issues/5562#issuecomment-614034210) 進行測試，如果顯示機器碼 `movabsq $__svml_atan24` 代表安裝成功，此時我們將 fastmath 關閉後發現向量化失敗，偵錯訊息顯示 `LV: Found FP op with unsafe algebra.`。

為甚麼敢說本篇是最正確的教學，對於其他文章我就問一句話， **效能測試時有裝 SVML 嗎？** 這甚至都不用改程式就可以帶來極大幅度的效能提升，但是筆者從來沒看過任何文章提到過。

### 如何除錯

Numba 官方文檔有如何除錯的教學，使用 `@jit(debug=True)`，詳情請見 [Troubleshooting and tips](https://numba.readthedocs.io/en/stable/user/troubleshoot.html)。

另外一個是筆者的土砲方法，當年在寫 Numba 在出現錯誤時 Numba 的報錯資訊不明確，那時的土砲方法是「找到錯誤行數的方式是二分法直接刪程式碼到 Numba 不報錯」

錯誤通常來自於使用 Numba 不支援的函式，除錯請先看函式是否支援以免當冤大頭，再來就是檢查變數型別錯誤，例如誤用不支援相加的不同的變數型別。

- [Supported Python features](https://numba.readthedocs.io/en/stable/reference/pysupported.html)
- [Supported NumPy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)

### 小結

1. Numba likes loops 在心裡默念十次
2. Numba likes NumPy functions
3. Numba likes NumPy broadcasting
4. 記憶體、I/O 操作、分支預測是三大效能下降主因
7. 所有優化方式都是 case-specific，不能說哪些項目效能一定很好，一切取決於被編譯的程式碼如何設計，如果程式依照以上設計還是很慢，試試看開關選項，或者嘗試向量化裝飾器。
8. ***還是 Numba likes loops***

讀到這裡你已經學會基礎，但是包含大部分場景的使用方式。如果有競爭危害的知識再開啟自動平行化功能，否則請務必關閉以免跑很快但全錯。接下來建議先跳到 [See Also](#see-also) 看延伸閱讀，裡面包含各種速度優化方式。

---

## 自動平行化與競爭危害

本章節對官方文檔 [Automatic parallelization with @jit](https://numba.readthedocs.io/en/stable/user/parallel.html#) 進行翻譯和重新編排，如果不熟悉競爭危害建議**避免啟用 parallel 和 nogil 功能**。

### 自動平行化

> 設定 Numba 自動平行化的官方文檔，由於內容已經很精練，知識也很重要，所以翻譯完貼在這裡。  

在 `jit()` 函式中設置 `parallel` 選項，可以啟用 Numba 的轉換過程，嘗試自動平行化函式（或部分函式）以執行其他優化。目前此功能僅適用於CPU。

一些在用戶定義的函式中執行的操作（例如對陣列加上純量）已知具有平行語意。用戶的程式碼可能包含很多這種操作，雖然每個操作都可以單獨平行化，但這種方法通常會因為快取行為不佳而導致性能下降。相反地，通過自動平行化，Numba 會嘗試識別用戶程式碼中的這類操作並將相鄰的操作合併到一起，形成一個或多個自動平行執行的 kernels。這個過程是完全自動的，無需修改用戶程式碼，這與 Numba 的 `vectorize()` 或 `guvectorize()` 機制形成對比，後者需要手動創建並行 kernels。

- [**支援的運算符**](https://numba.readthedocs.io/en/stable/user/parallel.html#supported-operations)  
此處列出所有帶有平行化語意的運算符，Numba 會試圖平行化這些運算。

- **顯式的標明平行化的迴圈**  
使用平行化時，需使用 `prange` 取代 `range` 顯式的標明被平行化的迴圈，對於巢狀的 `prange` ，Numba 只會平行化最外層的迴圈。在裝飾器中設定 `parallel=False` 會導致 `prange` 回退為一般的 `range`。

:::warning 文檔翻譯問題

[中文文檔中的 reduction 翻譯錯誤](https://apachecn.github.io/numba-doc-zh/#/docs/21)，這裡分為兩種情況，一是平行化處理的術語 parallel reduction[^reduction2]，指的是「將各個執行緒的變數寫回主執行緒」，二是減少，代表該函式降低輸入維度，全部翻譯成減少顯然語意錯誤。

<!-- [^reduction1]: [平行程式設計的簡單範例](https://datasciocean.tech/others/parallel-programming-example/) -->
[^reduction2]: [Avoid Race Condition in Numba](https://stackoverflow.com/questions/61372937/avoid-race-condition-in-numba)：此文章詢問關於多線程競爭該如何解決，並且使用 `config.NUMBA_NUM_THREADS` 顯式進行線程 reduction。
:::

### 競爭危害

整理官方文檔中展示競爭危害的簡易範例、解決方式和正確使用方式。

<!-- 顯示出競爭危害的存在，請不要錯誤的推斷為 scalar 運算可以避免而 vector 運算不行，**任何時候我們都應該避免競爭危害的可能**。那我們就不能寫 for 迴圈了嗎？其實有其他方法，例如這下面的解決方式和正確使用範例。 -->

<!-- <details>
<summary>競爭危害範例</summary> -->

<Tabs>
  <TabItem value="1" label="發生競爭危害的範例">

```py
import numpy as np
from numba import njit, prange


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


x = np.random.randint(-100, 100, size=10000000)

result_numba = prange_wrong_result_numba(x)
result_python = prange_wrong_result_python(x)
print("Are the outputs equal?", np.allclose(result_numba, result_python))

result_numba_mod = prange_wrong_result_mod_numba(x)
result_python_mod = prange_wrong_result_mod_python(x)
print("Are the outputs equal?", np.allclose(result_numba_mod, result_python_mod))

# 輸出

# Are the outputs equal? False
# Are the outputs equal? False
```

</TabItem>

  <TabItem value="2" label="解決方式">

```py
import numpy as np
from numba import njit, prange


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


x = np.random.randint(-100, 100, size=(1000000, 4))

result_numba_whole_arr = prange_ok_result_whole_arr(x)
result_python_whole_arr = prange_ok_result_whole_arr_python(x)
print("Are the outputs equal?", np.allclose(result_numba_whole_arr, result_python_whole_arr))

result_numba_outer_slice = prange_ok_result_outer_slice(x)
result_python_outer_slice = prange_ok_result_outer_slice_python(x)
print("Are the outputs equal?", np.allclose(result_numba_outer_slice, result_python_outer_slice))

# 輸出

# Are the outputs equal? True
# Are the outputs equal? True
```

</TabItem>

  <TabItem value="3" label="正確使用範例">

```py
import numpy as np
from numba import njit, prange


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


n = 10000000
A = np.random.randint(-100, 100, size=n)
result_numba_test = prange_test(A)
result_python_test = sum(A)
print("Are the outputs equal?", np.allclose(result_numba_test, result_python_test))

result_numba_prod = two_d_array_reduction_prod(n)
result_python_prod = np.power(2, n) * np.ones((13, 17), dtype=np.int_)
print("Are the outputs equal?", np.allclose(result_numba_prod, result_python_prod))


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

:::warning 迴圈引導變數的隱性轉型

關閉平行化時，迴圈變數（induction variable）與 Python 預設行為一致，使用有號整數。然而如果開啟平行化且範圍可被識別為嚴格正數，則會被自動轉型為 `uint64`，而 `uint64` 和其他變數計算時**有機會不小心的返回一個浮點數**。

:::

### 平行化的優化技巧

節錄官方文檔中介紹如何撰寫迴圈才可使 Numba 加速最大化的技巧。

1. **迴圈融合 (Loop Fusion)：** 將相同迴圈邊界的迴圈合併成一個大迴圈，提高資料局部性進而提升效能。
2. **迴圈序列化 (Loop Serialization)：** Numba 不支援巢狀平行化，當多個 `prange` 迴圈嵌套時只有最外層的 `prange` 迴圈會被平行化，內層的 `prange` 迴圈會被視為普通的 `range` 執行。
3. **提出不變的程式碼 (Loop Invariant Code Motion)：** 將不影響迴圈結果的語句移到迴圈外。
4. **分配外提 (Allocation Hoisting)**：範例是拆分 `np.zeros` 成 `np.empty` 和 `temp[:] = 0` 避免重複初始化分配。

進一步優化：使用診斷功能，請見 [Diagnostics your parallel optimization](https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics)。

### nogil 的作用

對於 nogil 沒概念的用戶，這個段落根據解釋 Numba nogil 選項到底做了什麼事。

> 此段落根據 [Why doesn't the following code need a python interpreter?](https://stackoverflow.com/questions/70433667/why-doesnt-the-following-code-need-a-python-interpreter) 改寫。

- 原理  
原理是 Numba 優化時除了將 Python 程式碼優化成機器碼以外，還會建立一個 Python 對象的包裝函式，使 Python 能夠調用這些機器碼。但是包裝函式仍舊在 CPython 層面，受到 GIL 制約，於是 nogil 用於提前關閉 GIL，計算完成再把鎖鎖上。

- 實際  
實際上，程式碼加上這行後並使用 threadpool 才會真正實現平行處理，否則即使有多個 thread 還是順序執行。

- 和 parallel 的差異  
parallel 將迴圈平行化處理，而 nogil 是一次執行多個函式實例。

## 進階使用

```sh
# 這是用來阻止你繼續讀的 placeholder！
 _   _                       _             
| \ | |  _   _   _ __ ___   | |__     __ _ 
|  \| | | | | | | '_ ` _ \  | '_ \   / _` |
| |\  | | |_| | | | | | | | | |_) | | (_| |
|_| \_|  \__,_| |_| |_| |_| |_.__/   \__,_|

```

除非你是進階用戶，否則 **你不應該看進階使用章節！** 看了反而模糊焦點，應該先掌握基礎使用，因為基礎使用已涵蓋七成以上的使用情境。

只有 [使用字典傳遞參數](numba-tutorial-accelerate-python-computing-2#dict-var) 和 [向量化裝飾器](numba-tutorial-accelerate-python-computing-2#vectorize) 可以先偷看。

文章太長了，進階使用部分請見[下篇](numba-tutorial-accelerate-python-computing-2)。

## 常見問題

1. 我要學會寫平行運算？  
不用，網路上在亂教，numba 會自動處理平行運算，官方文檔也表示其內建的自動平行化功能效能比手寫還好，下一篇文章我們會討論各種設定的效能。

2. [可不可以把函式當參數給 numba 優化？](https://numba.readthedocs.io/en/stable/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function)  
可以，但是會造成額外 call stack 開銷，請考慮工廠模式。

3. 提前編譯執行效率會變高嗎？  
不會。根據文檔，提前編譯會生成最泛用的函式而不是最符合當前 CPU/GPU 的函式。

4. Numba JIT 和 Python JIT 一樣嗎？  
根據 [PEP 744](https://peps.python.org/pep-0744/) CPython JIT 使用 micro-ops 和 copy-and-patch 技術，並且使用運行時的分析資訊進行優化，而 Numba 是基於 LLVM 編譯器優化數值運算的 JIT 編譯器，筆者在文檔或者 Numba Github repo 中也完全搜不到有關熱點分析的關鍵字，都是 JIT，實際上略有不同。

5. Numba 可能會產生和 Numpy 不一樣的結果  
根據[浮點陷阱](https://numba.readthedocs.io/en/stable/reference/fpsemantics.html)，我們應該避免對同一矩陣重複使用 Numba 運算以免計算誤差被放大。

## See Also

這裡放筆者覺得有用的文章。

- [官方使用範例](https://numba.readthedocs.io/en/stable/user/examples.html)
- 🔥🔥🔥 **非常優質的連續三篇系列文章，你最好把這裡全部看過！**  
[Making Python extremely fast with Numba: Advanced Deep Dive (1/3)](https://medium.com/@mflova/making-python-extremely-fast-with-numba-advanced-deep-dive-1-3-4d303edeede4)  
[Making Python extremely fast with Numba: Advanced Deep Dive (2/3)](https://medium.com/@mflova/making-python-extremely-fast-with-numba-advanced-deep-dive-2-3-f809b43f8300)  
[Making Python extremely fast with Numba: Advanced Deep Dive (3/3)](https://medium.com/@mflova/making-python-extremely-fast-with-numba-advanced-deep-dive-3-3-695440b62030)  
- 🔥 對 Numba 程式碼進行效能分析。  
[Profiling your Numba code](https://pythonspeed.com/articles/numba-profiling/)
- 🔥 陣列運算降低 Numba 速度的範例。  
[The wrong way to speed up your code with numba](https://pythonspeed.com/articles/slow-numba/)  
- 🔥 分支預測如何降低程式碼速度，以強迫寫入解決。  
[Understanding CPUs can help speed up Numba and NumPy code](https://pythonspeed.com/articles/speeding-up-numba/)
- 影像處理演算法的優化：從基礎實現開始  
[Speeding up your code when multiple cores aren’t an option](https://pythonspeed.com/articles/optimizing-dithering/)
- 為每個線程建立 local storage 以提升效能  
[Tips for optimising parallel numba code](https://chryswoods.com/accelerating_python/numba_bonus.html)
- 🔥 CUDA 加速並且有完整的對比，值得一看。  
[28000x speedup with Numba.CUDA](https://curiouscoding.nl/posts/numba-cuda-speedup/)
- 非常長的 CUDA 教學文章。  
[用 Numba 學 CUDA! 從入門到精通 (上)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8A-ede7b381f6c7)
- 非常長的 CUDA 教學文章。  
[用 Numba 學 CUDA! 從入門到精通 (下)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8B-770c11bffd37)
- 使用 Dask + Numba 的簡單範例，其中包括 guvectoize 的使用。  
[Dask + Numba for Efficient In-Memory Model Scoring](https://medium.com/capital-one-tech/dask-numba-for-efficient-in-memory-model-scoring-dfc9b68ba6ce)
- 使用 Numba CUDA 功能加上 Dask 分散式加速運算並解決顯卡記憶體不足的問題。  
[Accelerated Portfolio Construction with Numba and Dask in Python](https://developer.nvidia.com/blog/accelerated-portfolio-construction-with-numba-and-dask-in-python/)
- 需要有計算機組織的知識才能讀懂得性能優化指南  
[How to Write Fast Numerical Code](https://people.inf.ethz.ch/markusp/teaching/263-2300-ETH-spring14/slides/06-locality-caches.pdf)

- 非官方[中文文檔](https://github.com/apachecn/numba-doc-zh) 只更新到 0.44，按需觀看，舊版缺乏使用警告可能導致意想不到的錯誤。

## 附錄

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

- 參考資料  
  雖然本文自己說面向中階用戶，但是在 Anaconda 爸爸的投影片裡面我們才在最初階而已 (p13)
  [Accelerated Computing in Python (with Numba)](https://indico.cern.ch/event/824917/contributions/3571661/attachments/1934964/3206289/2019_10_DANCE_Numba.pdf)

## 結語

長達一萬字的教學結束了，Markdown 總字數超過三萬，應該來個一鍵三連吧。

目標讀者其實就是在說通訊系，也就是當年的自己。

開頭的最快、最正確和最完整，其實是自己看網路文章一直以來的不舒服感，完整的太詳細（跟讀文檔沒兩樣），快且正確的文章又不完整，好像永遠沒辦法兼顧。於是本文和我寫的其他教學文章一樣，主要照顧初學者，讓初學者可以快速上手，讀起來又完整，而且內容還正確，當讀者不需要使用平行化時可以在十分鐘之內搞定 Numba，需要平行化或 vectorize 等高級使用技巧時也對網路上許多錯誤做出勘誤和實測結果，感謝能讀完的各位。

>內容基於 Numba 文檔，作者：Anaconda, Inc.，授權：BSD 2-Clause。
>
>- GitHub: https://github.com/numba/numba
>- 文檔: https://numba.readthedocs.io/
