---
title: Numba 教學：加速 Python 科學計算（下）
description: 你能找到最好的 Numba 教學！
sidebar_label: Numba 教學（下）
slug: /numba-tutorial-2
tags:
  - Python
  - Numba
  - Performance
  - 教學
keywords:
  - Python
  - Numba
  - Performance
  - Numpy
  - 教學
  - Speed-Up
  - Accelerate
last_update:
  date: 2024-10-18T00:00:00+08:00
  author: zsl0621
first_publish:
  date: 2024-10-18T00:00:00+08:00
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Numba 教學：加速 Python 科學計算（下）

除非你是進階用戶，否則 **你不應該看進階使用章節！** 看了反而模糊焦點，應該先掌握基礎使用，因為基礎使用已涵蓋七成以上的使用情境，基礎使用請看[教學上篇](numba-tutorial-1)。

只有 [使用字典傳遞參數](#dict-var) 和 [向量化裝飾器](#vectorize) 可以先偷看。

## 使用 CUDA 加速運算

[官方文檔](https://numba.readthedocs.io/en/stable/cuda/overview.html)

優化 CUDA 相較於針對 CPU 優化只要加上裝飾器來說更為複雜，因為需要對 CUDA 特別寫函式，導致程式只能在 GPU 上跑，所以筆者目前還沒寫過，不過基本注意事項一樣是注意 IO、工作量太小的不適合 CUDA。

<!-- 那比較什麼函式適合 CPU 而不是 CUDA 呢？

1. **順序處理而不是平行處理**，影像處理以外的演算法大概都是這類
2. 記憶體超過顯卡記憶體上限
3. 大量分支處理 (if-else)
4. 顯卡雙精度浮點運算效能差，深度學習和遊戲都吃單精度，但是科學計算需要雙精度，而我們又只能買到遊戲卡
5. 需要只支援 CPU 的 library

如果你需要使用 CUDA，這裡也有好用的指南連結：

- [28000x speedup with numba.CUDA](https://curiouscoding.nl/posts/numba-cuda-speedup/)：使用 CUDA 加速並且有完整的對比。  
- [用 numba 學 CUDA! 從入門到精通 (上)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8A-ede7b381f6c7)
- [用 numba 學 CUDA! 從入門到精通 (下)](https://medium.com/@spacetime0311/%E7%94%A8-numba-%E5%AD%B8-cuda-%E5%BE%9E%E5%85%A5%E9%96%80%E5%88%B0%E7%B2%BE%E9%80%9A-%E4%B8%8B-770c11bffd37) -->

經過一些研究後，筆者認為不該用 Numba 調用 CUDA，工欲善其事，既然程式碼沒有便攜性，那還不如直接用專門優化 CUDA 的套件。根據[此篇 reddit 的討論](https://www.reddit.com/r/Python/comments/xausj8/options_for_gpu_accelerated_python_experiments/)可以看到 CuPy 是一個不錯的選擇，是專門調用 CUDA 的套件。研究途中也發現一個新穎的套件 [Taichi](https://github.com/taichi-dev/taichi) 也可以調用 CUDA，稍微看過文檔後其特色大概在專攻<u>物理粒子計算</u>以及支援自動微分，官方也有[測試效能的文章](https://docs.taichi-lang.org/blog/taichi-compared-to-cub-cupy-numba)，根據該文章的測試結果，我們沒什麼理由使用 Numba 調用 CUDA。

## 使用字典傳遞參數{#dict-var}

[官方文檔](https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-dict)

在數值模擬中，我們一定會遇到參數量超多的問題，Numba 其實支援[用字典傳遞參數](https://stackoverflow.com/questions/55078628/using-dictionaries-with-numba-njit-function)。

## Signature

[官方文檔 + 可用的 signature 列表](https://numba.readthedocs.io/en/stable/reference/types.html#numbers)  

顯式的告訴 Numba 輸出輸入型別，某些功能強制要求標示，語法是 `list[輸出1(輸入1A, 輸入1B), 輸出2(輸入2A, 輸入2B)]`。`float64[:]` 表示一維陣列，`float64[:,:]` 表示二維陣列。

<details>
<summary>簡單的 Numba signature 範例</summary>

```py
import numpy as np
import numba as nb


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


if __name__ == "__main__":
    x = np.random.random(100000)
    y = np.random.random(100000)

    _ = add_and_sqrt(x, y)
    _ = add_and_sqrt_no_sig(x, y)

    result_with_sig = add_and_sqrt(x, y)
    result_without_sig = add_and_sqrt_no_sig(x, y)
    print(f"結果是否相同: {np.allclose(result_with_sig, result_without_sig)}")

# 結果是否相同: True
```

</details>

<details>
<summary>複雜的 Numba signature 範例</summary>

此範例演示一個函式包含多個輸出輸入，也支援多種輸出輸入維度。

```py
from numba import njit, float64, types
import numpy as np


@njit(
    [
        types.Tuple((float64, float64))(float64, float64),
        types.Tuple((float64[:], float64[:]))(float64[:], float64[:]),
    ]
)
def cosd(angle1, angle2):
    result1 = np.cos(np.radians(angle1))
    result2 = np.cos(np.radians(angle2))
    return result1, result2


# Test with single values
angle1 = 45.0
angle2 = 90.0
result1, result2 = cosd(angle1, angle2)
print(f"Results for single values: {result1}, {result2}")

# Test with arrays
angles1 = np.array([0.0, 45.0, 90.0])
angles2 = np.array([30.0, 60.0, 90.0])
results1, results2 = cosd(angles1, angles2)
print(f"Results for arrays:\n{results1}\n{results2}")
```

</details>

## 其他裝飾器

常見裝飾器有

- vectorize
- guvectorize
- jitclass
- stencil

### vectorize

[官方文檔](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-vectorize-decorator)

vectorize 裝飾器語法限制輸入輸出和所有運算都只能是純量不能是向量，允許把純量操作向量化，並且可以把函式當作 [Numpy ufunc](http://docs.scipy.org/doc/numpy/reference/ufuncs.html) 使用。官方文檔花了很大的篇幅在描述該方法可以簡單的建立 Numpy ufunc 函式，因為[傳統方法](https://numpy.org/devdocs/user/c-info.ufunc-tutorial.html)需要寫 C 語言。對於效能，文檔很帥氣的輕描淡寫了一句話：

> Numba will generate the surrounding loop (or kernel) allowing efficient iteration over the actual inputs.

官網對如何優化 jit 專門寫了一篇文章，對於 vectorize 的效能僅僅就只寫了這麼一句話，看起來此裝飾器重點好像不是擺在效能上，然而[此文章](https://medium.com/@mflova/making-python-extremely-fast-with-numba-advanced-deep-dive-2-3-f809b43f8300)中的 vectorize 速度比起 jit 又快了 20 倍！根據他的解釋，vectorize 會告訴額外訊息給 LLVM，於是 LLVM 就可以藉此使用 CPU 的向量運算指令集 SIMD。

下方是基礎語法範例，效能測試我們放到下面強化版的 `guvectorize`：

```py
# Edit from: https://github.com/numba/numba/blob/main/numba/tests/doc_examples/test_examples.py
# test_vectorize_multiple_signatures
from numba import vectorize
import numpy as np

# 格式是 list[輸出1(輸入1A, 輸入1B), 輸出2(輸入2A, 輸入2B)]
@vectorize(["int32(int32, int32)",
            "int64(int64, int64)",
            "float32(float32, float32)",
            "float64(float64, float64)"])
def f(x, y):
    return x + y   # 定義時，函式內只能進行純量運算

a = np.arange(6)
result = f(a, a)   # 使用時，允許輸入向量
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

### guvectorize

[官方文檔](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-guvectorize-decorator)

generalized universal vectorizer，強化版的 vectorize，允許輸入是任意數量的 ufunc 元素，接受任意形狀輸入輸出的元素。黑魔法又來了，這裡官方仍舊沒有描述效能，然而[這篇文章](https://medium.com/@mflova/making-python-extremely-fast-with-numba-advanced-deep-dive-3-3-695440b62030)中測試 `guvectorize` 竟然又更快，比 `vectorize` 還快了六倍。

附上語法範例和效能測試

<Tabs>

  <TabItem value="1" label="語法示範">

    ```py
    # 以矩陣相乘示範 guvectorize 的語法
    from numba import guvectorize, prange
    import numpy as np
    import time


    # 簽章格式：在 list 中設定輸入選項，依序填寫 "輸入1, 輸入2, 輸出"。此範例可接受四種類型的輸入
    # 輸出輸入維度：只需定義維度 (m,n),(n,p)->(m,p)，也可以空白表示未指定，函式中不可寫 return
    # 函式輸入：最後一項輸入是應該要被回傳的計算結果，guvectorize 不 return 而是直接將結果寫入輸入矩陣
    @guvectorize(
        [
            "float64[:,:], float64[:,:], float64[:,:]",
            "float32[:,:], float32[:,:], float32[:,:]",
            "int64[:,:], int64[:,:], int64[:,:]",
            "int32[:,:], int32[:,:], int32[:,:]",
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

<TabItem value="2" label="測試矩陣相乘">

程式碼來自於 [Dask + Numba for Efficient In-Memory Model Scoring](https://medium.com/capital-one-tech/dask-numba-for-efficient-in-memory-model-scoring-dfc9b68ba6ce)。

    ```py
    # 測試矩陣相乘的效能
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

    print("Are the results the same?", np.allclose(res_python, res_fast_pred))
    print("Are the results the same?", np.allclose(res_python, res_jitted_signature))

    # Time: 0.039077 seconds, pure Python
    # Time: 0.027496 seconds, pure @njit
    # Time: 0.027633 seconds, @njit with signature
    # Time: 0.005587 seconds, @guvectorize
    # Are the results the same? True
    # Are the results the same? True
    ```
</TabItem>

<TabItem value="3" label="測試邏輯迴歸">

這是[官方文檔的範例](https://numba.readthedocs.io/en/stable/user/parallel.html#examples)，文檔目的是說明使用 `guvectorize` 會把簡單的程式碼展開到很複雜，此範例中 `guvectorize` 比 `njit` 慢。

    ```py
    # 測試邏輯回歸的效能
    import time
    import numpy as np
    from numba import guvectorize, njit, prange


    # Python
    def logistic_regression_py(Y, X, w, iterations):
        for i in range(iterations):
            w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
        return w


    # njit
    @njit(parallel=True)
    def logistic_regression_njit(Y, X, w, iterations):
        for _ in prange(iterations):
            w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
        return w


    # guvectorize
    @guvectorize(
        ["void(float64[:], float64[:,:], float64[:], int64, float64[:])"],
        "(n),(n,m),(m),()->(m)",
        nopython=True,
        target="parallel",
    )
    def logistic_regression_guvec(Y, X, w, iterations, result):
        for i in prange(iterations):
            temp = np.zeros_like(w)
            for j in range(X.shape[0]):
                dot_product = np.dot(X[j].copy(), w.copy())
                sigmoid = 1.0 / (1.0 + np.exp(-Y[j] * dot_product))
                temp += (sigmoid - 1.0) * Y[j] * X[j]
            w -= temp / X.shape[0]


    np.random.seed(0)
    Y = np.abs(np.random.randn(4000))
    X = np.abs(np.random.randn(4000, 10))
    w = np.abs(np.random.randn(10))
    iterations = 100
    n_runs = 10

    w_guvec = np.zeros_like(w)
    logistic_regression_njit(Y, X, w.copy(), iterations)
    logistic_regression_guvec(Y, X, w.copy(), iterations, w_guvec)

    start_py = time.time()
    for _ in range(n_runs):
        w_py = logistic_regression_py(Y, X, w.copy(), iterations)
    end_py = time.time()
    print(f"Time: {(end_py - start_py) / n_runs:.6f} seconds, pure Python")

    start_njit = time.time()
    for _ in range(n_runs):
        w_njit = logistic_regression_njit(Y, X, w.copy(), iterations)
    end_njit = time.time()
    print(f"Time: {(end_njit - start_njit) / n_runs:.6f} seconds, @njit+parallel")

    start_guvec = time.time()
    for _ in range(n_runs):
        logistic_regression_guvec(Y, X, w.copy(), iterations, w_guvec)
    end_guvec = time.time()
    print(f"Time: {(end_guvec - start_guvec) / n_runs:.6f} seconds, guvectorize")

    # Time: 0.027217 seconds, pure Python
    # Time: 0.017704 seconds, @njit+parallel
    # Time: 0.100319 seconds, guvectorize
    ```

</TabItem>

<TabItem value="4" label="測試計算弧長">

本範例來自於 [Python 加速符文：高效能平行科學計算](https://stephlin.github.io/posts/Python/Python-speedup.html)，該文章沒有使用 `guvectorize` 就下結論了，這裡幫他補上後可以發現效能又快了將近**一個數量級**，優化後的效能甚至可能超過 pybind11。

    ```py
    # 測試計算弧長的效能
    import time
    import numpy as np
    from numba import guvectorize, njit, prange


    # Python 版本
    def arc_length_py(points: np.ndarray) -> float:
        piecewice_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.sum(piecewice_length)


    # njit 版本
    @njit(parallel=True)
    def arc_length_njit1(points: np.ndarray) -> float:
        length = 0

        for i in prange(points.shape[0] - 1):
            piecewice_length = np.sqrt(np.sum((points[i + 1] - points[i]) ** 2))
            length += piecewice_length

        return length


    # guvectorize 版本
    @guvectorize(["void(float64[:,:], float64[:])"], "(n,m)->()", nopython=True, target='parallel')
    def arc_length_guvec(points: np.ndarray, result: np.ndarray):
        length = 0

        for i in range(points.shape[0] - 1):
            piecewice_length = 0
            for j in range(points.shape[1]):
                piecewice_length += (points[i + 1, j] - points[i, j]) ** 2
            length += np.sqrt(piecewice_length)

        result[0] = length


    # 初始化
    n_runs = 100
    n_points = 10_000_000
    points = np.random.rand(n_points, 2)
    res = np.zeros(1)
    _ = arc_length_guvec(points, res)
    _ = arc_length_njit1(points)

    start_py = time.time()
    w_py = arc_length_py(points)
    end_py = time.time()
    print(f"Time: {(end_py - start_py):.6f} seconds, pure Python")

    start_njit = time.time()
    for _ in range(n_runs):
        w_njit = arc_length_njit1(points)
    end_njit = time.time()
    print(f"Time: {(end_njit - start_njit) / n_runs:.6f} seconds, njit")

    start_guvec = time.time()
    for _ in range(n_runs):
        w_guvec = arc_length_guvec(points, res)
    end_guvec = time.time()
    print(f"Time: {(end_guvec - start_guvec) / n_runs:.6f} seconds, guvectorize")

    isclose_njit = np.allclose(w_py, w_njit)
    isclose_guvec = np.allclose(w_py, res[0])
    print("Are the results the same? njit:", isclose_njit, "guvec:", isclose_guvec)

    # Time: 0.326715 seconds, pure Python
    # Time: 0.138515 seconds, njit
    # Time: 0.017213 seconds, guvectorize
    # Are the results the same? njit: True guvec: True
    ```

</TabItem>

</Tabs>

<br/>
:::tip 參數

guvectorize 和 vectorize 的 parallel 語法是 `target="option"`，選項有 cpu, parallel 和 cuda 三種。

如果測試數據太小 parallel 和 cuda 性能反而會下降，因為才剛搬東西到記憶體就結束了。官方給出的建議是使用 parallel 數據至少大於 1KB，至於 cuda 那肯定要再大更多。

:::

### jitclass

[官方文檔](https://numba.readthedocs.io/en/stable/user/jitclass.html)  

把 class 中所有 methods 都用 Numba 優化，還在實驗版本。

個人認為這種方法不太好用，因為需要明確指定 class 中所有成員的資料類型。不如直接在外面寫好 Numba 裝飾的函式，然後在 class 中定義方法來調用會更簡單，附上[有使用到 jitclass 的教學](https://curiouscoding.nl/posts/numba-cuda-speedup/)。

### stencil

[官方文檔](https://numba.readthedocs.io/en/stable/user/stencil.html)

用於固定模式（stencil kernel）的運算以簡化程式碼，例如對上下左右取平均，可以寫成如下方形式，可讀性高，專有名詞似乎叫做 stencil computing。

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

#### overload

[官方文檔](https://numba.pydata.org/numba-doc/dev/extending/overloading-guide.html)

除了手刻不支援的函式以外，Numba 提供了一個高階方式讓你替代不支援的函式，官方範例是使用 `@overload(scipy.linalg.norm)` 替代 `scipy.linalg.norm`，範例中使用手刻的 `_oneD_norm_2` 實現範數的實作。

這個裝飾器像他的名字一樣用於重載整個函式，用於修改原本的函式內容，是很高階的使用方式，除非必要不建議使用，會大幅增加程式維護難度。

## 線程設定

[官方文檔](https://numba.readthedocs.io/en/stable/user/threading-layer.html)

Numba 可以設定 threading layer 使用哪種方式管理，有以下四種選項：

- `default` provides no specific safety guarantee and is the default.
- `safe` is both fork and thread safe, this requires the tbb package (Intel TBB libraries) to be installed.
- `forksafe` provides a fork safe library.
- `threadsafe` provides a thread safe library.
<br/>

```py
# 設定只使用兩個線程執行，此指令等效於 NUMBA_NUM_THREADS=2
# 官方文檔說明：在某些情形下應該設定為較低的值，以便 numba 可以與更高層級的平行性一起使用（但是文檔沒有說是哪些情形）
set_num_threads(2)
sen: %s" % threading_layer())
```

## 提前編譯

[官方文檔](https://numba.readthedocs.io/en/stable/user/pycc.html)

Numba 主要是使用即時編譯，但也支援像 C 語言一樣提前編譯打包後執行。

- 優點
  - 執行時不需 Numba 套件。
  - 沒有編譯時間開銷。  
- 缺點
  - 不支援 ufunc。
  - 必須明確指定函式簽名 (signatures)。
  - 導出的函式不會檢查傳遞的參數類型，調用者需提供正確的類型。
  - AOT 編譯生成針對 CPU 架構系列的通用程式碼（如 "x86-64"），而 JIT 編譯則生成針對特定 CPU 型號的優化程式碼。

## jit_module

[官方文檔](https://numba.readthedocs.io/en/stable/user/jit-module.html)

開發者用，讓整個模組的函式都自動被 jit 裝飾。除了官方文檔，這裡節錄 Github 原始碼中的註解：

> Note that ``jit_module`` should only be called at the end of the module to be jitted. In addition, only functions which are defined in the module ``jit_module`` is called from are considered for automatic jit-wrapping.

## 結合分佈式計算

常見的分佈式工具有 Ray 和 Dask，比如說我們可以結合 Dask + Numba 打一套組合拳，例如

- [資料層級的平行化處理](https://blog.dask.org/2019/04/09/numba-stencil)，也包含 stencil 範例。
- [減少記憶體使用量](https://medium.com/capital-one-tech/dask-numba-for-efficient-in-memory-model-scoring-dfc9b68ba6ce)。

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

## 結語

長達一萬字的教學結束了，Markdown 總字數超過三萬，應該來個一鍵三連吧。

目標讀者其實就是在說通訊系，也就是當年的自己。

開頭的最快、最正確和最完整，其實是自己看網路文章一直以來的感受到的問題，完整的太詳細（跟讀文檔沒兩樣），快且正確的文章又不完整，好像永遠沒辦法兼顧。於是本文和我寫的其他教學文章一樣，主要照顧初學者，讓初學者可以快速上手，讀起來又完整，而且內容還正確，當讀者不需要使用平行化時可以在十分鐘之內搞定 Numba，需要平行化或 vectorize 等高級使用技巧時也對網路上許多錯誤做出勘誤和實測結果。

>內容基於 Numba 文檔，作者：Anaconda, Inc.，授權：BSD 2-Clause。
>
>- GitHub: https://github.com/numba/numba
>- 文檔: https://numba.readthedocs.io/
