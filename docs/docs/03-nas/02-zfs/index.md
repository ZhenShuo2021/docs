---
title: "[轉載] Z 檔案系統 (ZFS)"
description: "[轉載] Z 檔案系統 (ZFS)"
tags:
  - NAS
  - Linux
  - ZFS
keywords:
  - Linux
  - ZFS
last_update:
  date: 2024-09-25 GMT+8
  author: zsl0621
---


轉載 FreeBSD Handbook 的 ZFS 文檔，如有任何問題請告知。

這是我找到唯一一個 ZFS 中文文檔，但是翻譯的非常好，放在這裡用於備份使用，是爬蟲工具抓下來的，閱讀文檔請到[原始頁面](https://docs.freebsd.org/zh-tw/books/handbook/zfs/)，可以先看 [什麼使 ZFS 與眾不同](https://docs.freebsd.org/zh-tw/books/handbook/zfs/#zfs-differences) 和 [ZFS 特色與術語](https://docs.freebsd.org/zh-tw/books/handbook/zfs/#zfs-term)，這樣你可以對 ZFS 有基礎了解，至少知道自己在操作什麼，而不是像網路上的教學總是劈頭給你一堆指令...無奈。

簡單介紹 ZFS 的話，是常用於企業級儲存的檔案系統，**內建軟體 Raid** 的文件系統，使用 **scrub** 功能以 checksum 檢驗所有文件完整性，**支援各式快取**方式例如 L2ARC 和 ZIL，**使用大量記憶體**完成以上 feature，用量為 1T 硬碟 1G 記憶體。其他 minor feature 包括：

- 委託 (Delegate)  
- 備份 (Replicate)  
- 快照 (Snapshot)  
- 監禁 (Jail)  
- 資料壓縮 (Compression)  
- 資料去重 (Deduplication)  
- 檔案系統層級的加密 (Encryption)  
- 快照回滾 (Rollback)  
- 自動修復 (Self-healing)  

如果還是拿不定主意選擇哪種檔案系統，這是叫 ChatGPT 整理的表格：

以下是 ZFS 與 Btrfs 和 EXT4 的比較，根據重要性排列，並以表格方式呈現：

| **比較項目**          | **ZFS**                                                              | **Btrfs**                                                             | **EXT4**                                                               | **差異說明**                                                                                                                                                   |
|-----------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **數據完整性**         | 強大的檢查碼機制 (checksum)，自動修復損壞數據。             | 提供基礎的數據校驗，但在大規模部署中穩定性略遜。 | 不提供內建數據校驗或修復機制。                | ZFS 的數據完整性保護最強，Btrfs 次之，EXT4 完全依賴外部工具保護數據完整性。                                                                                         |
| **快照與備份**         | 支持文件系統級別的快照，並可克隆快照。                     | 支持更靈活的文件級快照，適合個別文件的快速複製。            | 不支持內建快照功能。                                       | ZFS 和 Btrfs 都支持快照，但 EXT4 沒有內建功能，須依賴外部工具進行數據備份和恢復。                                                                                  |
| **儲存擴展與管理**     | 支持動態磁碟池擴展，無需停機，可跨多存儲設備。               | 提供較靈活的管理功能，但在磁碟池擴展上不如 ZFS。            | 不支持原生動態擴展功能。                                  | ZFS 具有更強的動態擴展與管理功能，Btrfs 靈活度高，EXT4 則需要手動進行擴展操作。                                                                                     |
| **去重 (Deduplication)** | 原生支持數據去重，但需要大量內存。                         | 尚未有成熟的內建去重功能。                                | 不支持數據去重。                                           | ZFS 的去重功能先進，但資源需求高，Btrfs 和 EXT4 尚無此功能或需依賴外部工具。                                                                                      |
| **RAID 支持**           | 支持 RAID-Z，提供更靈活和高效的 RAID 配置。                 | 支持 RAID 0、1、10，但 RAID 5/6 不穩定。                   | 不支持內建 RAID。                                          | ZFS 在 RAID 支持和數據冗餘上表現最佳，Btrfs 有 RAID 支持，但某些 RAID 配置不穩定，EXT4 需依賴外部軟體如 mdadm 來支持 RAID。                                             |
| **性能與資源使用**     | 需要大量內存和 CPU 資源，特別是在啟用壓縮或去重時。          | 資源需求適中，相對 ZFS 來說性能較靈活。                     | 資源使用最少，性能穩定且成熟。                              | EXT4 在性能和資源需求上表現最優，ZFS 次之，Btrfs 在大型系統中能有良好表現，但在資源要求上不及 EXT4。                                                                |
| **穩定性與成熟度**     | 經過多年企業級驗證，非常穩定。                              | 仍在持續開發，某些功能尚不穩定。                            | 最成熟的檔案系統，廣泛用於多數 Linux 伺服器和桌面環境。       | EXT4 是最穩定和廣泛使用的檔案系統，Btrfs 的功能尚未完全穩定，ZFS 則更適合高需求的企業環境。                                                                           |
| **加密支持**           | 原生支持強大的數據加密。                                    | 需依賴 Linux 內建加密功能。                               | 不支持內建加密。                                           | ZFS 的加密功能最強，Btrfs 可依賴 Linux 的加密模組，EXT4 需額外加密工具來進行數據保護。                                                                               |
| **兼容性與授權**       | 採用 CDDL 授權，無法與 Linux kernel 整合。                   | 採用 GPL 授權，與 Linux kernel 完全整合。                   | 完全整合於 Linux kernel。                                 | ZFS 在 Linux 系統上需要額外安裝，Btrfs 和 EXT4 則完全整合於 Linux kernel，Btrfs 擁有更現代的架構。                                                                 |

綜合來看，ZFS 在數據完整性、快照和 RAID 支持上更強大，適合企業應用；Btrfs 則靈活且與 Linux 深度整合，而 EXT4 是性能和穩定性最佳的選擇，適合一般用途。

<details>
  <summary>ChatGPT 給的參考資料</summary>

https://www.libe.net/en/btrfs-vs-zfs

https://blog.osnexus.com/2013/04/24/btrfs-zfs-the-good-the-bad-and-some-differences/

https://www.ituonline.com/blogs/btrfs-vs-zfs/

https://www.managedserver.eu/zfs-vs-btrfs-a-practical-comparison-and-a-guide-to-choosing-in-different-contexts/

https://www.wundertech.net/btrfs-vs-zfs-comparison/

</details>

# 章 19. Z 檔案系統 (ZFS)

This translation may be out of date. To help with the translations please access the [FreeBSD translations instance](https://translate-dev.freebsd.org/).

### 目錄

*   [19.1. 什麼使 ZFS 與眾不同](#zfs-differences)
*   [19.2. 快速入門指南](#zfs-quickstart)
*   [19.3. `zpool` 管理](#zfs-zpool)
*   [19.4. `zfs` 管理](#zfs-zfs)
*   [19.5. 委託管理](#zfs-zfs-allow)
*   [19.6. 進階主題](#zfs-advanced)
*   [19.7. 其他資源](#zfs-links)
*   [19.8. ZFS 特色與術語](#zfs-term)

_Z 檔案系統_ 或 ZFS 是設計來克服許多在以往設計中發現的主要問題的一個先進的檔案系統。

最初由 Sun™ 所開發，後來的開放源始碼 ZFS 開發已移到 [OpenZFS 計劃](http://open-zfs.org)。

ZFS 的設計目標主要有三個：

*   資料完整性：所有資料都會有一個資料的校驗碼 ([checksum](#zfs-term-checksum))，資料寫入時會計算校驗碼然後一併寫入，往後讀取資料時會再計算一次校驗碼，若校驗碼與當初寫入時不相符，便可偵測到資料錯誤，此時若有可用的資料備援 (Data redundancy)，ZFS 會嘗試自動修正錯誤。
    
*   儲存池：實體的儲存裝置都會先被加入到一個儲存池 (Pool)，這個共用的儲存池可用來配置儲存空間，儲存池的空間可被所有的檔案系統使用且透過加入新的儲存裝置來增加空間。
    
*   效能：提供多個快取機制來增加效能。先進、以記憶體為基礎的讀取快取可使用 [ARC](#zfs-term-arc)。第二層以磁碟為基礎的讀取快取可使用 [L2ARC](#zfs-term-l2arc)，以磁碟為基礎的同步寫入快取則可使用 [ZIL](#zfs-term-zil)。
    

完整的功能清單與術語在 [ZFS 特色與術語](#zfs-term) 中有詳述。

## 19.1. 什麼使 ZFS 與眾不同[](#zfs-differences)

ZFS 與以往任何的檔案系統有顯著的不同，因為它不只是一個檔案系統，ZFS 的獨特優點來自結合了以往被分開的磁碟區管理程式 (Volume Manager) 及檔案系統兩個角色，讓檔案系統也能夠察覺磁碟底層結構的變動。傳統在一個磁碟上只能建立一個檔案系統，若有兩個磁碟則會需要建立兩個分開的檔案系統，在傳統要解決這個問題要使用硬體 RAID 來製作一個空間實際上由數顆實體磁碟所組成的單一的邏輯磁碟給作業系統，作業系統便可在這個邏輯磁碟上放置檔案系統，即使是在那些使用 GEOM 提供的軟體 RAID 解決方案也是一樣，把 UFS 檔案系統放在 RAID Transform 上面當做是一個單一的裝置。ZFS 結合了磁碟區管理程式 (Volume Manager) 與檔案系統來解決這個問題並讓建立多個檔案系統可以共用一個儲存池 (Pool)。ZFS 最大的優點是可以察覺實體磁碟配置的變動，當有額外的磁碟加入到儲存池時可以自動擴增現有的檔案系統，所有的檔案系統便可使用這個新的空間。ZFS 也有數個不同的屬性可以套用到各別檔案系統上，比起單一檔案系統，對建立數個不同檔案系統與資料集 (Dataset) 時有許多的好處。

## 19.2. 快速入門指南[](#zfs-quickstart)

這裡有一個啟動機制，可讓 FreeBSD 在系統初始化時掛載 ZFS 儲存池。要開啟這個功能，可加入此行到 /etc/rc.conf：

zfs\_enable="YES"

然後啟動服務：

```
# service zfs start
```

本節的例子會假設有三個 SCSI 磁碟，名稱分別為 da0, da1 及 da2。SATA 硬體的使用者裝置名稱改為 ada 。

### 19.2.1. 單磁碟儲存池[](#zfs-quickstart-single-disk-pool)

要使用一個磁碟裝置建立一個簡單、無備援的儲存池可：

```
# zpool create example /dev/da0
```

要檢視這個新的儲存池，可查看 `df` 的輸出結果：

```
# df
Filesystem  1K-blocks    Used    Avail Capacity  Mounted on
/dev/ad0s1a   2026030  235230  1628718    13%    /
devfs               1       1        0   100%    /dev
/dev/ad0s1d  54098308 1032846 48737598     2%    /usr
example      17547136       0 17547136     0%    /example
```

這個輸出結果說明 `example` 儲存池已建立且被掛載，現在已經可以作為檔案系統存取，可以在上面建立檔案且使用者可以瀏覽：

```
# cd /example
# ls
# touch testfile
# ls -al
total 4
drwxr-xr-x   2 root  wheel    3 Aug 29 23:15 .
drwxr-xr-x  21 root  wheel  512 Aug 29 23:12 ..
-rw-r--r--   1 root  wheel    0 Aug 29 23:15 testfile
```

但是，這個儲存池並未運用到任何 ZFS 功能，若要在這個儲存池上建立一個有開啟壓縮功能的資料集：

```
# zfs create example/compressed
# zfs set compression=gzip example/compressed
```

`example/compressed` 資料集現在是一個 ZFS 壓縮的檔案系統，可以試著複製較大的檔案到 /example/compressed。

壓縮功能也可以使用以下指令關閉：

```
# zfs set compression=off example/compressed
```

要卸載檔案系統，使用 `zfs umount` 然後再使用 `df` 確認：

```
# zfs umount example/compressed
# df
Filesystem  1K-blocks    Used    Avail Capacity  Mounted on
/dev/ad0s1a   2026030  235232  1628716    13%    /
devfs               1       1        0   100%    /dev
/dev/ad0s1d  54098308 1032864 48737580     2%    /usr
example      17547008       0 17547008     0%    /example
```

要重新掛載檔案系統以便再次使用，使用 `zfs mount` 然後以 `df` 檢查：

```
# zfs mount example/compressed
# df
Filesystem         1K-blocks    Used    Avail Capacity  Mounted on
/dev/ad0s1a          2026030  235234  1628714    13%    /
devfs                      1       1        0   100%    /dev
/dev/ad0s1d         54098308 1032864 48737580     2%    /usr
example             17547008       0 17547008     0%    /example
example/compressed  17547008       0 17547008     0%    /example/compressed
```

儲存池與檔案系統也可以從 `mount` 的結果查詢到：

```
# mount
/dev/ad0s1a on / (ufs, local)
devfs on /dev (devfs, local)
/dev/ad0s1d on /usr (ufs, local, soft-updates)
example on /example (zfs, local)
example/compressed on /example/compressed (zfs, local)
```

在建立之後，ZFS 的資料集可如同其他檔案系統一般使用，且有許多額外功能可在每個資料集上設定。例如，建立一個預計存放重要的資料的新檔案系統 `data`，要設定每個資料區塊 (Data block) 要保留兩份備份：

```
# zfs create example/data
# zfs set copies=2 example/data
```

現在，可以使用 `df` 指令來查看資料與空間的使用率：

```
# df
Filesystem         1K-blocks    Used    Avail Capacity  Mounted on
/dev/ad0s1a          2026030  235234  1628714    13%    /
devfs                      1       1        0   100%    /dev
/dev/ad0s1d         54098308 1032864 48737580     2%    /usr
example             17547008       0 17547008     0%    /example
example/compressed  17547008       0 17547008     0%    /example/compressed
example/data        17547008       0 17547008     0%    /example/data
```

注意，從這個可以發現每個在儲存池上的檔案系統都擁有相同的可用空間，這是為什麼要在這些範例使用 `df` 的原因，為了要顯示檔案系統只會用它們所需要使用到的空間，且均取自同一個儲存池。ZFS 淘汰了磁碟區 (Volume) 與分割區 (Partition) 的概念，且允許多個檔案系統共用相同的儲存池。

不需要使用時可摧毀檔案系統後再摧毀儲存池：

```
# zfs destroy example/compressed
# zfs destroy example/data
# zpool destroy example
```

### 19.2.2. RAID-Z[](#zfs-quickstart-raid-z)

磁碟損壞時，要避免資料因磁碟故障造成遺失便是使用 RAID。ZFS 在它的儲存池設計中支援了這項功能。RAID-Z 儲存池需要使用三個或更多的磁碟，但可以提供比鏡像 (Mirror) 儲存池更多的可用空間。

這個例子會建立一個 RAID-Z 儲存池，並指定要加入這個儲存池的磁碟：

```
# zpool create storage raidz da0 da1 da2
```

|     |     |
| --- | --- |
|     | Sun™ 建議用在 RAID-Z 設定的裝置數在三到九個之間。若需要由 10 個或更多磁碟組成單一儲存池的環境，可考慮分成較小的 RAID-Z 群組。若只有兩個可用的磁碟且需要做備援 (Redundancy)，可考慮使用 ZFS 鏡像 (Mirror)。請參考 [zpool(8)](https://man.freebsd.org/cgi/man.cgi?query=zpool&sektion=8&format=html) 取得更多詳細資訊。 |

先前的例子已經建立了 `storage` 儲存池 (zpool)，現在這個例子會在該儲存池中建立一個新的檔案系統，名稱為 `home`：

```
# zfs create storage/home
```

可以設定開啟壓縮及保留目錄及檔案額外備份的功能：

```
# zfs set copies=2 storage/home
# zfs set compression=gzip storage/home
```

要讓這個空間作為使用者的新家目錄位置，需複製使用者資料到這個目錄並建立適合的符號連結 (Symbolic link)：

```
# cp -rp /home/* /storage/home
# rm -rf /home /usr/home
# ln -s /storage/home /home
# ln -s /storage/home /usr/home
```

現在使用者的資料會儲存在新建立的 /storage/home，可以加入新使用者並登入該使用者來測試。

試著建立檔案系統快照 (Snapshot)，稍後可用來還原 (Rollback)：

```
# zfs snapshot storage/home@08-30-08
```

快照只可以使用整個檔案系統製作，無法使用各別目錄或檔案。

`@` 字元用來區隔檔案系統名稱 (File system) 或磁碟區 (Volume) 名稱，若有重要的目錄意外被刪除，檔案系統可以備份然後還原到先前目錄還存在時的快照 (Snapshot)：

```
# zfs rollback storage/home@08-30-08
```

要列出所有可用的快照，可在檔案系統的 .zfs/snapshot 目錄執行 `ls`，舉例來說，要查看先前已做的快照：

```
# ls /storage/home/.zfs/snapshot
```

也可以寫一個 Script 來對使用者資料做例行性的快照，但隨著時間快照可能消耗大量的磁碟空間。先前的快照可以使用指令移除：

```
# zfs destroy storage/home@08-30-08
```

在測試之後，便可讓 /storage/home 成為真正的 /home 使用此指令：

```
# zfs set mountpoint=/home storage/home
```

執行 `df` 興 `mount` 來確認系統現在是否以把檔案系統做為真正的 /home：

```
# mount
/dev/ad0s1a on / (ufs, local)
devfs on /dev (devfs, local)
/dev/ad0s1d on /usr (ufs, local, soft-updates)
storage on /storage (zfs, local)
storage/home on /home (zfs, local)
# df
Filesystem   1K-blocks    Used    Avail Capacity  Mounted on
/dev/ad0s1a    2026030  235240  1628708    13%    /
devfs                1       1        0   100%    /dev
/dev/ad0s1d   54098308 1032826 48737618     2%    /usr
storage       26320512       0 26320512     0%    /storage
storage/home  26320512       0 26320512     0%    /home
```

這個動作完成 RAID-Z 最後的設定，有關已建立的檔案系統每日狀態更新可以做為 [periodic(8)](https://man.freebsd.org/cgi/man.cgi?query=periodic&sektion=8&format=html) 的一部份在每天晚上執行。加入此行到 /etc/periodic.conf：

daily\_status\_zfs\_enable="YES"

### 19.2.3. 復原 RAID-Z[](#zfs-quickstart-recovering-raid-z)

每個軟體 RAID 都有監控其狀態 (`state`) 的方式，而 RAID-Z 裝置的狀態可以使用這個指令來查看：

```
# zpool status -x
```

如果所有儲存池為上線 ([Online](#zfs-term-online)) 且正常，則訊息會顯示：

```
all pools are healthy
```

如果有發生問題，可能磁碟會呈現離線 ([Offline](#zfs-term-offline)) 的狀態，此時儲存池的狀態會是：

```
  pool: storage
 state: DEGRADED
status: One or more devices has been taken offline by the administrator.
	Sufficient replicas exist for the pool to continue functioning in a
	degraded state.
action: Online the device using 'zpool online' or replace the device with
	'zpool replace'.
 scrub: none requested
config:

	NAME        STATE     READ WRITE CKSUM
	storage     DEGRADED     0     0     0
	  raidz1    DEGRADED     0     0     0
	    da0     ONLINE       0     0     0
	    da1     OFFLINE      0     0     0
	    da2     ONLINE       0     0     0

errors: No known data errors
```

這代表著裝置在之前被管理者使用此指令拿下線：

```
# zpool offline storage da1
```

現在系統可以關機然後更換 da1，當系統恢復上線，則可以替換掉儲存池中故障的磁碟：

```
# zpool replace storage da1
```

到這裡，可以再檢查狀態一次，這時不需使用 `-x` 參數來顯示所有的儲存池：

```
# zpool status storage
 pool: storage
 state: ONLINE
 scrub: resilver completed with 0 errors on Sat Aug 30 19:44:11 2008
config:

	NAME        STATE     READ WRITE CKSUM
	storage     ONLINE       0     0     0
	  raidz1    ONLINE       0     0     0
	    da0     ONLINE       0     0     0
	    da1     ONLINE       0     0     0
	    da2     ONLINE       0     0     0

errors: No known data errors
```

在這個例子中，所有的磁碟均已正常運作。

### 19.2.4. 資料檢驗[](#zfs-quickstart-data-verification)

ZFS 使用校驗碼 (Checksum) 來檢驗資料的完整性 (Integrity)，會在建立檔案系統時便自動開啟。

|     |     |
| --- | --- |
|     | 校驗碼 (Checksum) 可以關閉，但並_不_建議！校驗碼只會使用非常少的儲存空間來確保資料的完整性。若關閉校驗碼會使許多 ZFS 功能無法正常運作，且關閉校驗碼對並不會明顯的改善效能。 |

檢驗校驗碼這個動作即所謂的_清潔 (Scrub)_，可以使用以下指令來檢驗 `storage` 儲存池的資料完整性：

```
# zpool scrub storage
```

清潔所需要的時間依儲存的資料量而定，較大的資料量相對會需要花費較長的時間來檢驗。清潔會對 I/O 有非常密集的操作且一次只能進行一個清潔動作。在清潔完成之後，可以使用 `status` 來查看狀態：

```
# zpool status storage
 pool: storage
 state: ONLINE
 scrub: scrub completed with 0 errors on Sat Jan 26 19:57:37 2013
config:

	NAME        STATE     READ WRITE CKSUM
	storage     ONLINE       0     0     0
	  raidz1    ONLINE       0     0     0
	    da0     ONLINE       0     0     0
	    da1     ONLINE       0     0     0
	    da2     ONLINE       0     0     0

errors: No known data errors
```

查詢結果會顯示上次完成清潔的時間來協助追蹤是否要再做清潔。定期清潔可以協助保護資料不會默默損壞且確保儲存池的完整性。

請參考 [zfs(8)](https://man.freebsd.org/cgi/man.cgi?query=zfs&sektion=8&format=html) 及 [zpool(8)](https://man.freebsd.org/cgi/man.cgi?query=zpool&sektion=8&format=html) 來取得其他 ZFS 選項。

## 19.3. `zpool` 管理[](#zfs-zpool)

ZFS 管理分成兩個主要的工具。`zpool` 工具用來控制儲存池的運作並可處理磁碟的新增、移除、更換與管理。[`zfs`](#zfs-zfs) 工具用來建立、摧毀與管理檔案系統 ([File system](#zfs-term-filesystem)) 與磁碟區 ([Volume](#zfs-term-volume)) 的資料集。

### 19.3.1. 建立與摧毀儲存池[](#zfs-zpool-create)

建立 ZFS 儲存池 (_zpool_) 要做幾個涉及長遠規劃的決定，因為建立儲存池之後便無法再更改儲存池的結構。最重要的決定是要使用那一種型態的 vdev 來將實體磁碟設為同一群組。請參考 [vdev 型態](#zfs-term-vdev) 的清單來取得有關可用選項的詳細資訊。大部份的 vdev 型態不允許在建立儲存池之後再加入額外的磁碟，鏡像 (Mirror) 是可以允許加入額外的磁碟到 vdev 的其中一個例外，另一個則是串連 (Stripe)，可以加入額外的磁碟到 vdev 來升級為鏡像。雖然可以加入額外的 vdev 來擴充儲存池，但儲存池的配置在建立之後便無法更改，若要要更改，則必須先備份資料，把儲存池摧毀後再重新建立。

建立一個簡單的鏡像儲存池：

```
# zpool create mypool mirror /dev/ada1 /dev/ada2
# zpool status
  pool: mypool
 state: ONLINE
  scan: none requested
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada1    ONLINE       0     0     0
            ada2    ONLINE       0     0     0

errors: No known data errors
```

可以一次建立數個 vdev，磁碟群組間使用 vdev 型態關鍵字來區隔，在這個例子使用 `mirror`：

```
# zpool create mypool mirror /dev/ada1 /dev/ada2 mirror /dev/ada3 /dev/ada4
  pool: mypool
 state: ONLINE
  scan: none requested
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada1    ONLINE       0     0     0
            ada2    ONLINE       0     0     0
          mirror-1  ONLINE       0     0     0
            ada3    ONLINE       0     0     0
            ada4    ONLINE       0     0     0

errors: No known data errors
```

儲存池也可以不使用整個磁碟而改使用分割區 (Partition) 來建立。把 ZFS 放到不同的分割區可讓同一個磁碟有其他的分割區可做其他用途，尤其是有 Bootcode 與檔案系統要用來開機的分割區，這讓磁碟可以用來開機也同樣可以做為儲存池的一部份。在 FreeBSD 用分割區來替代整個磁碟並不會對效能有影響。使用分割區也讓管理者可以對磁碟容量做 _少算的預備_，使用比完整容量少的容量，未來若要替換的磁碟號稱與原磁碟相同，但實際上卻比較小時，也可符合這個較小的分割區容量，以使用替換的磁碟。

使用分割區建立一個 [RAID-Z2](#zfs-term-vdev-raidz) 儲存池：

```
# zpool create mypool raidz2 /dev/ada0p3 /dev/ada1p3 /dev/ada2p3 /dev/ada3p3 /dev/ada4p3 /dev/ada5p3
# zpool status
  pool: mypool
 state: ONLINE
  scan: none requested
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          raidz2-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0
            ada2p3  ONLINE       0     0     0
            ada3p3  ONLINE       0     0     0
            ada4p3  ONLINE       0     0     0
            ada5p3  ONLINE       0     0     0

errors: No known data errors
```

不需使用的儲存池可以摧毀，來讓磁碟可以再次使用。摧毀一個儲存池要先卸載所有該儲存池的資料集。若資料集在使用中，卸載的操作會失敗且儲存池不會被摧毀。儲存池的摧毀可以使用 `-f` 來強制執行，但這可能造成那些有開啟這些資料集之中檔案的應用程式無法辨識的行為。

### 19.3.2. 加入與移除裝置{#zfs-zpool-attach}

加入磁碟到儲存池 (zpool) 會有兩種情形：使用 `zpool attach` 加入一個磁碟到既有的 vdev，或使用 `zpool add` 加入 vdev 到儲存池。只有部份 [vdev 型態](#zfs-term-vdev) 允許在 vdev 建立之後加入磁碟。

由單一磁碟所建立的儲存池缺乏備援 (Redundancy) 功能，可以偵測到資料的損壞但無法修復，因為資料沒有其他備份可用。備份數 ([Copies](#zfs-term-copies)) 屬性可以讓您從較小的故障中復原，如磁碟壞軌 (Bad sector)，但無法提供與鏡像或 RAID-Z 同樣層級的保護。由單一磁碟所建立的儲存池可以使用 `zpool attach` 來加入額外的磁碟到 vdev，來建立鏡像。`zpool attach` 也可用來加入額外的磁碟到鏡像群組，來增加備援與讀取效率。若使用的磁碟已有分割區，可以複製該磁碟的分割區配置到另一個，使用 `gpart backup` 與 `gpart restore` 可讓這件事變的很簡單。

加入 _ada1p3_ 來升級單一磁碟串連 (stripe) vdev _ada0p3_ 採用鏡像型態 (mirror)：

```
# zpool status
  pool: mypool
 state: ONLINE
  scan: none requested
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          ada0p3    ONLINE       0     0     0

errors: No known data errors
# zpool attach mypool ada0p3 ada1p3
Make sure to wait until resilver is done before rebooting.

If you boot from pool 'mypool', you may need to update
boot code on newly attached disk 'ada1p3'.

Assuming you use GPT partitioning and 'da0' is your new boot disk
you may use the following command:

        gpart bootcode -b /boot/pmbr -p /boot/gptzfsboot -i 1 da0
# gpart bootcode -b /boot/pmbr -p /boot/gptzfsboot -i 1 ada1
bootcode written to ada1
# zpool status
  pool: mypool
 state: ONLINE
status: One or more devices is currently being resilvered.  The pool will
        continue to function, possibly in a degraded state.
action: Wait for the resilver to complete.
  scan: resilver in progress since Fri May 30 08:19:19 2014
        527M scanned out of 781M at 47.9M/s, 0h0m to go
        527M resilvered, 67.53% done
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0  (resilvering)

errors: No known data errors
# zpool status
  pool: mypool
 state: ONLINE
  scan: resilvered 781M in 0h0m with 0 errors on Fri May 30 08:15:58 2014
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0

errors: No known data errors
```

若不想選擇加入磁碟到既有的 vdev ，對 RAID-Z 來說，可選擇另一種方式，便是加入另一個 vdev 到儲存池。額外的 vdev 可以提供更高的效能，分散寫入資料到 vdev 之間，每個 vdev 會負責自己的備援。也可以混合使用不同的 vdev 型態，但並不建議，例如混合使用 `mirror` 與 `RAID-Z`，加入一個無備援的 vdev 到一個含有 mirror 或 RAID-Z vdev 的儲存池會讓資料損壞的風險擴大整個儲存池，由於會分散寫入資料，若在無備援的磁碟上發生故障的結果便是遺失大半寫到儲存池的資料區塊。

在每個 vdev 間的資料是串連的，例如，有兩個 mirror vdev，便跟 RAID 10 一樣在兩個 mirror 間分散寫入資料，且會做空間的分配，因此 vdev 會在同時達到全滿 100% 的用量。若 vdev 間的可用空間量不同則會影響到效能，因為資料量會不成比例的寫入到使用量較少的 vdev。

當連接額外的裝置到一個可以開機的儲存池，要記得更新 Bootcode。

連接第二個 mirror 群組 (ada2p3 及 ada3p3) 到既有的 mirror：

```
# zpool status
  pool: mypool
 state: ONLINE
  scan: resilvered 781M in 0h0m with 0 errors on Fri May 30 08:19:35 2014
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0

errors: No known data errors
# zpool add mypool mirror ada2p3 ada3p3
# gpart bootcode -b /boot/pmbr -p /boot/gptzfsboot -i 1 ada2
bootcode written to ada2
# gpart bootcode -b /boot/pmbr -p /boot/gptzfsboot -i 1 ada3
bootcode written to ada3
# zpool status
  pool: mypool
 state: ONLINE
  scan: scrub repaired 0 in 0h0m with 0 errors on Fri May 30 08:29:51 2014
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0
          mirror-1  ONLINE       0     0     0
            ada2p3  ONLINE       0     0     0
            ada3p3  ONLINE       0     0     0

errors: No known data errors
```

現在已無法從儲存池上移除 vdev，且磁碟只能夠在有足夠備援空間的情況下從 mirror 移除，若在 mirror 群組中只剩下一個磁碟，便會取消 mirror 然後還原為 stripe，若剩下的那個磁碟故障，便會影響到整個儲存池。

從一個三方 mirror 群組移除一個磁碟：

```
# zpool status
  pool: mypool
 state: ONLINE
  scan: scrub repaired 0 in 0h0m with 0 errors on Fri May 30 08:29:51 2014
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0
            ada2p3  ONLINE       0     0     0

errors: No known data errors
# zpool detach mypool ada2p3
# zpool status
  pool: mypool
 state: ONLINE
  scan: scrub repaired 0 in 0h0m with 0 errors on Fri May 30 08:29:51 2014
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0

errors: No known data errors
```

### 19.3.3. 檢查儲存池狀態[](#zfs-zpool-status)

儲存池的狀態很重要，若有磁碟機離線或偵測到讀取、寫入或校驗碼 (Checksum) 錯誤，對應的錯誤計數便會增加。`status` 會顯示儲存池中每一個磁碟機的設定與狀態及整個儲存池的狀態。需要處置的方式與有關最近清潔 ([`Scrub`](#zfs-zpool-scrub)) 的詳細資訊也會一併顯示。

```
# zpool status
  pool: mypool
 state: ONLINE
  scan: scrub repaired 0 in 2h25m with 0 errors on Sat Sep 14 04:25:50 2013
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          raidz2-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0
            ada2p3  ONLINE       0     0     0
            ada3p3  ONLINE       0     0     0
            ada4p3  ONLINE       0     0     0
            ada5p3  ONLINE       0     0     0

errors: No known data errors
```

### 19.3.4. 清除錯誤[](#zfs-zpool-clear)

當偵測到錯誤發生，讀取、寫入或校驗碼 (Checksum) 的計數便會增加。使用 `zpool clear _mypool_` 可以清除錯誤訊息及重置計數。清空錯誤狀態對當儲存池發生錯誤要使用自動化 Script 通知的管理者來說會很重要，因在舊的錯誤尚未清除前不會回報後續的錯誤。

### 19.3.5. 更換運作中的裝置[](#zfs-zpool-replace)

可能有一些情況會需要更換磁碟為另一個磁碟，當要更換運作中的磁碟，此程序會維持舊有的磁碟在更換的過程為上線的狀態，儲存池不會進入降級 ([Degraded](#zfs-term-degraded)) 的狀態，來減少資料遺失的風險。`zpool replace` 會複製所有舊磁碟的資料到新磁碟，操作完成之後舊磁碟便會與 vdev 中斷連線。若新磁碟容量較舊磁碟大，也可以會增加儲存池來使用新的空間，請參考 [擴增儲存池](#zfs-zpool-online)。

更換儲存池中正在運作的狀置：

```
# zpool status
  pool: mypool
 state: ONLINE
  scan: none requested
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0

errors: No known data errors
# zpool replace mypool ada1p3 ada2p3
Make sure to wait until resilver is done before rebooting.

If you boot from pool 'zroot', you may need to update
boot code on newly attached disk 'ada2p3'.

Assuming you use GPT partitioning and 'da0' is your new boot disk
you may use the following command:

        gpart bootcode -b /boot/pmbr -p /boot/gptzfsboot -i 1 da0
# gpart bootcode -b /boot/pmbr -p /boot/gptzfsboot -i 1 ada2
# zpool status
  pool: mypool
 state: ONLINE
status: One or more devices is currently being resilvered.  The pool will
        continue to function, possibly in a degraded state.
action: Wait for the resilver to complete.
  scan: resilver in progress since Mon Jun  2 14:21:35 2014
        604M scanned out of 781M at 46.5M/s, 0h0m to go
        604M resilvered, 77.39% done
config:

        NAME             STATE     READ WRITE CKSUM
        mypool           ONLINE       0     0     0
          mirror-0       ONLINE       0     0     0
            ada0p3       ONLINE       0     0     0
            replacing-1  ONLINE       0     0     0
              ada1p3     ONLINE       0     0     0
              ada2p3     ONLINE       0     0     0  (resilvering)

errors: No known data errors
# zpool status
  pool: mypool
 state: ONLINE
  scan: resilvered 781M in 0h0m with 0 errors on Mon Jun  2 14:21:52 2014
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada2p3  ONLINE       0     0     0

errors: No known data errors
```

### 19.3.6. 處理故障裝置[](#zfs-zpool-resilver)

當儲存池中的磁碟故障，該故障硬碟所屬的 vdev 便會進入降級 ([Degraded](#zfs-term-degraded)) 狀態，所有的資料仍可使用，但效能可能會降低，因為遺失的資料必須從可用的備援資料計算才能取得。要將 vdev 恢復完整運作的狀態必須更換故障的實體裝置。然後 ZFS 便會開始修復 ([Resilver](#zfs-term-resilver)，古代鏡子的修復稱 Resilver) 作業，會從可用的備援資料計算出故障磁碟中的資料並寫入到替換的裝置上。完成後 vdev 便會重新返回上線 ([Online](#zfs-term-online)) 的狀態。

若 vdev 沒有任何備援資料或有多個裝置故障，沒有足夠的備援資料可以補償，儲存池便會進入故障 ([Faulted](#zfs-term-faulted)) 的狀態。

更換故障的磁碟時，故障磁碟的名稱會更換為裝置的 GUID，若替換裝置要使用相同的裝置名稱，則在 `zpool replace` 不須加上新裝置名稱參數。

使用 `zpool replace` 更換故障的磁碟：

```
# zpool status
  pool: mypool
 state: DEGRADED
status: One or more devices could not be opened.  Sufficient replicas exist for
        the pool to continue functioning in a degraded state.
action: Attach the missing device and online it using 'zpool online'.
   see: http://illumos.org/msg/ZFS-8000-2Q
  scan: none requested
config:

        NAME                    STATE     READ WRITE CKSUM
        mypool                  DEGRADED     0     0     0
          mirror-0              DEGRADED     0     0     0
            ada0p3              ONLINE       0     0     0
            316502962686821739  UNAVAIL      0     0     0  was /dev/ada1p3

errors: No known data errors
# zpool replace mypool 316502962686821739 ada2p3
# zpool status
  pool: mypool
 state: DEGRADED
status: One or more devices is currently being resilvered.  The pool will
        continue to function, possibly in a degraded state.
action: Wait for the resilver to complete.
  scan: resilver in progress since Mon Jun  2 14:52:21 2014
        641M scanned out of 781M at 49.3M/s, 0h0m to go
        640M resilvered, 82.04% done
config:

        NAME                        STATE     READ WRITE CKSUM
        mypool                      DEGRADED     0     0     0
          mirror-0                  DEGRADED     0     0     0
            ada0p3                  ONLINE       0     0     0
            replacing-1             UNAVAIL      0     0     0
              15732067398082357289  UNAVAIL      0     0     0  was /dev/ada1p3/old
              ada2p3                ONLINE       0     0     0  (resilvering)

errors: No known data errors
# zpool status
  pool: mypool
 state: ONLINE
  scan: resilvered 781M in 0h0m with 0 errors on Mon Jun  2 14:52:38 2014
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada2p3  ONLINE       0     0     0

errors: No known data errors
```

### 19.3.7. 清潔儲存池[](#zfs-zpool-scrub)

建議儲存池要定期清潔 ([Scrub](#zfs-term-scrub))，最好是每一個月清潔一次。 `scrub` 作業對磁碟操作非常的密集，在執行時會降低磁碟的效能。在排程 `scrub` 時避免在使用高峰的時期，或使用 [`vfs.zfs.scrub_delay`](#zfs-advanced-tuning-scrub_delay) 來調整 `scrub` 的相對優先權來避免影響其他的工作。

```
# zpool scrub mypool
# zpool status
  pool: mypool
 state: ONLINE
  scan: scrub in progress since Wed Feb 19 20:52:54 2014
        116G scanned out of 8.60T at 649M/s, 3h48m to go
        0 repaired, 1.32% done
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          raidz2-0  ONLINE       0     0     0
            ada0p3  ONLINE       0     0     0
            ada1p3  ONLINE       0     0     0
            ada2p3  ONLINE       0     0     0
            ada3p3  ONLINE       0     0     0
            ada4p3  ONLINE       0     0     0
            ada5p3  ONLINE       0     0     0

errors: No known data errors
```

若發生需要取消清潔作業的事，可以下 `zpool scrub -s _mypool_`。

### 19.3.8. 自我修復[](#zfs-zpool-selfheal)

校驗碼 (Checksum) 會隨資料區塊一併儲存，這使得檔案系統可以做到_自我修復_。這個功能可以在校驗碼與儲存池中的另一個裝置不同時自動修復資料。舉例來說，有兩個磁碟做鏡像 (Mirror)，其中一個磁碟機開始失常並無法正常儲存資料，甚至是資料放在長期封存的儲存裝置上，已經很久沒有被存取。傳統的檔案系統需要執行演算法來檢查並修復資料如 [fsck(8)](https://man.freebsd.org/cgi/man.cgi?query=fsck&sektion=8&format=html)，這些指令耗費時間，且在嚴重時需要管理者手動決定要做那一種修復操作。當 ZFS 偵測到資料區塊的校驗碼不對時，它除了把資料交給需要的應用程式外，也會修正在磁碟上錯誤的資料。這件事不需要與系統管理者作任何互動便會在一般的儲存池操作時完成。

接下來的例子會示範自我修復會如何運作。建立一個使用磁碟 /dev/ada0 及 /dev/ada1 做鏡像的儲存池。

```
# zpool create healer mirror /dev/ada0 /dev/ada1
# zpool status healer
  pool: healer
 state: ONLINE
  scan: none requested
config:

    NAME        STATE     READ WRITE CKSUM
    healer      ONLINE       0     0     0
      mirror-0  ONLINE       0     0     0
       ada0     ONLINE       0     0     0
       ada1     ONLINE       0     0     0

errors: No known data errors
# zpool list
NAME     SIZE  ALLOC   FREE   CKPOINT  EXPANDSZ   FRAG   CAP  DEDUP  HEALTH  ALTROOT
healer   960M  92.5K   960M         -         -     0%    0%  1.00x  ONLINE  -
```

將部份需要使用自我修復功能來保護的重要資料複製到該儲存池，建立一個儲存池的校驗碼供稍後做比較時使用。

```
# cp /some/important/data /healer
# zfs list
NAME     SIZE  ALLOC   FREE    CAP  DEDUP  HEALTH  ALTROOT
healer   960M  67.7M   892M     7%  1.00x  ONLINE  -
# sha1 /healer > checksum.txt
# cat checksum.txt
SHA1 (/healer) = 2753eff56d77d9a536ece6694bf0a82740344d1f
```

寫入隨機的資料到鏡像的第一個磁碟來模擬資料損毀的情況。要避免 ZFS 偵測到錯誤時馬上做修復，接著要將儲存池匯出，待模擬資料損毀之後再匯入。

|     |     |
| --- | --- |
|     | 這是一個危險的操作，會破壞重要的資料。在這裡使用僅為了示範用，不應在儲存池正常運作時嘗試使用，也不應將這個故意損壞資料的例子用在任何其他的檔案系統上，所以請勿使用任何不屬於該儲存池的其他磁碟裝置名稱並確定在執行指令前已對儲存池做正確的備份！ |

```
# zpool export healer
# dd if=/dev/random of=/dev/ada1 bs=1m count=200
200+0 records in
200+0 records out
209715200 bytes transferred in 62.992162 secs (3329227 bytes/sec)
# zpool import healer
```

儲存池的狀態顯示有一個裝置發生了錯誤。注意，應用程式從儲存池讀取的資料中並沒有任何的錯誤資料，ZFS 會自 ada0 裝置提供有正確校驗碼的資料。結果裡面 `CKSUM` 欄位含有非零值便是有錯誤校驗碼的裝置。

```
# zpool status healer
    pool: healer
   state: ONLINE
  status: One or more devices has experienced an unrecoverable error.  An
          attempt was made to correct the error.  Applications are unaffected.
  action: Determine if the device needs to be replaced, and clear the errors
          using 'zpool clear' or replace the device with 'zpool replace'.
     see: http://illumos.org/msg/ZFS-8000-4J
    scan: none requested
  config:

      NAME        STATE     READ WRITE CKSUM
      healer      ONLINE       0     0     0
        mirror-0  ONLINE       0     0     0
         ada0     ONLINE       0     0     0
         ada1     ONLINE       0     0     1

errors: No known data errors
```

錯誤已經被偵測到並且由未被影響的 ada0 鏡像磁碟上的備援提供資料。可與原來的校驗碼做比較來看儲存池是否已修復為一致。

```
# sha1 /healer >> checksum.txt
# cat checksum.txt
SHA1 (/healer) = 2753eff56d77d9a536ece6694bf0a82740344d1f
SHA1 (/healer) = 2753eff56d77d9a536ece6694bf0a82740344d1f
```

儲存池在故意竄改資料前與後的兩個校驗碼仍相符顯示了 ZFS 在校驗碼不同時偵測與自動修正錯誤的能力。注意，這只在當儲存池中有足夠的備援時才可做到，由單一裝置組成的儲存池並沒有自我修復的能力。這也是為什麼在 ZFS 中校驗碼如此重要，任何原因都不該關閉。不需要 [fsck(8)](https://man.freebsd.org/cgi/man.cgi?query=fsck&sektion=8&format=html) 或類似的檔案系統一致性檢查程式便能夠偵測與修正問題，且儲存儲存池在發生問題時仍可正常運作。接著需要做清潔作業來覆蓋在 ada1 上的錯誤資料。

```
# zpool scrub healer
# zpool status healer
  pool: healer
 state: ONLINE
status: One or more devices has experienced an unrecoverable error.  An
            attempt was made to correct the error.  Applications are unaffected.
action: Determine if the device needs to be replaced, and clear the errors
            using 'zpool clear' or replace the device with 'zpool replace'.
   see: http://illumos.org/msg/ZFS-8000-4J
  scan: scrub in progress since Mon Dec 10 12:23:30 2012
        10.4M scanned out of 67.0M at 267K/s, 0h3m to go
        9.63M repaired, 15.56% done
config:

    NAME        STATE     READ WRITE CKSUM
    healer      ONLINE       0     0     0
      mirror-0  ONLINE       0     0     0
       ada0     ONLINE       0     0     0
       ada1     ONLINE       0     0   627  (repairing)

errors: No known data errors
```

清潔作業會從 ada0 讀取資料並重新寫入任何在 ada1 上有錯誤校驗碼的資料。這個操作可以由 `zpool status` 的輸出中呈現修復中 `(repairing)` 的項目來辨識。這個作業完成後，儲存池的狀態會更改為：

```
# zpool status healer
  pool: healer
 state: ONLINE
status: One or more devices has experienced an unrecoverable error.  An
        attempt was made to correct the error.  Applications are unaffected.
action: Determine if the device needs to be replaced, and clear the errors
             using 'zpool clear' or replace the device with 'zpool replace'.
   see: http://illumos.org/msg/ZFS-8000-4J
  scan: scrub repaired 66.5M in 0h2m with 0 errors on Mon Dec 10 12:26:25 2012
config:

    NAME        STATE     READ WRITE CKSUM
    healer      ONLINE       0     0     0
      mirror-0  ONLINE       0     0     0
       ada0     ONLINE       0     0     0
       ada1     ONLINE       0     0 2.72K

errors: No known data errors
```

清潔操作完成便同步了 ada0 到 ada1 間的所有資料。執行 `zpool clear` 可以清除 ([Clear](#zfs-zpool-clear)) 儲存池狀態的錯誤訊息。

```
# zpool clear healer
# zpool status healer
  pool: healer
 state: ONLINE
  scan: scrub repaired 66.5M in 0h2m with 0 errors on Mon Dec 10 12:26:25 2012
config:

    NAME        STATE     READ WRITE CKSUM
    healer      ONLINE       0     0     0
      mirror-0  ONLINE       0     0     0
       ada0     ONLINE       0     0     0
       ada1     ONLINE       0     0     0

errors: No known data errors
```

儲存池現在恢復完整運作的狀態且清除所有的錯誤了。

### 19.3.9. 擴增儲存池[](#zfs-zpool-online)

可用的備援儲存池大小會受到每個 vdev 中容量最小的裝置限制。最小的裝置可以替換成較大的裝置，在更換 ([Replace](#zfs-zpool-replace)) 或修復 ([Resilver](#zfs-term-resilver)) 作業後，儲存池可以成長到該新裝置的可用容量。例如，要做一個 1 TB 磁碟機與一個 2 TB 磁碟機的鏡像，可用的空間會是 1 TB，當 1 TB 磁碟機備更換成另一個 2 TB 的磁碟機時，修復程序會複製既有的資料到新的磁碟機，由於現在兩個裝置都有 2 TB 的容量，所以鏡像的可用空間便會成長到 2 TB。

可以在每個裝置用 `zpool online -e` 來觸發擴充的動作，在擴充完所有裝置後，儲存池便可使用額外的空間。

### 19.3.10. 匯入與匯出儲存池[](#zfs-zpool-import)

儲存池在移動到其他系統之前需要做匯出 (_Export_)，會卸載所有的資料集，然後標記每個裝置為已匯出，為了避免被其他磁碟子系統存取，因此仍會鎖定這些裝置。這個動作讓儲存池可以在支援 ZFS 的其他機器、其他作業系統做匯入 (_Import_)，甚至是不同的硬體架構 (有一些注意事項，請參考 [zpool(8)](https://man.freebsd.org/cgi/man.cgi?query=zpool&sektion=8&format=html))。當資料集有被開啟的檔案，可使用 `zpool export -f` 來強制匯出儲存池，使用這個指令需要小心，資料集是被強制卸載的，因此有可能造成在該資料集開啟檔案的應用程式發生無法預期的結果。

匯出未使用的儲存池：

```
# zpool export mypool
```

匯入儲存池會自動掛載資料集，若不想自動掛載，可以使用 `zpool import -N`。`zpool import -o` 可以設定在匯入時暫時使用的屬性。`zpool import altroot=` 允許匯入時指定基礎掛載點 (Base mount point) 來替換檔案系統根目錄。若儲存池先前用在不同的系統且不正常匯出，可能會需要使用 `zpool import -f` 來強制匯入。`zpool import -a` 會匯入所有沒有被其他系統使用的儲存池。

列出所有可以匯入的儲存池：

```
# zpool import
   pool: mypool
     id: 9930174748043525076
  state: ONLINE
 action: The pool can be imported using its name or numeric identifier.
 config:

        mypool      ONLINE
          ada2p3    ONLINE
```

使用替代的根目錄匯入儲存池：

```
# zpool import -o altroot=/mnt mypool
# zfs list
zfs list
NAME                 USED  AVAIL  REFER  MOUNTPOINT
mypool               110K  47.0G    31K  /mnt/mypool
```

### 19.3.11. 升級儲存儲存池[](#zfs-zpool-upgrade)

在升級 FreeBSD 之後或儲存池是由其他使用舊版 ZFS 的系統匯入，儲存池可以手動升級到最新版本的 ZFS 來支援新的功能。在升級前請評估儲存池是否還要在舊的系統做匯入，由於升級是一個單向的程序，舊的儲存池可以升級，但有新功能的儲存池無法降級。

升級一個 v28 的儲存以支援功能旗標 (`Feature Flags`)：

```
# zpool status
  pool: mypool
 state: ONLINE
status: The pool is formatted using a legacy on-disk format.  The pool can
        still be used, but some features are unavailable.
action: Upgrade the pool using 'zpool upgrade'.  Once this is done, the
        pool will no longer be accessible on software that does not support feat
        flags.
  scan: none requested
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
	    ada0    ONLINE       0     0     0
	    ada1    ONLINE       0     0     0

errors: No known data errors
# zpool upgrade
This system supports ZFS pool feature flags.

The following pools are formatted with legacy version numbers and can
be upgraded to use feature flags.  After being upgraded, these pools
will no longer be accessible by software that does not support feature
flags.

VER  POOL
---  ------------
28   mypool

Use 'zpool upgrade -v' for a list of available legacy versions.
Every feature flags pool has all supported features enabled.
# zpool upgrade mypool
This system supports ZFS pool feature flags.

Successfully upgraded 'mypool' from version 28 to feature flags.
Enabled the following features on 'mypool':
  async_destroy
  empty_bpobj
  lz4_compress
  multi_vdev_crash_dump
```

ZFS 的新功能在 `zpool upgrade` 尚未完成之前無法使用。可以用 `zpool upgrade -v` 來查看升級後有那些新功能，也同時會列出已經支援那些功能。

升級儲存池支援新版的功能旗標 (Feature flags)：

```
# zpool status
  pool: mypool
 state: ONLINE
status: Some supported features are not enabled on the pool. The pool can
        still be used, but some features are unavailable.
action: Enable all features using 'zpool upgrade'. Once this is done,
        the pool may no longer be accessible by software that does not support
        the features. See zpool-features(7) for details.
  scan: none requested
config:

        NAME        STATE     READ WRITE CKSUM
        mypool      ONLINE       0     0     0
          mirror-0  ONLINE       0     0     0
	    ada0    ONLINE       0     0     0
	    ada1    ONLINE       0     0     0

errors: No known data errors
# zpool upgrade
This system supports ZFS pool feature flags.

All pools are formatted using feature flags.

Some supported features are not enabled on the following pools. Once a
feature is enabled the pool may become incompatible with software
that does not support the feature. See zpool-features(7) for details.

POOL  FEATURE
---------------
zstore
      multi_vdev_crash_dump
      spacemap_histogram
      enabled_txg
      hole_birth
      extensible_dataset
      bookmarks
      filesystem_limits
# zpool upgrade mypool
This system supports ZFS pool feature flags.

Enabled the following features on 'mypool':
  spacemap_histogram
  enabled_txg
  hole_birth
  extensible_dataset
  bookmarks
  filesystem_limits
```

|     |     |
| --- | --- |
|     | 在使用儲存池來開機的系統上的 Boot code 也必須一併更新來支援新的儲存池版本，可在含有 Boot code 的分割區使用 `gpart bootcode` 來更新。目前有兩種 Boot code 可使用，需視系統開機的方式使用：GPT (最常用的選項) 以及 EFI (較新的系統)。針對傳統使用 GPT 開機的系統，可以使用以下指令：```# gpart bootcode -b /boot/pmbr -p /boot/gptzfsboot -i 1 ada1```針對使用 EFI 開機的系統可以執行以下指令：```# gpart bootcode -p /boot/boot1.efifat -i 1 ada1```套用 Boot code 到所有儲存池中可開機的磁碟。請參考 [gpart(8)](https://man.freebsd.org/cgi/man.cgi?query=gpart&sektion=8&format=html) 以取得更多資訊。 |

### 19.3.12. 顯示已記錄的儲存池歷史日誌[](#zfs-zpool-history)

修改儲存池的指令會被記錄下來，會記錄的動作包含資料集的建立，屬性更改或更換磁碟。這個歷史記錄用來查看儲存池是如何建立、由誰執行、什麼動作及何時。歷史記錄並非儲存在日誌檔 (Log file)，而是儲存在儲存池。查看這個歷史記錄的指令名稱為 `zpool history`：

```
# zpool history
History for 'tank':
2013-02-26.23:02:35 zpool create tank mirror /dev/ada0 /dev/ada1
2013-02-27.18:50:58 zfs set atime=off tank
2013-02-27.18:51:09 zfs set checksum=fletcher4 tank
2013-02-27.18:51:18 zfs create tank/backup
```

輸出結果顯示曾在該儲存池上執行的 `zpool` 與 `zfs` 指令以及時間戳記。只有會修改儲存池或類似的指令會被記錄下來，像是 `zfs list` 這種指令並不會被記錄。當不指定儲存池名稱時，會列出所有儲存池的歷史記錄。

在提供選項 `-i` 或 `-l` 時 `zpool history` 可以顯更多詳細資訊。`-i` 會顯示使用者觸發的事件外，也會顯示內部記錄的 ZFS 事件。

```
# zpool history -i
History for 'tank':
2013-02-26.23:02:35 [internal pool create txg:5] pool spa 28; zfs spa 28; zpl 5;uts  9.1-RELEASE 901000 amd64
2013-02-27.18:50:53 [internal property set txg:50] atime=0 dataset = 21
2013-02-27.18:50:58 zfs set atime=off tank
2013-02-27.18:51:04 [internal property set txg:53] checksum=7 dataset = 21
2013-02-27.18:51:09 zfs set checksum=fletcher4 tank
2013-02-27.18:51:13 [internal create txg:55] dataset = 39
2013-02-27.18:51:18 zfs create tank/backup
```

更多詳細的資訊可加上 `-l` 來取得，歷史記錄會以較長的格式顯示，包含的資訊有執行指令的使用者名稱、主機名稱以及更改的項目。

```
# zpool history -l
History for 'tank':
2013-02-26.23:02:35 zpool create tank mirror /dev/ada0 /dev/ada1 [user 0 (root) on :global]
2013-02-27.18:50:58 zfs set atime=off tank [user 0 (root) on myzfsbox:global]
2013-02-27.18:51:09 zfs set checksum=fletcher4 tank [user 0 (root) on myzfsbox:global]
2013-02-27.18:51:18 zfs create tank/backup [user 0 (root) on myzfsbox:global]
```

輸出結果顯示 `root` 使用者使用 /dev/ada0 及 /dev/ada1 建立鏡像的儲存池。主機名稱 `myzfsbox` 在建立完儲存池後也同樣會顯示。由於儲存池可以從一個系統匯出再匯入到另一個系統，因此主機名稱也很重要，這樣一來可以清楚的辦識在其他系統上執行的每一個指令的主機名稱。

兩個 `zpool history` 選項可以合併使用來取得最完整的儲存池詳細資訊。儲存池歷史記錄在追蹤執行什麼動作或要取得除錯所需的輸出結果提供了非常有用的資訊。

### 19.3.13. 監視效能[](#zfs-zpool-iostat)

內建的監視系統可以即時顯示儲存池的 I/O 統計資訊。它會顯示儲存池剩餘的空間與使用的空間，每秒執行了多少讀取與寫入的操作，有多少 I/O 頻寬被使用。預設會監視所有在系統中的儲存池都並顯示出來，可以提供儲存池名稱來只顯示該儲存池的監視資訊。舉一個簡單的例子：

```
# zpool iostat
               capacity     operations    bandwidth
pool        alloc   free   read  write   read  write
----------  -----  -----  -----  -----  -----  -----
data         288G  1.53T      2     11  11.3K  57.1K
```

要持續監視 I/O 的活動可以在最後的參數指定一個數字，這個數字代表每次更新資訊所間隔的秒數。在每次經過間隔的時間後會列出新一行的統計資訊，按下 Ctrl+C 可以中止監視。或者在指令列的間隔時間之後再指定一個數字，代表總共要顯示的統計資訊筆數。

使用 `-v` 可以顯示更詳細的 I/O 統計資訊。每個在儲存池中的裝置會以一行統計資訊顯示。這可以幫助了解每一個裝置做了多少讀取與寫入的操作，並可協助確認是否有各別裝置拖慢了整個儲存池的速度。以下範例會顯示有兩個裝置的鏡像儲存池：

```
# zpool iostat -v
                            capacity     operations    bandwidth
pool                     alloc   free   read  write   read  write
-----------------------  -----  -----  -----  -----  -----  -----
data                      288G  1.53T      2     12  9.23K  61.5K
  mirror                  288G  1.53T      2     12  9.23K  61.5K
    ada1                     -      -      0      4  5.61K  61.7K
    ada2                     -      -      1      4  5.04K  61.7K
-----------------------  -----  -----  -----  -----  -----  -----
```

### 19.3.14. 分割儲存儲存池[](#zfs-zpool-split)

由一個或多個鏡像 vdev 所組成的儲存池可以切分開成兩個儲存池。除非有另外指定，否則每個鏡像的最後一個成員會被分離來然用來建立一個含有相同資料的新儲存池。在做這個操作的第一次應先使用 `-n`，會顯示預計會做的操作而不會真的執行，這可以協助確認操作是否與使用者所要的相同。

## 19.4. `zfs` 管理[](#zfs-zfs)

`zfs` 工具負責建立、摧毀與管理在一個儲存池中所有的 ZFS 資料集。儲存池使用 [`zpool`](#zfs-zpool) 來管理。

### 19.4.1. 建立與摧毀資料集[](#zfs-zfs-create)

不同於傳統的磁碟與磁碟區管理程式 (Volume manager) ，在 ZFS 中的空間並_不_會預先分配。傳統的檔案系統在分割與分配空間完後，若沒有增加新的磁碟便無法再增加額外的檔案系統。在 ZFS，可以隨時建立新的檔案系統，每個資料集 ([_Dataset_](#zfs-term-dataset)) 都有自己的屬性，包含壓縮 (Compression)、去重複 (Deduplication)、快取 (Caching) 與配額 (Quota) 功能以及其他有用的屬性如唯讀 (Readonly)、區分大小寫 (Case sensitivity)、網路檔案分享 (Network file sharing) 以及掛載點 (Mount point)。資料集可以存在於其他資料集中，且子資料集會繼承其父資料集的屬性。每個資料集都可以作為一個單位來管理、委託 ([Delegate](#zfs-zfs-allow))、備份 ([Replicate](#zfs-zfs-send))、快照 ([Snapshot](#zfs-zfs-snapshot))、監禁 ([Jail](#zfs-zfs-jail)) 與摧毀 (Destroy)，替每種不同類型或集合的檔案建立各別的資料集還有許多的好處。唯一的缺點是在當有非常大數量的資料集時，部份指令例如 `zfs list` 會變的較緩慢，且掛載上百個或其至上千個資料集可能會使 FreeBSD 的開機程序變慢。

建立一個新資料集並開啟 [LZ4 壓縮](#zfs-term-compression-lz4)：

```
# zfs list
NAME                  USED  AVAIL  REFER  MOUNTPOINT
mypool                781M  93.2G   144K  none
mypool/ROOT           777M  93.2G   144K  none
mypool/ROOT/default   777M  93.2G   777M  /
mypool/tmp            176K  93.2G   176K  /tmp
mypool/usr            616K  93.2G   144K  /usr
mypool/usr/home       184K  93.2G   184K  /usr/home
mypool/usr/ports      144K  93.2G   144K  /usr/ports
mypool/usr/src        144K  93.2G   144K  /usr/src
mypool/var           1.20M  93.2G   608K  /var
mypool/var/crash      148K  93.2G   148K  /var/crash
mypool/var/log        178K  93.2G   178K  /var/log
mypool/var/mail       144K  93.2G   144K  /var/mail
mypool/var/tmp        152K  93.2G   152K  /var/tmp
# zfs create -o compress=lz4 mypool/usr/mydataset
# zfs list
NAME                   USED  AVAIL  REFER  MOUNTPOINT
mypool                 781M  93.2G   144K  none
mypool/ROOT            777M  93.2G   144K  none
mypool/ROOT/default    777M  93.2G   777M  /
mypool/tmp             176K  93.2G   176K  /tmp
mypool/usr             704K  93.2G   144K  /usr
mypool/usr/home        184K  93.2G   184K  /usr/home
mypool/usr/mydataset  87.5K  93.2G  87.5K  /usr/mydataset
mypool/usr/ports       144K  93.2G   144K  /usr/ports
mypool/usr/src         144K  93.2G   144K  /usr/src
mypool/var            1.20M  93.2G   610K  /var
mypool/var/crash       148K  93.2G   148K  /var/crash
mypool/var/log         178K  93.2G   178K  /var/log
mypool/var/mail        144K  93.2G   144K  /var/mail
mypool/var/tmp         152K  93.2G   152K  /var/tmp
```

摧毀資料集會比刪除所有在資料集上所殘留的檔案來的快，由於摧毀資料集並不會掃描所有檔案並更新所有相關的 Metadata。

摧毀先前建立的資料集：

```
# zfs list
NAME                   USED  AVAIL  REFER  MOUNTPOINT
mypool                 880M  93.1G   144K  none
mypool/ROOT            777M  93.1G   144K  none
mypool/ROOT/default    777M  93.1G   777M  /
mypool/tmp             176K  93.1G   176K  /tmp
mypool/usr             101M  93.1G   144K  /usr
mypool/usr/home        184K  93.1G   184K  /usr/home
mypool/usr/mydataset   100M  93.1G   100M  /usr/mydataset
mypool/usr/ports       144K  93.1G   144K  /usr/ports
mypool/usr/src         144K  93.1G   144K  /usr/src
mypool/var            1.20M  93.1G   610K  /var
mypool/var/crash       148K  93.1G   148K  /var/crash
mypool/var/log         178K  93.1G   178K  /var/log
mypool/var/mail        144K  93.1G   144K  /var/mail
mypool/var/tmp         152K  93.1G   152K  /var/tmp
# zfs destroy mypool/usr/mydataset
# zfs list
NAME                  USED  AVAIL  REFER  MOUNTPOINT
mypool                781M  93.2G   144K  none
mypool/ROOT           777M  93.2G   144K  none
mypool/ROOT/default   777M  93.2G   777M  /
mypool/tmp            176K  93.2G   176K  /tmp
mypool/usr            616K  93.2G   144K  /usr
mypool/usr/home       184K  93.2G   184K  /usr/home
mypool/usr/ports      144K  93.2G   144K  /usr/ports
mypool/usr/src        144K  93.2G   144K  /usr/src
mypool/var           1.21M  93.2G   612K  /var
mypool/var/crash      148K  93.2G   148K  /var/crash
mypool/var/log        178K  93.2G   178K  /var/log
mypool/var/mail       144K  93.2G   144K  /var/mail
mypool/var/tmp        152K  93.2G   152K  /var/tmp
```

在最近版本的 ZFS，`zfs destroy` 是非同步的，且釋放出的空間會許要花費數分鐘才會出現在儲存池上，可使用 `zpool get freeing _poolname_` 來查看 `freeing` 屬性，這個屬性會指出資料集在背景已經釋放多少資料區塊了。若有子資料集，如快照 ([Snapshot](#zfs-term-snapshot)) 或其他資料集存在的話，則會無法摧毀父資料集。要摧毀一個資料集及其所有子資料集，可使用 `-r` 來做遞迴摧毀資料集及其所有子資料集，可用 `-n -v` 來列出會被這個操作所摧毀的資料集及快照，而不會真的摧毀，因摧毀快照所釋放出的空間也會同時顯示。

### 19.4.2. 建立與摧毀磁碟區[](#zfs-zfs-volume)

磁碟區 (Volume) 是特殊類型的資料集，不會被掛載成一個檔案系統，而是會被當做儲存區塊裝置出現在 /dev/zvol/poolname/dataset 下。這讓磁碟區可供其他檔案系統使用、拿來備份虛擬機器的磁碟或是使用 iSCSI 或 HAST 通訊協定匯出。

磁碟區可以被格式化成任何檔案系統，或不使用檔案系統來儲存原始資料。對一般使用者，磁碟區就像是一般的磁碟，可以放置一般的檔案系統在這些 _zvols_ 上，並提供一般磁碟或檔案系統一般所沒有的功能。例如，使用壓縮屬性在一個 250 MB 的磁碟區可建立一個壓縮的 FAT 檔案系統。

```
# zfs create -V 250m -o compression=on tank/fat32
# zfs list tank
NAME USED AVAIL REFER MOUNTPOINT
tank 258M  670M   31K /tank
# newfs_msdos -F32 /dev/zvol/tank/fat32
# mount -t msdosfs /dev/zvol/tank/fat32 /mnt
# df -h /mnt | grep fat32
Filesystem           Size Used Avail Capacity Mounted on
/dev/zvol/tank/fat32 249M  24k  249M     0%   /mnt
# mount | grep fat32
/dev/zvol/tank/fat32 on /mnt (msdosfs, local)
```

摧毀一個磁碟區與摧毀一個一般的檔案系統資料集差不多。操作上幾乎是即時的，但在背景會需要花費數分鐘來讓釋放空間再次可用。

### 19.4.3. 重新命名資料集[](#zfs-zfs-rename)

資料集的名稱可以使用 `zfs rename` 更改。父資料集也同樣可以使用這個指令來更改名稱。重新命名一個資料集到另一個父資料集也會更改自父資料集繼承的屬性值。重新命名資料集後，會被卸載然後重新掛載到新的位置 (依繼承的新父資料集而定)，可使用 `-u` 來避免重新掛載。

重新命名一個資料集並移動該資料集到另一個父資料集：

```
# zfs list
NAME                   USED  AVAIL  REFER  MOUNTPOINT
mypool                 780M  93.2G   144K  none
mypool/ROOT            777M  93.2G   144K  none
mypool/ROOT/default    777M  93.2G   777M  /
mypool/tmp             176K  93.2G   176K  /tmp
mypool/usr             704K  93.2G   144K  /usr
mypool/usr/home        184K  93.2G   184K  /usr/home
mypool/usr/mydataset  87.5K  93.2G  87.5K  /usr/mydataset
mypool/usr/ports       144K  93.2G   144K  /usr/ports
mypool/usr/src         144K  93.2G   144K  /usr/src
mypool/var            1.21M  93.2G   614K  /var
mypool/var/crash       148K  93.2G   148K  /var/crash
mypool/var/log         178K  93.2G   178K  /var/log
mypool/var/mail        144K  93.2G   144K  /var/mail
mypool/var/tmp         152K  93.2G   152K  /var/tmp
# zfs rename mypool/usr/mydataset mypool/var/newname
# zfs list
NAME                  USED  AVAIL  REFER  MOUNTPOINT
mypool                780M  93.2G   144K  none
mypool/ROOT           777M  93.2G   144K  none
mypool/ROOT/default   777M  93.2G   777M  /
mypool/tmp            176K  93.2G   176K  /tmp
mypool/usr            616K  93.2G   144K  /usr
mypool/usr/home       184K  93.2G   184K  /usr/home
mypool/usr/ports      144K  93.2G   144K  /usr/ports
mypool/usr/src        144K  93.2G   144K  /usr/src
mypool/var           1.29M  93.2G   614K  /var
mypool/var/crash      148K  93.2G   148K  /var/crash
mypool/var/log        178K  93.2G   178K  /var/log
mypool/var/mail       144K  93.2G   144K  /var/mail
mypool/var/newname   87.5K  93.2G  87.5K  /var/newname
mypool/var/tmp        152K  93.2G   152K  /var/tmp
```

快照也可以像這樣重新命名，由於快照的本質使其無法被重新命名到另一個父資料集。要遞迴重新命名快照可指定 `-r`，然後在子資料集中所有同名的快照也會一併被重新命名。

```
# zfs list -t snapshot
NAME                                USED  AVAIL  REFER  MOUNTPOINT
mypool/var/newname@first_snapshot      0      -  87.5K  -
# zfs rename mypool/var/newname@first_snapshot new_snapshot_name
# zfs list -t snapshot
NAME                                   USED  AVAIL  REFER  MOUNTPOINT
mypool/var/newname@new_snapshot_name      0      -  87.5K  -
```

### 19.4.4. 設定資料集屬性[](#zfs-zfs-set)

每個 ZFS 資料集有數個屬性可以用來控制其行為。大部份的屬性會自動繼承自其父資料集，但可以被自己覆蓋。設定資料集上的屬性可使用 `zfs set _property=value dataset_`。大部份屬性有限制可用的值，`zfs get` 會顯示每個可以使用的屬性及其可用的值。大部份可以使用 `zfs inherit` 還原成其繼承的值。

也可設定使用者自訂的屬性。這些屬性也會成為資料集設定的一部份，且可以被用來提供資料集或其內容的額外資訊。要別分自訂屬性與 ZFS 提供的屬性，會使用冒號 (`:`) 建立一個自訂命名空間供自訂屬性使用。

```
# zfs set custom:costcenter=1234 tank
# zfs get custom:costcenter tank
NAME PROPERTY           VALUE SOURCE
tank custom:costcenter  1234  local
```

要移除自訂屬性，可用 `zfs inherit` 加上 `-r`。若父資料集未定義任何自訂屬性，將會將該屬性完全移除 (更改動作仍會記錄於儲存池的歷史記錄)。

```
# zfs inherit -r custom:costcenter tank
# zfs get custom:costcenter tank
NAME    PROPERTY           VALUE              SOURCE
tank    custom:costcenter  -                  -
# zfs get all tank | grep custom:costcenter
#
```

#### 19.4.4.1. 取得與設定共享屬性[](#zfs-zfs-set-share)

Two commonly used and useful dataset properties are the NFS and SMB share options. Setting these define if and how ZFS datasets may be shared on the network. At present, only setting sharing via NFS is supported on FreeBSD. To get the current status of a share, enter:

```
# zfs get sharenfs mypool/usr/home
NAME             PROPERTY  VALUE    SOURCE
mypool/usr/home  sharenfs  on       local
# zfs get sharesmb mypool/usr/home
NAME             PROPERTY  VALUE    SOURCE
mypool/usr/home  sharesmb  off      local
```

To enable sharing of a dataset, enter:

```
#  zfs set sharenfs=on mypool/usr/home
```

It is also possible to set additional options for sharing datasets through NFS, such as `-alldirs`, `-maproot` and `-network`. To set additional options to a dataset shared through NFS, enter:

```
#  zfs set sharenfs="-alldirs,-maproot=root,-network=192.168.1.0/24" mypool/usr/home
```

### 19.4.5. 管理快照 (Snapshot)[](#zfs-zfs-snapshot)

快照 ([Snapshot](#zfs-term-snapshot)) 是 ZFS 最強大的功能之一。快照提供了資料集唯讀、單一時間點 (Point-in-Time) 的複製功能，使用了寫入時複製 (Copy-On-Write, COW) 的技術，可以透過保存在磁碟上的舊版資料快速的建立快照。若沒有快照存在，在資料被覆蓋或刪除時，便回收空間供未來使用。由於只記錄前一個版本與目前資料集的差異，因此快照可節省磁碟空間。快照只允許在整個資料集上使用，無法在各別檔案或目錄。當建立了一個資料集的快照時，便備份了所有內含的資料，這包含了檔案系統屬性、檔案、目錄、權限等等。第一次建立快照時只會使用到更改參照到資料區塊的空間，不會用到其他額外的空間。使用 `-r` 可以對使用同名的資料集及其所有子資料集的建立一個遞迴快照，提供一致且即時 (Moment-in-time) 的完整檔案系統快照功能，這對於那些彼此有相關或相依檔案存放在不同資料集的應用程式非常重要。不使用快照所備份的資料其實是分散不同時間點的。

ZFS 中的快照提供了多種功能，即使是在其他缺乏快照功能的檔案系統上。一個使用快照的典型例子是在安裝軟體或執行系統升級這種有風險的動作時，能有一個快速的方式可以備份檔案系統目前的狀態，若動作失敗，可以使用快照還原 (Roll back) 到與快照建立時相同的系統狀態，若升級成功，便可刪除快照來釋放空間。若沒有快照功能，升級失敗通常會需要使用備份來恢復 (Restore) 系統，而這個動作非常繁瑣、耗時且可能會需要停機一段時間系統無法使用。使用快照可以快速的還原，即使系統正在執行一般的運作，只而要短暫或甚至不需停機。能夠節省大量在有數 TB 的儲存系統上從備份複製所需資料的時間。快照並非要用來取代儲存池的完整備份，但可以用在快速且簡單的保存某個特定時間點的資料集。

#### 19.4.5.1. 建立快照[](#zfs-zfs-snapshot-creation)

快照可以使用 `zfs snapshot _dataset_@_snapshotname_` 來建立。加入 `-r` 可以遞迴對所有同名的子資料集建立快照。

建立一個整個儲存池的遞迴快照：

```
# zfs list -t all
NAME                                   USED  AVAIL  REFER  MOUNTPOINT
mypool                                 780M  93.2G   144K  none
mypool/ROOT                            777M  93.2G   144K  none
mypool/ROOT/default                    777M  93.2G   777M  /
mypool/tmp                             176K  93.2G   176K  /tmp
mypool/usr                             616K  93.2G   144K  /usr
mypool/usr/home                        184K  93.2G   184K  /usr/home
mypool/usr/ports                       144K  93.2G   144K  /usr/ports
mypool/usr/src                         144K  93.2G   144K  /usr/src
mypool/var                            1.29M  93.2G   616K  /var
mypool/var/crash                       148K  93.2G   148K  /var/crash
mypool/var/log                         178K  93.2G   178K  /var/log
mypool/var/mail                        144K  93.2G   144K  /var/mail
mypool/var/newname                    87.5K  93.2G  87.5K  /var/newname
mypool/var/newname@new_snapshot_name      0      -  87.5K  -
mypool/var/tmp                         152K  93.2G   152K  /var/tmp
# zfs snapshot -r mypool@my_recursive_snapshot
# zfs list -t snapshot
NAME                                        USED  AVAIL  REFER  MOUNTPOINT
mypool@my_recursive_snapshot                   0      -   144K  -
mypool/ROOT@my_recursive_snapshot              0      -   144K  -
mypool/ROOT/default@my_recursive_snapshot      0      -   777M  -
mypool/tmp@my_recursive_snapshot               0      -   176K  -
mypool/usr@my_recursive_snapshot               0      -   144K  -
mypool/usr/home@my_recursive_snapshot          0      -   184K  -
mypool/usr/ports@my_recursive_snapshot         0      -   144K  -
mypool/usr/src@my_recursive_snapshot           0      -   144K  -
mypool/var@my_recursive_snapshot               0      -   616K  -
mypool/var/crash@my_recursive_snapshot         0      -   148K  -
mypool/var/log@my_recursive_snapshot           0      -   178K  -
mypool/var/mail@my_recursive_snapshot          0      -   144K  -
mypool/var/newname@new_snapshot_name           0      -  87.5K  -
mypool/var/newname@my_recursive_snapshot       0      -  87.5K  -
mypool/var/tmp@my_recursive_snapshot           0      -   152K  -
```

建立的快照不會顯示在一般的 `zfs list` 操作結果，要列出快照需在 `zfs list` 後加上 `-t snapshot`，使用 `-t all` 可以同時列出檔案系統的內容及快照。

快照並不會直接掛載，因此 `MOUNTPOINT` 欄位的路徑如此顯示。在 `AVAIL` 欄位不會有可用的磁碟空間，因為快照建立之後便無法再寫入。比較快照與其原來建立時的資料集：

```
# zfs list -rt all mypool/usr/home
NAME                                    USED  AVAIL  REFER  MOUNTPOINT
mypool/usr/home                         184K  93.2G   184K  /usr/home
mypool/usr/home@my_recursive_snapshot      0      -   184K  -
```

同時顯示資料集與快照可以了解快照如何使用 [COW](#zfs-term-cow) 技術來運作。快照只會保存有更動 (_差異_) 的資料，並非整個檔案系統的內容，這個意思是說，快照只會在有做更動時使用一小部份的空間，複製一個檔案到該資料集，可以讓空間使用量變的更明顯，然後再做第二個快照：

```
# cp /etc/passwd /var/tmp
# zfs snapshot mypool/var/tmp@after_cp
# zfs list -rt all mypool/var/tmp
NAME                                   USED  AVAIL  REFER  MOUNTPOINT
mypool/var/tmp                         206K  93.2G   118K  /var/tmp
mypool/var/tmp@my_recursive_snapshot    88K      -   152K  -
mypool/var/tmp@after_cp                   0      -   118K  -
```

第二快照只會包含了資料集做了複製動作後的更動，這樣的機制可以節省大量的空間。注意在複製之後快照 _mypool/var/tmp@my\_recursive\_snapshot_ 於 `USED` 欄位中的大小也更改了，這說明了這個更動在前次快照與之後快照間的關係。

#### 19.4.5.2. 比對快照[](#zfs-zfs-snapshot-diff)

ZFS 提供了內建指令可以用來比對兩個快照 (Snapshot) 之間的差異，在使用者想要查看一段時間之間檔案系統所的變更時非常有用。例如 `zfs diff` 可以讓使用者在最後一次快照中找到意外刪除的檔案。對前面一節所做的兩個快照使用這個指令會產生以下結果：

```
# zfs list -rt all mypool/var/tmp
NAME                                   USED  AVAIL  REFER  MOUNTPOINT
mypool/var/tmp                         206K  93.2G   118K  /var/tmp
mypool/var/tmp@my_recursive_snapshot    88K      -   152K  -
mypool/var/tmp@after_cp                   0      -   118K  -
# zfs diff mypool/var/tmp@my_recursive_snapshot
M       /var/tmp/
+       /var/tmp/passwd
```

指令會列出指定快照 (在這個例子中為 `_mypool/var/tmp@my_recursive_snapshot_`) 與目前檔案系統間的更改。第一個欄位是更改的類型：

|     |     |
| --- | --- |
| +   | 加入了該路徑或檔案。 |
| \-  | 刪除了該路徑或檔案。 |
| M   | 修改了該路徑或檔案。 |
| R   | 重新命名了該路徑或檔案。 |

對照這個表格來看輸出的結果，可以明顯的看到 passwd 是在快照 `_mypool/var/tmp@my_recursive_snapshot_` 建立之後才加入的，結果也同樣看的到掛載到 `_/var/tmp_` 的父目錄已經做過修改。

在使用 ZFS 備份功能來傳輸一個資料集到另一個主機備份時比對兩個快照也同樣很有用。

比對兩個快照需要提供兩個資料集的完整資料集名稱與快照名稱：

```
# cp /var/tmp/passwd /var/tmp/passwd.copy
# zfs snapshot mypool/var/tmp@diff_snapshot
# zfs diff mypool/var/tmp@my_recursive_snapshot mypool/var/tmp@diff_snapshot
M       /var/tmp/
+       /var/tmp/passwd
+       /var/tmp/passwd.copy
# zfs diff mypool/var/tmp@my_recursive_snapshot mypool/var/tmp@after_cp
M       /var/tmp/
+       /var/tmp/passwd
```

備份管理者可以比對兩個自傳送主機所接收到的兩個快照並查看實際在資料集中的變更。請參考 [備份](#zfs-zfs-send) 一節來取得更多資訊。

#### 19.4.5.3. 使用快照還原[](#zfs-zfs-snapshot-rollback)

只要至少有一個可用的快照便可以隨時還原。大多數在已不需要目前資料集，想要改用較舊版的資料的情況，例如，本地開發的測試發生錯誤、不良的系統更新破壞了系統的整體功能或需要還原意外刪除檔案或目錄 …​ 等，都是非常常見的情形。幸運的，要還原到某個快照只需要簡單輸入 `zfs rollback _snapshotname_`。會依快照所做的變更數量來決定處理的時間，還原的操作會在一段時間後完成。在這段時間中，資料集會一直保持一致的狀態，類似一個符合 ACID 原則的資料庫在做還原。還原可在資料集處於上線及可存取的情況下完成，不需要停機。還原到快照之後，資料集便回到當初執行快照時相同的狀態，所有沒有在快照中的其他資料便會被丟棄，因此往後若還有可能需要部份資料時，建議在還原到前一個快照之前先對目前的資料集做快照，這樣一來，使用者便可以在快照之間來回快換，而不會遺失重要的資料。

在第一個範例中，因為 `rm` 操作不小心移除了預期外的資料，要還原到快照。

```
# zfs list -rt all mypool/var/tmp
NAME                                   USED  AVAIL  REFER  MOUNTPOINT
mypool/var/tmp                         262K  93.2G   120K  /var/tmp
mypool/var/tmp@my_recursive_snapshot    88K      -   152K  -
mypool/var/tmp@after_cp               53.5K      -   118K  -
mypool/var/tmp@diff_snapshot              0      -   120K  -
# ls /var/tmp
passwd          passwd.copy     vi.recover
# rm /var/tmp/passwd*
# ls /var/tmp
vi.recover
```

在此時，使用者發現到刪除了太多檔案並希望能夠還原。ZFS 提供了簡單的方可以取回檔案，便是使用還原 (Rollback)，但這只在有定期對重要的資料使用快照時可用。要拿回檔案並從最後一次快照重新開始，可執行以下指令：

```
# zfs rollback mypool/var/tmp@diff_snapshot
# ls /var/tmp
passwd          passwd.copy     vi.recover
```

還原操作會將資料集還原為最後一次快照的狀態。這也可以還原到更早之前，有其他在其之後建立的快照。要這麼做時，ZFS 會發出這個警告：

```
# zfs list -rt snapshot mypool/var/tmp
AME                                   USED  AVAIL  REFER  MOUNTPOINT
mypool/var/tmp@my_recursive_snapshot    88K      -   152K  -
mypool/var/tmp@after_cp               53.5K      -   118K  -
mypool/var/tmp@diff_snapshot              0      -   120K  -
# zfs rollback mypool/var/tmp@my_recursive_snapshot
cannot rollback to 'mypool/var/tmp@my_recursive_snapshot': more recent snapshots exist
use '-r' to force deletion of the following snapshots:
mypool/var/tmp@after_cp
mypool/var/tmp@diff_snapshot
```

這個警告是因在該快照與資料集的目前狀態之間有其他快照存在，然而使用者想要還原到該快照。要完成這樣的還原動作，必須刪除在這之間的快照，因為 ZFS 無法追蹤不同資料集狀態間的變更。在使用者未指定 `-r` 來確認這個動作前，ZFS 不會刪除受影響的快照。若確定要這麼做，那麼必須要知道會遺失所有在這之間的快照，然後可執行以下指令：

```
# zfs rollback -r mypool/var/tmp@my_recursive_snapshot
# zfs list -rt snapshot mypool/var/tmp
NAME                                   USED  AVAIL  REFER  MOUNTPOINT
mypool/var/tmp@my_recursive_snapshot     8K      -   152K  -
# ls /var/tmp
vi.recover
```

可從 `zfs list -t snapshot` 的結果來確認 `zfs rollback -r` 會移除的快照。

#### 19.4.5.4. 從快照還原個別檔案[](#zfs-zfs-snapshot-snapdir)

快照會掛載在父資料集下的隱藏目錄：.zfs/snapshots/snapshotname。預設不會顯示這些目錄，即使是用 `ls -a` 指令。雖然該目錄不會顯示，但該目錄實際存在，而且可以像一般的目錄一樣存取。一個名稱為 `snapdir` 的屬性可以控制是否在目錄清單中顯示這些隱藏目錄，設定該屬性為可見 (`visible`) 可以讓這些目錄出現在 `ls` 以及其他處理目錄內容的指令中。

```
# zfs get snapdir mypool/var/tmp
NAME            PROPERTY  VALUE    SOURCE
mypool/var/tmp  snapdir   hidden   default
# ls -a /var/tmp
.               ..              passwd          vi.recover
# zfs set snapdir=visible mypool/var/tmp
# ls -a /var/tmp
.               ..              .zfs            passwd          vi.recover
```

要還原個別檔案到先前的狀態非常簡單，只要從快照中複製檔案到父資料集。在 .zfs/snapshot 目錄結構下有一個與先前所做的快照名稱相同的目錄，可以很容易的找到。在下個範例中，我們會示範從隱藏的 .zfs 目錄還原一個檔案，透過從含有該檔案的最新版快照複製：

```
# rm /var/tmp/passwd
# ls -a /var/tmp
.               ..              .zfs            vi.recover
# ls /var/tmp/.zfs/snapshot
after_cp                my_recursive_snapshot
# ls /var/tmp/.zfs/snapshot/after_cp
passwd          vi.recover
# cp /var/tmp/.zfs/snapshot/after_cp/passwd /var/tmp
```

執行 `ls .zfs/snapshot` 時，雖然 `snapdir` 可能已經設為隱藏，但仍可能可以顯示該目錄中的內容，這取決於管理者是否要顯示這些目錄，可以只顯示特定的資料集，而其他的則不顯示。從這個隱藏的 .zfs/snapshot 複製檔案或目錄非常簡單，除此之外，嘗試其他的動作則會出現以下錯誤：

```
# cp /etc/rc.conf /var/tmp/.zfs/snapshot/after_cp/
cp: /var/tmp/.zfs/snapshot/after_cp/rc.conf: Read-only file system
```

這個錯誤用來提醒使用者快照是唯讀的，在建立之後不能更改。無法複製檔案進去或從該快照目錄中移除，因為這會變更該資料集所代表的狀態。

快照所消耗的空間是依據自快照之後父檔案系統做了多少變更來決定，快照的 `written` 屬性可以用來追蹤有多少空間被快照所使用。

使用 `zfs destroy _dataset_@_snapshot_` 可以摧毀快照並回收空間。加上 `-r` 可以遞迴移除所有在父資料集下使用同名的快照。加入 `-n -v` 來顯示將要移除的快照清單以及估計回收的空間，而不會實際執行摧毀的操作。

### 19.4.6. 管理複本 (Clone)[](#zfs-zfs-clones)

複本 (Clone) 是快照的複製，但更像是一般的資料集，與快照不同的是，複本是非唯讀的 (可寫)，且可掛載，可以有自己的屬性。使用 `zfs clone` 建立複本之後，便無法再摧毀用來建立複本的快照。複本與快照的父/子關係可以使用 `zfs promote` 來對換。提升複本之後 ，快照便會成為複本的子資料集，而不是原來的父資料集，這個動作會改變空間計算的方式，但並不會實際改變空間的使用量。複本可以被掛載到 ZFS 檔案系統階層中的任何一點，並非只能位於原來快照的位置底下。

要示範複本功能會用到這個範例資料集：

```
# zfs list -rt all camino/home/joe
NAME                    USED  AVAIL  REFER  MOUNTPOINT
camino/home/joe         108K   1.3G    87K  /usr/home/joe
camino/home/joe@plans    21K      -  85.5K  -
camino/home/joe@backup    0K      -    87K  -
```

會使用到複本一般是要在可以保留快照以便出錯時可還原的情況下使用指定的資料集做實驗，由於快照並無法做更改，所以會建立一個可以讀/寫的快照複本。當在複本中做完想要執行的動作後，便可以提升複本成資料集，然後移除舊的檔案系統。嚴格來說這並非必要，因為複本與資料集可同時存在，不會有任何問題。

```
# zfs clone camino/home/joe@backup camino/home/joenew
# ls /usr/home/joe*
/usr/home/joe:
backup.txz     plans.txt

/usr/home/joenew:
backup.txz     plans.txt
# df -h /usr/home
Filesystem          Size    Used   Avail Capacity  Mounted on
usr/home/joe        1.3G     31k    1.3G     0%    /usr/home/joe
usr/home/joenew     1.3G     31k    1.3G     0%    /usr/home/joenew
```

建立完的複本便有與建立快照時狀態相同的資料集，現在複本可以獨立於原來的資料集來做更改。剩下唯一與資料集之間的關係便是快照，ZFS 會在屬性 `origin` 記錄這個關係，一旦在快照與複本之間的相依關係因為使用 `zfs promote` 提升而移除時，複本的 `origin` 也會因為成為一個完全獨立的資料集而移除。以下範例會示範這個動作：

```
# zfs get origin camino/home/joenew
NAME                  PROPERTY  VALUE                     SOURCE
camino/home/joenew    origin    camino/home/joe@backup    -
# zfs promote camino/home/joenew
# zfs get origin camino/home/joenew
NAME                  PROPERTY  VALUE   SOURCE
camino/home/joenew    origin    -       -
```

做為部份更改之後，例如複製 loader.conf 到提升後的複本，這個例子中的舊目錄便無須保留，取而代之的是提升後的複本，這個動作可以用兩個連續的指令來完成：在舊資料集上執行 `zfs destroy` 並在與舊資料相似名稱 (也可能用完全不同的名稱) 的複本上執行 `zfs rename`。

```
# cp /boot/defaults/loader.conf /usr/home/joenew
# zfs destroy -f camino/home/joe
# zfs rename camino/home/joenew camino/home/joe
# ls /usr/home/joe
backup.txz     loader.conf     plans.txt
# df -h /usr/home
Filesystem          Size    Used   Avail Capacity  Mounted on
usr/home/joe        1.3G    128k    1.3G     0%    /usr/home/joe
```

快照的複本現在可以如同一般資料集一樣使用，它的內容包含了所有來自原始快照的資料以及後來加入的檔案，例如 loader.conf。複本可以在許多不同的情境下使用提供 ZFS 的使用者有用的功能，例如，Jail 可以透過含有已安裝了各種應用程式集的快照來提供，使用者可以複製這些快照然後加入自己想要嘗試的應用程式，一但更改可以滿足需求，便可提升複本為完整的資料集然後提供給終端使用者，讓終端使用者可以如同實際擁有資料集一般的使用，這個以節省提供這些 Jail 的時間與管理成本。

### 19.4.7. 備份 (Replication)[](#zfs-zfs-send)

將資料保存在單一地點的單一儲存池上會讓資料暴露在盜竊、自然或人為的風險之下，定期備份整個儲存池非常重要，ZFS 提供了內建的序列化 (Serialization) 功能可以將資料以串流傳送到標準輸出。使用這項技術，不僅可以將資料儲存到另一個已連結到本地系統的儲存池，也可以透過網路將資料傳送到另一個系統，這種備份方式以快照為基礎 (請參考章節 [ZFS 快照(Snapshot)](#zfs-zfs-snapshot))。用來備份資料的指令為 `zfs send` 及 `zfs receive`。

以下例子將示範使用兩個儲存池來做 ZFS 備份：

```
# zpool list
NAME    SIZE  ALLOC   FREE   CKPOINT  EXPANDSZ   FRAG   CAP  DEDUP  HEALTH  ALTROOT
backup  960M    77K   896M         -         -     0%    0%  1.00x  ONLINE  -
mypool  984M  43.7M   940M         -         -     0%    4%  1.00x  ONLINE  -
```

名為 _mypool_ 的儲存池為主要的儲存池，資料會定期寫入與讀取的位置。第二個儲存池 _backup_ 用來待命 (Standby)，萬一主要儲存池無法使用時可替換。注意，ZFS 並不會自動做容錯移轉 (Fail-over)，必須要由系統管理者在需要的時候手動完成。快照會用來提供一個與檔系統一致的版本來做備份，_mypool_ 的快照建立之後，便可以複製到 _backup_ 儲存池，只有快照可以做備份，最近一次快照之後所做的變更不會含在內容裡面。

```
# zfs snapshot mypool@backup1
# zfs list -t snapshot
NAME                    USED  AVAIL  REFER  MOUNTPOINT
mypool@backup1             0      -  43.6M  -
```

快照存在以後，便可以使用 `zfs send` 來建立一個代表快照內容的串流，這個串流可以儲存成檔案或由其他儲存池接收。串流會寫入到標準輸出，但是必須要重新導向到一個檔案或轉接到其他地方，否則會錯誤：

```
# zfs send mypool@backup1
Error: Stream can not be written to a terminal.
You must redirect standard output.
```

要使用 `zfs send` 備份一個資料集，可重新導向到一個位於在已掛載到備份儲存池上的檔案。確定該儲存池有足夠的空間容納要傳送的快照，這裡指的是該快照中內含的所有資料，並非只有上次快照到該快照間的變更。

```
# zfs send mypool@backup1 > /backup/backup1
# zpool list
NAME    SIZE  ALLOC   FREE   CKPOINT  EXPANDSZ   FRAG    CAP  DEDUP  HEALTH  ALTROOT
backup  960M  63.7M   896M         -         -     0%     6%  1.00x  ONLINE  -
mypool  984M  43.7M   940M         -         -     0%     4%  1.00x  ONLINE  -
```

`zfs send` 會傳輸在快照 _backup1_ 中所有的資料到儲存池 _backup_。可以使用 [cron(8)](https://man.freebsd.org/cgi/man.cgi?query=cron&sektion=8&format=html) 排程來自動完成建立與傳送快照的動作。

若不想將備份以封存檔案儲存，ZFS 可用實際的檔案系統來接收資料，讓備份的資料可以直接被存取。要取得實際包含在串流中的資料可以用 `zfs receive` 將串流轉換回檔案與目錄。以下例子會以管線符號連接 `zfs send` 及 `zfs receive`，將資料從一個儲存池複製到另一個，傳輸完成後可以直接使用接收儲存池上的資料。一個資料集只可以被複製到另一個空的資料集。

```
# zfs snapshot mypool@replica1
# zfs send -v mypool@replica1 | zfs receive backup/mypool
send from @ to mypool@replica1 estimated size is 50.1M
total estimated size is 50.1M
TIME        SENT   SNAPSHOT

# zpool list
NAME    SIZE  ALLOC   FREE   CKPOINT  EXPANDSZ   FRAG    CAP  DEDUP  HEALTH  ALTROOT
backup  960M  63.7M   896M         -         -     0%     6%  1.00x  ONLINE  -
mypool  984M  43.7M   940M         -         -     0%     4%  1.00x  ONLINE  -
```

#### 19.4.7.1. 漸進式備份[](#zfs-send-incremental)

`zfs send` 也可以比較兩個快照之間的差異，並且只傳送兩者之間的差異，這麼做可以節省磁碟空間及傳輸時間。例如：

```
# zfs snapshot mypool@replica2
# zfs list -t snapshot
NAME                    USED  AVAIL  REFER  MOUNTPOINT
mypool@replica1         5.72M      -  43.6M  -
mypool@replica2             0      -  44.1M  -
# zpool list
NAME    SIZE  ALLOC   FREE   CKPOINT  EXPANDSZ   FRAG   CAP  DEDUP  HEALTH  ALTROOT
backup  960M  61.7M   898M         -         -     0%    6%  1.00x  ONLINE  -
mypool  960M  50.2M   910M         -         -     0%    5%  1.00x  ONLINE  -
```

會建立一個名為 _replica2_ 的第二個快照，這個快照只中只會含有目前與前次快照 _replica1_ 之間檔案系統所做的變更。使用 `zfs send -i` 並指定要用來產生漸進備份串流的快照，串流中只會含有做過更改的資料。這個動作只在接收端已經有初始快照時才可用。

```
# zfs send -v -i mypool@replica1 mypool@replica2 | zfs receive /backup/mypool
send from @replica1 to mypool@replica2 estimated size is 5.02M
total estimated size is 5.02M
TIME        SENT   SNAPSHOT

# zpool list
NAME    SIZE  ALLOC   FREE   CKPOINT  EXPANDSZ   FRAG  CAP  DEDUP  HEALTH  ALTROOT
backup  960M  80.8M   879M         -         -     0%   8%  1.00x  ONLINE  -
mypool  960M  50.2M   910M         -         -     0%   5%  1.00x  ONLINE  -

# zfs list
NAME                         USED  AVAIL  REFER  MOUNTPOINT
backup                      55.4M   240G   152K  /backup
backup/mypool               55.3M   240G  55.2M  /backup/mypool
mypool                      55.6M  11.6G  55.0M  /mypool

# zfs list -t snapshot
NAME                                         USED  AVAIL  REFER  MOUNTPOINT
backup/mypool@replica1                       104K      -  50.2M  -
backup/mypool@replica2                          0      -  55.2M  -
mypool@replica1                             29.9K      -  50.0M  -
mypool@replica2                                 0      -  55.0M  -
```

如此一來，便成功傳輸漸進式的串流，只有做過更改的資料會被備份，不會傳送完整的 _replica1_。由於不會備份完整的儲存池，只傳送差異的部份，所以可以減少傳輸的時間並節省磁碟空間，特別是在網路緩慢或需要考量每位元傳輸成本時非常有用。

從儲存池 _mypool_ 複製所有檔案與資料的新檔案系統 _backup/mypool_ 便可以使用。若指定 `-P`，會一併複製資料集的屬性，這包含壓縮 (Compression) 設定，配額 (Quota) 及掛載點 (Mount point)。若指定 `-R`，會複製所有指定資料集的子資料集，及這些子資料集的所有屬性。可將傳送與接收自動化來定期使用第二個儲存池做備份。

#### 19.4.7.2. 透過 SSH 傳送加密的備份[](#zfs-send-ssh)

透過網路來傳送串流是一個做遠端備份不錯的方式，但是也有一些缺點，透過網路連線傳送的資料沒有加密，這會讓任何人都可以在未告知傳送方的情況下攔截並轉換串流回資料，這是我們所不想見到的情況，特別是在使用網際網路傳送串流到遠端的主機時。SSH 可用來加密要透過網路連線傳送的資料，在 ZFS 只需要將串流重新導向到標準輸出，如此一來便可簡單的轉接到 SSH。若要讓檔案系統內容在傳送或在遠端系統中也維持在加密的狀態可考慮使用 [PEFS](https://wiki.freebsd.org/PEFS)。

有一些設定以及安全性注意事項必須先完成，只有對 `zfs send` 操作必要的步驟才會在此說明，要取得更多有關 SSH 的資訊請參考 [OpenSSH](../security/#openssh)。

必要的環境設定：

*   使用 SSH 金鑰設定傳送端與接收端間無密碼的 SSH 存取
    
*   正常會需要 `root` 的權限來傳送與接收串流，這需要可以 `root` 登入到接收端系統。但是，預設因安全性考慮會關閉以 `root` 登入。ZFS 委託 ([ZFS Delegation](#zfs-zfs-allow)) 系統可以用來允許一個非 `root` 使用者在每個系統上執行各自的發送與接收操作。
    
*   在傳送端系統上：
    
    ```
    # zfs allow -u someuser send,snapshot mypool
    ```
    
*   要掛載儲存池，無權限的使用者必須擁有該目錄且必須允許一般的使用者掛載檔案系統。在接收端系統上：
    
    ```
    # sysctl vfs.usermount=1
    vfs.usermount: 0 -> 1
    # sysrc -f /etc/sysctl.conf vfs.usermount=1
    # zfs create recvpool/backup
    # zfs allow -u someuser create,mount,receive recvpool/backup
    # chown someuser /recvpool/backup
    ```
    

無權限的使用者現在有能力可以接收並掛載資料集，且 _home_ 資料集可以被複製到遠端系統：

```
% zfs snapshot -r mypool/home@monday
% zfs send -R mypool/home@monday | ssh someuser@backuphost zfs recv -dvu recvpool/backup
```

替儲存在儲存池 _mypool_ 上的檔案系統資料集 _home_ 製作一個遞迴快照 _monday_，然後使用 `zfs send -R` 來傳送包含該資料集及其所有子資料集、快照、複製與設定的串流。輸出會被導向到 SSH 連線的遠端主機 _backuphost_ 上等候輸入的 `zfs receive`，在此建議使用完整網域名稱或 IP 位置。接收端的機器會寫入資料到 _recvpool_ 儲存池上的 _backup_ 資料集，在 `zfs recv` 加上 `-d` 可覆寫在接收端使用相同名稱的快照，加上 `-u` 可讓檔案系統在接收端不會被掛載，當使用 `-v`，會顯示更多有關傳輸的詳細資訊，包含已花費的時間及已傳輸的資料量。

### 19.4.8. 資料集、使用者以及群組配額[](#zfs-zfs-quota)

資料集配額 ([Dataset quota](#zfs-term-quota)) 可用來限制特定資料集可以使用的的空間量。參考配額 ([Reference Quota](#zfs-term-refquota)) 的功能也非常相似，差在參考配額只會計算資料集自己使用的空間，不含快照與子資料集。類似的，使用者 ([User](#zfs-term-userquota)) 與群組 ([Group](#zfs-term-groupquota)) 配額可以用來避免使用者或群組用掉儲存池或資料集的所有空間。

要設定 storage/home/bob 的資料集配額為 10 GB：

```
# zfs set quota=10G storage/home/bob
```

要設定 storage/home/bob 的參考配額為 10 GB：

```
# zfs set refquota=10G storage/home/bob
```

要移除 storage/home/bob 的 10 GB 配額：

```
# zfs set quota=none storage/home/bob
```

設定使用者配額的一般格式為 `userquota@_user_=_size_` 使用者的名稱必須使用以下格式：

*   POSIX 相容的名稱，如 _joe_。
    
*   POSIX 數字 ID，如 _789_。
    
*   SID 名稱，如 _joe.bloggs@example.com_。
    
*   SID 數字 ID，如 _S-1-123-456-789_。
    

例如，要設定使用者名為 _joe_ 的使用者配額為 50 GB：

```
# zfs set userquota@joe=50G
```

要移除所有配額：

```
# zfs set userquota@joe=none
```

|     |     |
| --- | --- |
|     | 使用者配額的屬性不會顯示在 `zfs get all`。非 `root` 的使用者只可以看到自己的配額，除非它們有被授予 `userquota` 權限，擁有這個權限的使用者可以檢視與設定任何人的配額。 |

要設定群組配額的一般格式為：`groupquota@_group_=_size_`。

要設定群組 _firstgroup_ 的配額為 50 GB 可使用：

```
# zfs set groupquota@firstgroup=50G
```

要移除群組 _firstgroup_ 的配額，或確保該群組未設定配額可使用：

```
# zfs set groupquota@firstgroup=none
```

如同使用者配額屬性，非 `root` 使用者只可以查看自己所屬群組的配額。而 `root` 或擁有 `groupquota` 權限的使用者，可以檢視並設定所有群組的任何配額。

要顯示在檔案系統或快照上每位使用者所使用的空間量及配額可使用 `zfs userspace`，要取得群組的資訊則可使用 `zfs groupspace`，要取得有關支援的選項資訊或如何只顯示特定選項的資訊請參考 [zfs(1)](https://man.freebsd.org/cgi/man.cgi?query=zfs&sektion=1&format=html)。

有足夠權限的使用者及 `root` 可以使用以下指令列出 storage/home/bob 的配額：

```
# zfs get quota storage/home/bob
```

### 19.4.9. 保留空間[](#zfs-zfs-reservation)

保留空間 ([Reservation](#zfs-term-reservation)) 可以確保資料集最少可用的空間量，其他任何資料集無法使用保留的空間，這個功能在要確保有足夠的可用空間來存放重要的資料集或日誌檔時特別有用。

`reservation` 屬性的一般格式為 `reservation=_size_`，所以要在 storage/home/bob 設定保留 10 GB 的空間可以用：

```
# zfs set reservation=10G storage/home/bob
```

要清除任何保留空間：

```
# zfs set reservation=none storage/home/bob
```

同樣的原則可以應用在 `refreservation` 屬性來設定參考保留空間 ([Reference Reservation](#zfs-term-refreservation))，參考保留空間的一般格式為 `refreservation=_size_`。

這個指令會顯示任何已設定於 storage/home/bob 的 reservation 或 refreservation：

```
# zfs get reservation storage/home/bob
# zfs get refreservation storage/home/bob
```

### 19.4.10. 壓縮 (Compression)[](#zfs-zfs-compression)

ZFS 提供直接的壓縮功能，在資料區塊層級壓縮資料不僅可以節省空間，也可以增加磁碟的效能。若資料壓縮了 25%，但壓縮的資料會使用了與未壓縮版本相同的速率寫入到磁碟，所以實際的寫入速度會是原來的 125%。壓縮功能也可來替代去重複 ([Deduplication](#zfs-zfs-deduplication)) 功能，因為壓縮並不需要使用額外的記憶體。

ZFS 提了多種不同的壓縮演算法，每一種都有不同的優缺點，隨著 ZFS v5000 引進了 LZ4 壓縮技術，可對整個儲存池開啟壓縮，而不像其他演算法需要消耗大量的效能來達成，最大的優點是 LZ4 擁有 _提早放棄_ 的功能，若 LZ4 無法在資料一開始的部份達成至少 12.5% 的壓縮率，便會以不壓縮的方式來寫入資料區塊來避免 CPU 在那些已經壓縮過或無法壓縮的資料上浪費運算能力。要取得更多有關 ZFS 中可用的壓縮演算法詳細資訊，可參考術語章節中的壓縮 ([Compression](#zfs-term-compression)) 項目。

管理者可以使用資料集的屬性來監視壓縮的效果。

```
# zfs get used,compressratio,compression,logicalused mypool/compressed_dataset
NAME        PROPERTY          VALUE     SOURCE
mypool/compressed_dataset  used              449G      -
mypool/compressed_dataset  compressratio     1.11x     -
mypool/compressed_dataset  compression       lz4       local
mypool/compressed_dataset  logicalused       496G      -
```

資料集目前使用了 449 GB 的空間 (在 used 屬性)。在尚未壓縮前，該資料集應該會使用 496 GB 的空間 (於 `logicalused` 屬性)，這個結果顯示目前的壓縮比為 1.11:1。

壓縮功能在與使用者配額 ([User Quota](#zfs-term-userquota)) 一併使用時可能會產生無法預期的副作用。使用者配額會限制一個使用者在一個資料集上可以使用多少空間，但衡量的依據是以 _壓縮後_ 所使用的空間，因此，若一個使用者有 10 GB 的配額，寫入了 10 GB 可壓縮的資料，使用者將還會有空間儲存額外的資料。若使用者在之後更新了一個檔案，例如一個資料庫，可能有更多或較少的可壓縮資料，那麼剩餘可用的空間量也會因此而改變，這可能會造成奇怪的現象便是，一個使用者雖然沒有增加實際的資料量 (於 `logicalused` 屬性)，但因為更改影響了壓縮率，導致使用者達到配額的上限。

壓縮功能在與備份功能一起使用時也可能會有類似的問題，通常會使用配額功能來限制能夠儲存的資料量來確保有足夠的備份空間可用。但是由於配額功能並不會考量壓縮狀況，可能會有比未壓縮版本備份更多的資料量會被寫入到資料集。

### 19.4.11. 去重複 (Deduplication)[](#zfs-zfs-deduplication)

當開啟，去重複 ([Deduplication](#zfs-term-deduplication)) 功能會使用每個資料區塊的校驗碼 (Checksum) 來偵測重複的資料區塊，當新的資料區塊與現有的資料區塊重複，ZFS 便會寫入連接到現有資料的參考來替代寫入重複的資料區塊，這在資料中有大量重複的檔案或資訊時可以節省大量的空間，要注意的是：去重複功能需要使用大量的記憶體且大部份可節省的空間可改開啟壓縮功能來達成，而壓縮功能不需要使用額外的記憶體。

要開啟去重複功能，需在目標儲存池設定 `dedup` 屬性：

```
# zfs set dedup=on pool
```

只有要被寫入到儲存池的新資料才會做去重複的動作，先前已被寫入到儲存池的資料不會因此啟動了這個選項而做去重複。查看已開啟去重複屬性的儲存池會如下：

```
# zpool list
NAME  SIZE ALLOC  FREE   CKPOINT  EXPANDSZ   FRAG   CAP   DEDUP   HEALTH   ALTROOT
pool 2.84G 2.19M 2.83G         -         -     0%    0%   1.00x   ONLINE   -
```

`DEDUP` 欄位會顯示儲存池的實際去重複率，數值為 `1.00x` 代表資料尚未被去重複。在下一個例子會在前面所建立的去重複儲存池中複製三份 Port 樹到不同的目錄中。

```
# for d in dir1 dir2 dir3; do
> mkdir $d && cp -R /usr/ports $d &
> done
```

已經偵測到重複的資料並做去重複：

```
# zpool list
NAME SIZE  ALLOC  FREE   CKPOINT  EXPANDSZ   FRAG  CAP   DEDUP   HEALTH   ALTROOT
pool 2.84G 20.9M 2.82G         -         -     0%   0%   3.00x   ONLINE   -
```

`DEDUP` 欄位顯示有 `3.00x` 的去重複率，這代表已偵測到多份複製的 Port 樹資料並做了去重複的動作，且只會使用第三份資料所佔的空間。去重複能節省空間的潛力可以非常巨大，但會需要消耗大量的記憶體來持續追蹤去重複的資料區塊。

去重複並非總是有效益的，特別是當儲存池中的資料本身並沒有重複時。ZFS 可以透過在現有儲存池上模擬開啟去重複功能來顯示可能節省的空間：

```
# zdb -S pool
Simulated DDT histogram:

bucket              allocated                       referenced
______   ______________________________   ______________________________
refcnt   blocks   LSIZE   PSIZE   DSIZE   blocks   LSIZE   PSIZE   DSIZE
------   ------   -----   -----   -----   ------   -----   -----   -----
     1    2.58M    289G    264G    264G    2.58M    289G    264G    264G
     2     206K   12.6G   10.4G   10.4G     430K   26.4G   21.6G   21.6G
     4    37.6K    692M    276M    276M     170K   3.04G   1.26G   1.26G
     8    2.18K   45.2M   19.4M   19.4M    20.0K    425M    176M    176M
    16      174   2.83M   1.20M   1.20M    3.33K   48.4M   20.4M   20.4M
    32       40   2.17M    222K    222K    1.70K   97.2M   9.91M   9.91M
    64        9     56K   10.5K   10.5K      865   4.96M    948K    948K
   128        2   9.50K      2K      2K      419   2.11M    438K    438K
   256        5   61.5K     12K     12K    1.90K   23.0M   4.47M   4.47M
    1K        2      1K      1K      1K    2.98K   1.49M   1.49M   1.49M
 Total    2.82M    303G    275G    275G    3.20M    319G    287G    287G

dedup = 1.05, compress = 1.11, copies = 1.00, dedup * compress / copies = 1.16
```

在 `zdb -S` 分析完儲存池後會顯示在啟動去重複後可達到的空間減少比例。在本例中，`1.16` 是非常差的空間節省比例，因為這個比例使用壓縮功能便能達成。若在此儲存池上啟動去重複並不能明顯的節省空間使用量，那麼就不值得耗費大量的記憶體來開啟去重複功能。透過公式 _ratio = dedup \* compress / copies_，系統管理者可以規劃儲存空間的配置，來判斷要處理的資料是否有足夠的重複資料區塊來平衡所需的記憶體。若資料是可壓縮的，那麼空間節少的效果可能會非常好，建議先開啟壓縮功能，且壓縮功能也可以大大提高效能。去重複功能只有在可以節省可觀的空間且有足夠的記憶體做 [DDT](#zfs-term-deduplication) 時才開啟。

### 19.4.12. ZFS 與 Jail[](#zfs-zfs-jail)

`zfs jail` 以及相關的 `jailed` 屬性可以用來將一個 ZFS 資料集委託給一個 [Jail](../jails/#jails) 管理。`zfs jail _jailid_` 可以將一個資料集連結到一個指定的 Jail，而 `zfs unjail` 則可解除連結。資料集要可以在 Jail 中控制需設定 `jailed` 屬性，一旦資料集被隔離便無法再掛載到主機，因為有掛載點可能會破壞主機的安全性。

## 19.5. 委託管理[](#zfs-zfs-allow)

一個全面性的權限委託系統可能無權限的使用者執行 ZFS 的管理功能。例如，若每個使用者的家目錄均為一個資料集，便可以給予使用者權限建立與摧毀它們家目錄中的快照。可以給予備份使用者使用備份功能的權限。一個使用量統計的 Script 可以允許其在執行時能存取所有使用者的空間利用率資料。甚至可以將委託權限委託給其他人，每個子指令與大多數屬性都可使用權限委託。

### 19.5.1. 委託資料集建立[](#zfs-zfs-allow-create)

`zfs allow _someuser_ create _mydataset_` 可以給予指定的使用者在指定的父資料集下建立子資料集的權限。這裡需要注意：建立新資料集會牽涉到掛載，因此需要設定 FreeBSD 的 `vfs.usermount` [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html) 為 `1` 來允許非 root 的使用者掛載一個檔案系統。這裡還有另一項限制可以避免濫用：非 `root` 使用者必須擁有掛載點在檔案系統中所在位置的權限才可掛載。

### 19.5.2. 委託權限委託[](#zfs-zfs-allow-allow)

`zfs allow _someuser_ allow _mydataset_` 可以給予指定的使用者有權限指派它們在目標資料集或其子資料集上擁有的任何權限給其他人。若該使用者擁有 `snapshot` 權限及 `allow` 權限，則該使用者可以授權 `snapshot` 權限給其他使用者。

## 19.6. 進階主題[](#zfs-advanced)

### 19.6.1. 調校[](#zfs-advanced-tuning)

這裡有數個可調校的項目可以調整，來讓 ZFS 在面對各種工作都能以最佳狀況運作。

*   `_vfs.zfs.arc_max_` - Maximum size of the [ARC](#zfs-term-arc). The default is all RAM but 1 GB, or 5/8 of all RAM, whichever is more. However, a lower value should be used if the system will be running any other daemons or processes that may require memory. This value can be adjusted at runtime with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html) and can be set in /boot/loader.conf or /etc/sysctl.conf.
    
*   `_vfs.zfs.arc_meta_limit_` - Limit the portion of the [ARC](#zfs-term-arc) that can be used to store metadata. The default is one fourth of `vfs.zfs.arc_max`. Increasing this value will improve performance if the workload involves operations on a large number of files and directories, or frequent metadata operations, at the cost of less file data fitting in the [ARC](#zfs-term-arc). This value can be adjusted at runtime with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html) and can be set in /boot/loader.conf or /etc/sysctl.conf.
    
*   `_vfs.zfs.arc_min_` - Minimum size of the [ARC](#zfs-term-arc). The default is one half of `vfs.zfs.arc_meta_limit`. Adjust this value to prevent other applications from pressuring out the entire [ARC](#zfs-term-arc). This value can be adjusted at runtime with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html) and can be set in /boot/loader.conf or /etc/sysctl.conf.
    
*   `_vfs.zfs.vdev.cache.size_` - A preallocated amount of memory reserved as a cache for each device in the pool. The total amount of memory used will be this value multiplied by the number of devices. This value can only be adjusted at boot time, and is set in /boot/loader.conf.
    
*   `_vfs.zfs.min_auto_ashift_` - Minimum `ashift` (sector size) that will be used automatically at pool creation time. The value is a power of two. The default value of `9` represents `2^9 = 512`, a sector size of 512 bytes. To avoid _write amplification_ and get the best performance, set this value to the largest sector size used by a device in the pool.
    
    Many drives have 4 KB sectors. Using the default `ashift` of `9` with these drives results in write amplification on these devices. Data that could be contained in a single 4 KB write must instead be written in eight 512-byte writes. ZFS tries to read the native sector size from all devices when creating a pool, but many drives with 4 KB sectors report that their sectors are 512 bytes for compatibility. Setting `vfs.zfs.min_auto_ashift` to `12` (`2^12 = 4096`) before creating a pool forces ZFS to use 4 KB blocks for best performance on these drives.
    
    Forcing 4 KB blocks is also useful on pools where disk upgrades are planned. Future disks are likely to use 4 KB sectors, and `ashift` values cannot be changed after a pool is created.
    
    In some specific cases, the smaller 512-byte block size might be preferable. When used with 512-byte disks for databases, or as storage for virtual machines, less data is transferred during small random reads. This can provide better performance, especially when using a smaller ZFS record size.
    
*   `_vfs.zfs.prefetch_disable_` - Disable prefetch. A value of `0` is enabled and `1` is disabled. The default is `0`, unless the system has less than 4 GB of RAM. Prefetch works by reading larger blocks than were requested into the [ARC](#zfs-term-arc) in hopes that the data will be needed soon. If the workload has a large number of random reads, disabling prefetch may actually improve performance by reducing unnecessary reads. This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.vdev.trim_on_init_` - Control whether new devices added to the pool have the `TRIM` command run on them. This ensures the best performance and longevity for SSDs, but takes extra time. If the device has already been secure erased, disabling this setting will make the addition of the new device faster. This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.vdev.max_pending_` - Limit the number of pending I/O requests per device. A higher value will keep the device command queue full and may give higher throughput. A lower value will reduce latency. This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.top_maxinflight_` - Maxmimum number of outstanding I/Os per top-level [vdev](#zfs-term-vdev). Limits the depth of the command queue to prevent high latency. The limit is per top-level vdev, meaning the limit applies to each [mirror](#zfs-term-vdev-mirror), [RAID-Z](#zfs-term-vdev-raidz), or other vdev independently. This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.l2arc_write_max_` - Limit the amount of data written to the [L2ARC](#zfs-term-l2arc) per second. This tunable is designed to extend the longevity of SSDs by limiting the amount of data written to the device. This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.l2arc_write_boost_` - The value of this tunable is added to [`vfs.zfs.l2arc_write_max`](#zfs-advanced-tuning-l2arc_write_max) and increases the write speed to the SSD until the first block is evicted from the [L2ARC](#zfs-term-l2arc). This "Turbo Warmup Phase" is designed to reduce the performance loss from an empty [L2ARC](#zfs-term-l2arc) after a reboot. This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.scrub_delay_` - Number of ticks to delay between each I/O during a [`scrub`](#zfs-term-scrub). To ensure that a `scrub` does not interfere with the normal operation of the pool, if any other I/O is happening the `scrub` will delay between each command. This value controls the limit on the total IOPS (I/Os Per Second) generated by the `scrub`. The granularity of the setting is determined by the value of `kern.hz` which defaults to 1000 ticks per second. This setting may be changed, resulting in a different effective IOPS limit. The default value is `4`, resulting in a limit of: 1000 ticks/sec / 4 = 250 IOPS. Using a value of _20_ would give a limit of: 1000 ticks/sec / 20 = 50 IOPS. The speed of `scrub` is only limited when there has been recent activity on the pool, as determined by [`vfs.zfs.scan_idle`](#zfs-advanced-tuning-scan_idle). This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.resilver_delay_` - Number of milliseconds of delay inserted between each I/O during a [resilver](#zfs-term-resilver). To ensure that a resilver does not interfere with the normal operation of the pool, if any other I/O is happening the resilver will delay between each command. This value controls the limit of total IOPS (I/Os Per Second) generated by the resilver. The granularity of the setting is determined by the value of `kern.hz` which defaults to 1000 ticks per second. This setting may be changed, resulting in a different effective IOPS limit. The default value is 2, resulting in a limit of: 1000 ticks/sec / 2 = 500 IOPS. Returning the pool to an [Online](#zfs-term-online) state may be more important if another device failing could [Fault](#zfs-term-faulted) the pool, causing data loss. A value of 0 will give the resilver operation the same priority as other operations, speeding the healing process. The speed of resilver is only limited when there has been other recent activity on the pool, as determined by [`vfs.zfs.scan_idle`](#zfs-advanced-tuning-scan_idle). This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.scan_idle_` - Number of milliseconds since the last operation before the pool is considered idle. When the pool is idle the rate limiting for [`scrub`](#zfs-term-scrub) and [resilver](#zfs-term-resilver) are disabled. This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    
*   `_vfs.zfs.txg.timeout_` - Maximum number of seconds between [transaction group](#zfs-term-txg)s. The current transaction group will be written to the pool and a fresh transaction group started if this amount of time has elapsed since the previous transaction group. A transaction group my be triggered earlier if enough data is written. The default value is 5 seconds. A larger value may improve read performance by delaying asynchronous writes, but this may cause uneven performance when the transaction group is written. This value can be adjusted at any time with [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html).
    

### 19.6.2. i386 上的 ZFS[](#zfs-advanced-i386)

ZFS 所提供的部份功能需要使用大量記憶體，且可能需要對有限 RAM 的系統調校來取得最佳的效率。

#### 19.6.2.1. 記憶體[](#_記憶體)

最低需求，總系統記憶體應至少有 1 GB，建議的 RAM 量需視儲存池的大小以及使用的 ZFS 功能而定。一般的經驗法則是每 1 TB 的儲存空間需要 1 GB 的 RAM，若有開啟去重複的功能，一般的經驗法則是每 1 TB 的要做去重複的儲存空間需要 5 GB 的 RAM。雖然有部份使用者成功使用較少的 RAM 來運作 ZFS，但系統在負載較重時有可能會因為記憶用耗而導致當機，對於要使用低於建議 RAM 需求量來運作的系統可能會需要更進一步的調校。

#### 19.6.2.2. 核心設定[](#_核心設定)

由於在 i386™ 平台上位址空間的限制，在 i386™ 架構上的 ZFS 使用者必須加入這個選項到自訂核心設定檔，重新編譯核心並重新開啟：

options        KVA\_PAGES=512

這個選項會增加核心位址空間，允許調整 `vm.kvm_size` 超出目前的 1 GB 限制或在 PAE 的 2 GB 限制。要找到這個選項最合適的數值，可以將想要的位址空間換算成 MB 然後除以 4，在本例中，以 2 GB 計算後即為 `512`。

#### 19.6.2.3. 載入程式可調參數[](#_載入程式可調參數)

在所有的 FreeBSD 架構上均可增加 kmem 位址空間，經測試在一個 1 GB 實體記憶體的測試系統上，加入以下選項到 /boot/loader.conf，重新開啟系統，可成功設定：

vm.kmem\_size="330M"
vm.kmem\_size\_max="330M"
vfs.zfs.arc\_max="40M"
vfs.zfs.vdev.cache.size="5M"

要取得更多詳細的 ZFS 相關調校的建議清單，請參考 [https://wiki.freebsd.org/ZFSTuningGuide](https://wiki.freebsd.org/ZFSTuningGuide)。

## 19.7. 其他資源[](#zfs-links)

*   [FreeBSD Wiki - ZFS](https://wiki.freebsd.org/ZFS)
    
*   [FreeBSD Wiki - ZFS Tuning](https://wiki.freebsd.org/ZFSTuningGuide)
    
*   [Illumos Wiki - ZFS](http://wiki.illumos.org/display/illumos/ZFS)
    
*   [Oracle Solaris ZFS Administration Guide](http://docs.oracle.com/cd/E19253-01/819-5461/index.html)
    
*   [Calomel Blog - ZFS Raidz Performance, Capacity and Integrity](https://calomel.org/zfs_raid_speed_capacity.html)
    

## 19.8. ZFS 特色與術語[](#zfs-term)

ZFS 是一個從本質上與眾不同的檔案系統，由於它並非只是一個檔案系統，ZFS 結合了檔案系統及磁碟區管理程式，讓額外的儲存裝置可以即時的加入到系統並可讓既有的檔案系統立即使用這些在儲存池中空間。透過結合傳統區分為二的兩個角色，ZFS 能夠克服以往 RAID 磁碟群組無法擴充的限制。每個在儲存池頂層的裝置稱作 _vdev_，其可以是一個簡單的磁碟或是一個 RAID 如鏡像或 RAID-Z 陣列。ZFS 的檔案系統 (稱作 _資料集 (Dataset)_) 每一個資料集均可存取整個存池所共通的可用空間，隨著使用儲存池來配置空間區塊，儲存池能給每個檔案系統使用的可用空間就會減少，這個方法可以避免擴大分割區會使的可用空間分散分割區之間的常見問題。

儲存池 (Pool)

_儲存池 (Pool)_ 是建構 ZFS 最基礎的單位。一個儲存池可由一個或多個 vdev 所組成，是用來儲存資料的底層裝置。儲存池會被拿來建立一個或多個檔案系統 (資料集 Dataset) 或區塊裝置 (磁碟區 Volume)，這些資料集與磁碟區會共用儲存池的剩餘可用空間。每一個儲存池可由名稱與 GUID 來辨識。可用的功能會依儲存池上的 ZFS 版本而有不同。

vdev 型態 (vdev Types)

儲存池是由一個或多個 vdev 所組成，vdev 可以是一個磁碟或是 RAID Transform 的磁碟群組。當使用多個 vdev，ZFS 可以分散資料到各個 vdev 來增加效能與最大的可用空間。

*   _磁碟 (Disk)_ - 最基本的 vdev 型態便是一個標準的資料區塊裝置，這可以是一整個磁碟 (例如 /dev/ada0 或 /dev/da0) 或一個分割區 (/dev/ada0p3)。在 FreeBSD 上，使用分割區來替代整個磁碟不會影響效能，這可能與 Solaris 說明文件所建議的有所不同。
    
*   _檔案 (File)_ - 除了磁碟外，ZFS 儲存池可以使用一般檔案為基礎，這在測試與實驗時特別有用。在 `zpool create` 時使用檔案的完整路徑作為裝置路徑。所有 vdev 必須至少有 128 MB 的大小。
    
*   _鏡像 (Mirror)_ - 要建立鏡像，需使用 `mirror` 關鍵字，後面接著要做為該鏡像成員裝置的清單。一個鏡像需要由兩個或多個裝置來組成，所有的資料都會被寫入到所有的成員裝置。鏡像 vdev 可以對抗所有成員故障只剩其中一個而不損失任何資料。

:::info

正常單一磁碟的 vdev 可以使用 [`zpool attach`](#zfs-zpool-attach) 隨時升級成為鏡像 vdev。 |

:::

*   _RAID-Z_ - ZFS 實作了 RAID-Z，以標準的 RAID-5 修改而來，可提供奇偶校驗 (Parity) 更佳的分散性並去除了 "RAID-5 write hole" 導致在預期之外的重啟後資料與奇偶校驗資訊不一致的問題。ZFS 支援三個層級的 RAID-Z，可提供不同程度的備援來換取減少不同程度的可用空間，類型的名稱以陣列中奇偶校驗裝置的數量與儲存池可以容許磁碟故障的數量來命名，從 RAID-Z1 到 RAID-Z3 。
    
    在 RAID-Z1 配置 4 個磁碟，每個磁碟 1 TB，可用的儲存空間則為 3 TB，且若其中一個磁碟故障仍可以降級 (Degraded) 的模式運作，若在故障磁碟尚未更換並修復 (Resilver) 之前又有磁碟故障，所有在儲存池中的資料便會遺失。
    
    在 RAID-Z3 配置 8 個 1 TB 的磁碟，磁碟區將會可以提供 5 TB 的可用空間且在 3 個磁碟故障的情況下仍可運作。Sun™ 建議單一個 vdev 不要使用超過 9 個磁碟。若配置需要使用更多磁碟，建議分成兩個 vdev，這樣儲存池的資料便會分散到這兩個 vdev。
    
    使用兩個 RAID-Z2 各由 8 個磁碟組成的 vdev 的配置可以建立一個類似 RAID-60 的陣列。RAID-Z 群組的儲存空量會接近其中最小的磁碟乘上非奇偶校驗磁碟的數量。4 個 1 TB 磁碟在 RAID-Z1 會有接近 3 TB 的實際大小，且一個由 8 個 1 TB 磁碟組成的 RAID-Z3 陣列會有 5 TB 的可用空間。
    
*   _備援 (Spare)_ - ZFS 有特殊的虛擬 vdev 型態可用來持續追蹤可用的熱備援裝置 (Hot spare)。注意，安裝的熱備援裝置並不會自動佈署，熱備援裝置需要手動使用 `zfs replace` 設定替換故障的裝置。
    
*   _日誌 (Log)_ - ZFS 記錄裝置，也被稱作 ZFS 意圖日誌 (ZFS Intent Log, [ZIL](#zfs-term-zil)) 會從正常的儲存池裝置移動意圖日誌到獨立的裝置上，通常是一個 SSD。有了獨立的日誌裝置，可以明顯的增進有大量同步寫入應用程式的效能，特別是資料庫。日誌裝置可以做成鏡像，但不支援 RAID-Z，若使用多個日誌裝置，寫入動作會被負載平衡分散到這些裝置。
    
*   _快取 (Cache)_ - 加入快取 vdev 到儲存池可以增加儲存空間的 [L2ARC](#zfs-term-l2arc) 快取。快取裝置無法做鏡像，因快取裝置只會儲存額外的現有資料的複本，並沒有資料遺失的風險。
    

交易群組 (Transaction Group, TXG)

交易群組是一種將更動的資料區塊包裝成一組的方式，最後再一次寫入到儲存池。交易群組是 ZFS 用來檢驗一致性的基本單位。每個交易群組會被分配一個獨一無二的 64-bit 連續代號。最多一次可以有三個活動中的交易群組，這三個交易群組的每一個都有這三種狀態：

\* _開放 (Open)_ - 新的交易群組建立之後便處於開放的狀態，可以接受新的寫入動作。永遠會有開放狀態的交易群組，即始交易群組可能會因到達上限而拒絕新的寫入動作。一但開放的交易群組到達上限或到達 [`vfs.zfs.txg.timeout`](#zfs-advanced-tuning-txg-timeout)，交易群組便會繼續進入下一個狀態。 \* _靜置中 (Quiescing)_ - 一個短暫的狀態，會等候任何未完成的操作完成，不會阻擋新開放的交易群組建立。一旦所有在群組中的交易完成，交易群組便會進入到最終狀態。 \* _同步中 (Syncing)_ - 所有在交易群組中的資料會被寫任到穩定的儲存空間，這個程序會依序修改其他也需同樣寫入到穩定儲存空間的資料，如 Metadata 與空間對應表。同步的程多會牽涉多個循環，首先是同步所有更改的資料區塊，也是最大的部份，接著是 Metadata，這可能會需要多個循環來完成。由於要配置空間供資料區塊使用會產生新的 Metadata，同步中狀態在到達循環完成而不再需要分配任何額外空間的狀態前無法結束。同步中狀態也是完成 _synctask_ 的地方，Synctask 是指管理操作，如：建立或摧毀快照與資料集，會修改 uberblock，也會在此時完成。同步狀態完成後，其他處於狀態中狀態的交易群組便會進入同步中狀態。 所有管理功能如快照 ([`Snapshot`](#zfs-term-snapshot)) 會作為交易群組的一部份寫入。當 synctask 建立之後，便會加入到目前開放的交易群組中，然後該群組會盡快的進入同步中狀態來減少管理指令的延遲。

Adaptive Replacement Cache (ARC)

ZFS 使用了自適應替換快取 (Adaptive Replacement Cache, ARC)，而不是傳統的最近最少使用 (Least Recently Used, LRU) 快取，LRU 快取在快取中是一個簡單的項目清單，會依每個物件最近使用的時間來排序，新項會加入到清單的最上方，當快取額滿了便會去除清單最下方的項目來空出空間給較常使用的物件。ARC 結合了四種快取清單，最近最常使用 (Most Recently Used, MRU) 及最常使用 (Most Frequently Used, MFU) 物件加上兩個清單各自的幽靈清單 (Ghost list)，這些幽靈清單會追蹤最近被去除的物件來避免又被加回到快取，避免過去只有偶爾被使用的物件加入清單可以增加快取的命中率。同時使用 MRU 及 MFU 的另外一個優點是掃描一個完整檔案系統可以去除在 MRU 或 LRU 快取中的所有資料，有利於這些才剛存取的內容。使用 ZFS 也有 MFU 可只追蹤最常使用的物件並保留最常被存取的資料區塊快取。

L2ARC

L2ARC 是 ZFS 快取系統的第二層，主要的 ARC 會儲存在 RAM 當中，但因為 RAM 可用的空間量通常有限，因此 ZFS 也可以使用 [快取 vdev (Cache vdev)](#zfs-term-vdev-cache)。固態磁碟 (Solid State Disk, SSD) 常被拿來此處作為快取裝置，因為比起傳統旋轉碟片的磁碟，固態磁碟有較快的速度與較低的延遲。L2ARC 是選用的，但使用可以明顯增進那些已使用 SSD 快取的檔案讀取速度，無須從一般磁碟讀取。L2ARC 也同樣可以加速去重複 ([Deduplication](#zfs-term-deduplication))，因為 DDT 並不適合放在 RAM，但適合放在 L2ARC，比起要從磁碟讀取，可以加快不少速度。為了避免 SSD 因寫入次速過多而過早耗損，加入到快取裝置的資料速率會被限制，直到快取用盡 (去除第一個資料區塊來騰出空間) 之前，寫入到 L2ARC 的資料速率會限制在寫入限制 (Write limit) 與加速限制 (Boost limit) 的總合，之後則會限制為寫入限制，可以控制這兩個速度限制的 [sysctl(8)](https://man.freebsd.org/cgi/man.cgi?query=sysctl&sektion=8&format=html) 數值分別為 [`vfs.zfs.l2arc_write_max`](#zfs-advanced-tuning-l2arc_write_max) 控制每秒有多少數位元組可寫入到快取，而 [`vfs.zfs.l2arc_write_boost`](#zfs-advanced-tuning-l2arc_write_boost) 可在 "渦輪預熱階段" (即寫入加速) 時增加寫入限制。

ZIL

ZIL 會使用比主要儲存池還更快的儲存裝置來加速同步寫入動作 (Synchronous transaction)，如 SSD。當應用程式請求做一個同步的寫入時 (保証資料會安全的儲存到磁碟，而不是先快取稍後再寫入)，資料會先寫入到速度較快的 ZIL 儲存空間，之後再一併寫入到一般的磁碟。這可大量的減少延遲並增進效能。ZIL 只會有利於使用像資料庫這類的同步工作，一般非同步的寫入像複製檔案，則完全不會用到 ZIL。

寫入時複製 (Copy-On-Write)

不像傳統的檔案系統，在 ZFS，當資料要被覆寫時，不會直接覆寫舊資料所在的位置，而是將新資料會寫入到另一個資料區塊，只在資料寫入完成後才會更新 Metadata 指向新的位置。因此，在發生寫入中斷 (在寫入檔案的過程中系統當機或電源中斷) 時，原來檔案的完整內容並不會遺失，只會放棄未寫入完成的新資料，這也意謂著 ZFS 在發生預期之外的關機後不需要做 [fsck(8)](https://man.freebsd.org/cgi/man.cgi?query=fsck&sektion=8&format=html)。

資料集 (Dataset)

_資料集 (Dataset)_ 是 ZFS 檔案系統、磁碟區、快照或複本的通用術語。每個資料集都有獨一無二的名稱使用 _poolname/path@snapshot_ 格式。儲存池的根部技術上來說也算一個資料集，子資料集會採用像目錄一樣的層級來命名，例如 _mypool/home_，home 資料集是 _mypool_ 的子資料集並且會繼承其屬性。這可以在往後繼續擴展成 _mypool/home/user_，這個孫資料集會繼承其父及祖父的屬性。在子資料集的屬性可以覆蓋預設繼承自父及祖父的屬性。資料集及其子資料級的管理權限可以委託 ([Delegate](#zfs-zfs-allow)) 給他人。

檔案系統 (File system)

ZFS 資料集最常被當做檔案系統使用。如同大多數其他的檔案系統，ZFS 檔案系統會被掛載在系統目錄層級的某一處且內含各自擁有權限、旗標及 Metadata 的檔案與目錄。

磁碟區 (Volume)

除了一般的檔案系統資料集之外，ZFS 也可以建立磁碟區 (Volume)，磁碟區是資料區塊裝置。磁碟區有許多與資料集相似的功能，包含複製時寫入、快照、複本以及資料校驗。要在 ZFS 的頂層執行其他檔案系統格式時使用磁碟區非常有用，例如 UFS 虛擬化或匯出 iSCSI 延伸磁區 (Extent)。

快照 (Snapshot)

ZFS 的寫入時複製 ([Copy-On-Write](#zfs-term-cow), COW) 設計可以使用任意的名稱做到幾乎即時、一致的快照。在製做資料集的快照或父資料集遞迴快照 (會包含其所有子資料集) 之後，新的資料會寫入到資的資料區塊，但不會回收舊的資料區塊為可用空間，快照中會使用原版本的檔案系統，而快照之後所做的變更則會儲存在目前的檔案系統，因此不會重複使用額外的空間。當新的資料寫入到目前的檔案系統，便會配置新的資料區塊來儲存這些資料。快照表面大小 (Apparent size) 會隨著在目前檔案系統停止使用的資料區塊而成長，但僅限於快照。可以用唯讀的方式掛載這些快照來復原先前版本的檔案，也可以還原 ([Rollback](#zfs-zfs-snapshot)) 目前的檔案系統到指定的快照，來還原任何在快照之後所做的變更。每個在儲存池中的資料區塊都會有一個參考記數器，可以用來持續追蹤有多少快照、複本、資料集或是磁碟區使用這個資料區塊，當刪除檔案與快照參照的計數變會滅少，直到沒有任何東西參考這個資料區塊才會被回收為可用空間。快照也可使用 [hold](#zfs-zfs-snapshot) 來標記，檔標記為 hold 時，任何嘗試要刪除該快照的動作便會回傳 `EBUSY` 的錯誤，每個快照可以標記多個不同唯一名稱的 hold，而 [release](#zfs-zfs-snapshot) 指令則可以移除 hold，這樣才可刪除快照。在磁碟區上快可以製作快照，但只能用來複製或還原，無法獨立掛載。

複本 (Clone)

快照也可以做複本，複本是可寫入版本的快照，讓檔案系統可分支成為新的資料集。如同快照，複本一開始不會消耗任何額外空間，隨著新資料寫入到複本會配置新的資料區塊，複本的表面大小 (Apparent size) 才會成長，當在複本檔案系統或磁碟區的資料區塊被覆寫時，在先前資料區塊的參考計數則會減少。建立複本所使用的快照無法被刪除，因為複本會相依該快照，快照為父，複本為子。複本可以被提升 (_promoted_)、反轉相依關係，來讓複本成為父，之前的父變為子，這個操作不需要額外的空間。由於反轉了父與子使用的空間量，所以可能會影響既有的配額 (Quota) 與保留空間 (Reservation)。

校驗碼 (Checksum)

配置每個資料區塊快的同時也會做資料校驗，資料校驗用的演算法是依資料集屬性而有所不同的，請參考 [`set`](#zfs-zfs-set)。每個資料區塊會在讀取的過成便完成校驗，讓 ZFS 可以偵測到隱藏的損壞，若資料不符合預期的校驗碼，ZFS 會嘗試從任何可用的備援來還原資料，例如鏡像 (Mirror) 或 RAID-Z。要檢驗所有資料的校驗碼可以使用清潔 ([`Scrub`](#zfs-term-scrub))，資料校驗的演算法有：

\* `fletcher2` \* `fletcher4` \* `sha256` `fletcher` 演算法最快，而 `sha256` 雖較消耗效能，但其有強大的密碼雜湊與較低的衝突率。也可關閉資料校驗，但並不建議。

壓縮 (Compression)

每個資料集都有壓縮 (Compression) 屬性，預設是關閉的，這個屬性可以設定使用以下幾個壓縮演算法的其中一個來壓縮寫入到資料集的新資料。壓縮除了減少空間使用量外，常也會增加讀取與寫入的吞吐量，因為會減少讀取與寫入的資料區塊。

\* _LZ4_ - ZFS 儲存池版本 5000 (功能旗標) 後所增加，LZ4 現在是建議的壓縮演算法，在處理可壓縮的資料時 LZ4 壓縮比 LZJB 快將近 50%，在處理不可壓縮的資料時快將近三倍，LZ4 解壓縮也比 LZJB 將近 80%。在現代的 CPU 上，LZ4 經常平均可用 500 MB/s 的速度壓縮，而解壓縮可到達 1.5 GB/s (每個 CPU 核心)。

\* _LZJB_ - 預設的壓縮演算法。由 Jeff Bonwick 所開發 (ZFS 的創始人之一)。LZJB 與 GZIP 相比，可以較低的 CPU 提供較佳的壓縮功能。在未來預設的壓縮演算法將會更換為 LZ4。

\* _GZIP_ - 在 ZFS 可用的熱門串流壓縮演算法。使用 GZIP 主要的優點之一便是可設定壓縮層級。當設定 `compress` 屬性，管理者可以選擇壓縮層級範圍從最低的壓縮層級 `gzip1` 到最高的壓縮層級 `gzip9`。這讓管理者可以控制要使用多少 CPU 來節省磁碟空間。

\* _ZLE_ - 零長度編號是一個特殊的壓縮演算法，它只會壓縮連續的零。這種壓縮演算法只在資料集中含有大量為零的資料區塊時有用。

備份數 (Copies)

當設定大於 1 的數值時，`copies` 屬性會指示 ZFS 備份每個在檔案系統 ([File System](#zfs-term-filesystem)) 或磁碟區 ([Volume](#zfs-term-volume)) 的資料區塊數份。在重要的資料集上設定這個屬性可以做額外的備援以在資料校驗碼不相符時可做復原。在沒有做備援的儲存池上，備份功能提供只是一種資料的備援方式，備份功能可以復原單一壞軌或其他情況的次要損壞，但無法復原儲存池中整個磁碟損壞所損失的資料。

去重複 (Deduplication)

校驗碼讓在寫入時可以偵測重複資料區塊，使用去重複，可以增加既有、完全相同的資料區塊參考數來節省儲存空間。要偵測重複的資料區塊需要在記憶體中儲存去重複資料表 (Deduplication table, DDT)，這個資料表中會有唯一的校驗碼清單、這些資料區塊的所在位置以及參考數。當寫入新資料時，便會計算校驗碼然後比對清單中是否有符合的既有資料區塊已在清單。去重複使用了 SHA256 校驗碼演算法來提供一個安全的加密雜湊，去重複功能是可以調校的，若 `dedup` 設為 `on` 只要符合校驗碼便會認為資料完全相同，若 `dedup` 設為 `verify` 則會一個一個位元檢查兩個資料區塊的資料來確保資料真的完全相同，若資料不同便會註記與雜湊衝突並會分別儲存兩個資料區塊。由於 DDT 須要儲存每個唯一資料區塊的雜湊，所以會消耗大量的記憶體，一般的經驗法則是每 1 TB 的去重複資料需要使用 5-6 GB 的記憶體。由於要有足夠的 RAM 來儲存整個 DDT 在實務上並不實際，導致在每個新資料區塊寫入前需要從磁碟來讀取 DDT 會對效能有很大的影響，去重複功能可以使用 L2ARC 儲存 DDT 以在快速的系統記憶體及較慢的磁碟之間取得一個平衡點。也可以考慮使用壓縮功能來取代此功能，因為壓縮也能節省相近的空間使用量而不需要大量額外的記憶體。

清潔 (Scrub)

ZFS 有 `scrub` 來替代 [fsck(8)](https://man.freebsd.org/cgi/man.cgi?query=fsck&sektion=8&format=html) 來做一致性的檢查。`scrub` 會讀取所有儲存在儲存池中的資料區塊並且根據儲存在 Metadata 中已知良好的校驗碼來檢驗這些資料區塊的校驗碼，定期檢查儲存池中儲存的所有資料可以確保實際使用這些資料前已將所有損壞的資料區塊復原。在不正常的關閉之後並不需要做清潔動作，但建議每三個月至少執行一次。在正常使用讀取時便會檢查每個資料區塊的校驗碼，但清潔動作可以確保那些不常用的資料也會被檢查以避免隱藏的損壞，如此便能增進資料的安全性，特別是對用來保存資料的儲存裝置。`scrub` 可以使用 [`vfs.zfs.scrub_delay`](#zfs-advanced-tuning-scrub_delay) 調整相對優先權來避免清潔動作降低儲存池上其他工作的效率。

資料集配額 (Dataset Quota)

除了配額及空間保留外，ZFS 提供非常快速且準確的資料集、使用者及群組空間的計算功能，這可讓管理者調整空間配置的方式且可為重要的檔案系統保留空間。

ZFS supports different types of quotas: the dataset quota, the [reference quota (refquota)](#zfs-term-refquota), the [user quota](#zfs-term-userquota), and the [group quota](#zfs-term-groupquota).

配額會限制資料集及後裔包含資料集的快照、子資料集及子資料集的快照能使用的空間量。

|     |     |
| --- | --- |
|     | 磁碟區上無法設定配額，因為 `volsize` 屬性已經被用來做內定的配額。 |

參考配額 (Reference Quota)

參考配額可以設定一個硬性限制 (Hard limit) 來限制資料集能使用的空間量，而這個硬性限制只包含了資料集參考的空間，並不含其後裔所使用的空間，如：檔案系統或快照。

使用者配額 (User Quota)

使用者配額在用來限制特定使用者能使用的空間量時非常有用。

群組配額 (Group Quota)

群組配額可以限制特定群組能使用的空間量。

資料集保留空間 (Dataset Reservation)

`reservation` 屬性可以確保對特定資料集及其後裔最小可用的空間量，若在 storage/home/bob 設定 10 GB 的保留空間且其他資料集嘗試使用所有剩餘的空間時，會保留至少 10 GB 的空間供這個資料集使用。若要製作 storage/home/bob 的快照，該快照所使用的空間也會被列入保留空間計算。 [`refreservation`](#zfs-term-refreservation) 屬性也以類似的方式運作，但是他 _不包含_ 後裔，例如：快照。

不管那一種保留空間在許多情境皆很有用，例如：要規劃與測試磁碟空間配置在新系統上的適應性，或是確保有足夠的空間供稽查日誌或系統還原程序及檔案使用。

參考保留空間 (Reference Reservation)

`refreservation` 屬性可以確保對特定資料集 _不包含_ 其後裔最小可用的空間，這代表若在 storage/home/bob 設定 10 GB 的保留空間且其他資料集嘗試使用所有剩餘的空間時，會保留至少 10 GB 的空間供這個資料集使用。於正常 [reservation](#zfs-term-reservation) 不同的是，由快照及後裔資料集所使用的空間並不會列入保留空間計算。例如，若要製作一個 storage/home/bob 的快照，在 `refreservation` 空間之外必須要有足夠的空間才能成功完成這項操作，主資料集的後裔並不會列入 `refreservation` 空間額計算，所以也不會佔用保留空間。

修復 (Resilver)

當有磁碟故障且被更換後，新的磁碟必須回存先前所遺失的資料，會使用分散在其他磁碟上的奇偶校驗資訊來計算並寫入遺失的資料到新的磁碟機的這個程序稱作 _修復 (Resilvering)_。

上線 (Online)

一個儲存池或 vdev 處於線上 (`Online`) 狀態時代表所有該裝置的成員均已連結且正常運作。個別裝置處於線上 (`Online`) 狀態時代表功能正常。

離線 (Offline)

若有足夠的備援可避免儲存池或 vdev 進入故障 ([Faulted](#zfs-term-faulted)) 狀態，個別裝置若可由管理者設為離線 (`Offline`) 狀態，管理者可以選擇要設定那一個磁碟為離線來準備更換或是讓其更容易辨識。

降級 (Degraded)

一個儲存池或 vdev 處於降級 (`Degraded`) 狀態代表其有一個或多個磁碟已斷線或故障，此時儲存池仍可以使用，但只要再有其他的裝置故障，儲存池會無法復原。重新連線缺少的裝置或更換故障的磁碟，並在新裝置完成修復 ([Resilver](#zfs-term-resilver)) 程序可讓儲存池返回線上 ([Online](#zfs-term-online)) 狀態。

故障 (Faulted)

一個儲存池或 vdev 處於故障 (`Faulted`) 狀態代表無法運作，會無法存取在該裝置上的資料。當在 vdev 中缺少或故障的裝置數超過備援的層級，儲存池或 vdev 會進入故障 (`Faulted`) 狀態。若缺少的裝置可以重新連結上，儲存池便會返回線上 ([Online](#zfs-term-online)) 狀態。若沒有足夠的備援可補償故障的磁碟數量便會遺失儲存池中的內容且只能從備份還原。



**最後修改於**: March 9, 2024 由 [Danilo G. Baio](https://cgit.freebsd.org/doc/commit/?id=6199af92e7)
