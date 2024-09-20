---
title: 在 NAS 中設定 Traefik 反向代理 Docker 服務
description: 如何在 NAS 中設定 Traefik 反向代理 Docker 服務，包含 DDNS 教學
tags:
  - NAS
  - Linux
  - Traefik
keywords:
  - NAS
  - Linux
  - Traefik
  - DDNS
  - Reverse-Proxy
  - 反向代理
last_update:
  date: 2024-09-21 GMT+8
  author: zsl0621
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 在 NAS 中設定 Traefik 反向代理 Docker 服務
本文記錄使用 NAS 中如何使用 Traefik 反向代理，在把各種 docker 服務反向代理使用，並且自動簽署 TLS 證書。

沒有很複雜但是再度被 Google 搜尋搞，前三篇文章放了三個不同的設定方式然後我就炸了，其實只有一種最常使用，本文專注介紹 NAS 如何設定，作為用戶我們只關心設定是否成功，外網連接是否安全，所以本文不會有任何原理，沒有由淺入深教學，就是複製貼上完事，快點設定完成服務上線才是最重要的。

## 為何使用 Traefik
常見的反向代理供具有 Nginx、Apache、Caddy 和 Cloudflare Tunnel。一樣先說明為何選擇 Traefik。
- Cloudflare Tunnel: 最好的反向代理，唯一缺點是單一檔案大小上限 100MB，並且不需開 port
- Nginx: 設定複雜，沒有自動證書更新，需要複雜設定才使用他。
- Apache: 單純是反向代理 Docker 服務的文章很少，我沒試過。
- Caddy: 設定非常簡單，但他不是專注在反向代理，所以需要過濾很多文章。
- Traefik: 專注反向代理，設定簡單，自動證書更新，文檔充足，Docker 支援完善

所以我也是到需要使用個人雲端硬碟的才開始使用 Traefik，Cloudflare Tunnel 光是免費版就可以完成 99% 需求，更別說整合 Zero Trust Access 後基本上安全無敵。

## 安全性
就算使用 Traefik 還是可以接上 Cloudflare Zero Trust 設定 Access 保護，如使用 OAuth 認證；或者是 DNS 開啟 Cloudflare proxy （預設開啟除非特地取消），這樣可隱藏主機 IP 地址避免主機被攻擊；開啟 Cloudflare proxy 後還可以設定 WAF，如最基本的國籍封鎖，或者是開啟 Bot fight mode/DDoS 等 Cloudflare 內建的免費服務。

我知道中英混雜看了眼睛很痛，這裡總結：
- Zero Trust Access with OAuth
- Cloudflare proxy to hide IP
- WAF/DDoS protection (requires Cloudflare proxy)

也就是說攻擊者需要越過三道安全防線。首先得面對 WAF 和 DDoS 保護 的挑戰，再繞過 Zero Trust Access 認證，最後是 Cloudflare 代理才能知道主機的真實 IP 地址。更重要的是，即使攻擊者突破了這些防線接觸到的也只是 Traefik 的反向代理層，並沒有直接接觸到真正的主機或應用服務器。

## 使用環境
- 系統：Ubuntu server
- traefik 版本：在 2.9 和 3.1 都可正常運行。
- 以 [filebrowser](https://github.com/filebrowser/filebrowser) 示範

如何設定 DDNS 請見先前文章。

## 方法一：合併其他容器部署 (BAD)
此設定基本和[官方教學](https://doc.traefik.io/traefik/user-guides/docker-compose/acme-tls/)一致，但是要部署多個容器時就有問題，可以當作一個基本範例練習。

```yaml
services:
  traefik:
    image: traefik:latest
    container_name: traefik
    command:
      - "--providers.docker=true"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
      - "--certificatesresolvers.myresolver.acme.email=leo01412123@gmail.com"
      - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
    ports:
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./letsencrypt:/letsencrypt
    networks:
      - proxy

  filebrowser:
    container_name: filebrowser
    image: filebrowser/filebrowser:s6
    restart: always
    volumes:
      - /mnt/filebrowser/srv:/srv
      - /home/leo/docker/filebrowser/filebrowser.db:/database/filebrowser.db
      - /home/leo/docker/filebrowser/settings.json:/config/settings.json
    environment:
      - PUID=$(id -u)
      - PGID=$(id -g)
    networks:
      - proxy
    ports:
      - 80:80
    labels:
      # 設定 filebrowser 在 traefik 中的路由
      - "traefik.enable=true"
      - "traefik.http.routers.filebrowser.rule=Host(`filebrowser.zsl0621.cc`)"
      - "traefik.http.routers.filebrowser.entrypoints=websecure"
      - "traefik.http.routers.filebrowser.tls.certresolver=myresolver"

      # 設定 security headers
      - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.X-Content-Type-Options=nosniff"
      # 啟用 middlewares
      - "traefik.http.routers.filebrowser.middlewares=securityHeaders"

networks:
  proxy:
    external: true
```

這個方法和官方教學的差異在於多了一個 `proxy` docker network，並且加入 security headers。這是符合大多數教學的範例。使用時先啟用網路 `docker network create proxy` 後即可 `docker compose up -d` 啟動容器。

## 方法二：獨立部署 (Good)
這個方法獨立 traefik 和其他應用部署，方便調整。

在 `docker/traefik` 中建立 `docker-compose.yml` 和 `letsencrypt` 資料夾：

```sh
mkdir letsencrypt
vim docker-compose.yml
```

然後兩個 yaml 檔案分別是：


<Tabs
    values={[
        { label: 'docker/traefik', value: 'traefik', },
        { label: 'docker/filebrowser', value: 'filebrowser', },
    ]
    }>
  <TabItem value="traefik">

    ```yaml
    services:
      traefik:
        image: traefik:latest
        container_name: traefik
        restart: always
        command:
          - "--log.level=DEBUG"
          - "--providers.docker=true"
          - "--entrypoints.websecure.http3"
          - "--entrypoints.websecure.address=:443"
          - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
          - "--certificatesresolvers.myresolver.acme.email=leo01412123@gmail.com"
          - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
          - "--api.insecure=true"
          - "--api.dashboard=true"
        ports:
          - "443:443"
          - "4315:8080"
        volumes:
          - /var/run/docker.sock:/var/run/docker.sock:ro
          - ./letsencrypt:/letsencrypt
        networks:
          - proxy
        labels:
          # Traefik dashboard
          - "traefik.http.routers.dashboard.rule=Host(`traefik.example.com`) && (PathPrefix(`/api`) || PathPrefix(`/dashboard`))"
          - "traefik.http.routers.dashboard.service=api@internal"
          - "traefik.http.routers.dashboard.middlewares=auth"
          - "traefik.http.middlewares.auth.basicauth.users=test:$$apr1$$H6uskkkW$$IgXLP6ewTrSuBkTrqE8wj/,test2:$$apr1$$d9hr9HBB$$4HxwgUir3HP4EsggP/QNo0"

          # Security headers
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Permissions-Policy=geolocation=(self), microphone=(), camera=(), fullscreen=*"
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.X-Frame-Options=SAMEORIGIN"
                - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.X-XSS-Protection=1; mode=block"
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Referrer-Policy=strict-origin-when-cross-origin"
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Strict-Transport-Security=max-age=63072000; includeSubDomains; preload"
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Access-Control-Allow-Origin=https://*.zsl0621.cc"
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Access-Control-Allow-Methods=GET, POST, OPTIONS"
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Access-Control-Allow-Headers=Content-Type, Authorization"
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Access-Control-Expose-Headers=X-Total-Count"

          # 簡易版 CSP 設定，到 https://developer.mozilla.org/en-US/observatory 測試 80 分
          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Content-Security-Policy=default-src 'self' 'unsafe-inline' https: data: blob:"


          # 手動版 CSP 設定，測試 115 分，但不可能去維護這東西
          #- "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.Content-Security-Policy=default-src 'self'; script-src 'self' 'sha256-244nvlHWdCM1kA8RZKfY3bUf3110bXhBCG22rCB4Ctw=' 'sha256-VA8O2hAdooB288EpSTrGLl7z3QikbWU9wwoebO/QaYk=' 'sha256-+5XkZFazzJo8n0iOP4ti/cLCMUudTf//Mzkb7xNPXIc=' https://cdnjs.cloudflare.com; style-src 'self' 'sha256-ZLN3N05ogeY2hIjEaMJwxgAJCpLsgSAoVZsTatKONMo=' 'sha256-w+cEYFEmAoad15jfTJgep5pwxjGpd9UJSHmyuIz6eos=' 'sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU='; img-src 'self' data:; font-src 'self' data: https://fonts.gstatic.com; manifest-src 'self' blob:; connect-src 'self'; frame-ancestors 'self'"

          - "traefik.http.middlewares.blockMetadataServer.headers.customResponseHeaders.X-Block=This request is blocked"
          - "traefik.http.routers.myrouter.middlewares=blockMetadataServer"
          - "traefik.http.routers.myrouter.rule=Host(`169.254.169.254`)"

          - "traefik.http.middlewares.securityHeaders.headers.customResponseHeaders.X-Content-Type-Options=nosniff"

    networks:
      proxy:
        external: true
    ```

  </TabItem>
  <TabItem value="filebrowser">

    ```yaml
    services:
      filebrowser:
        container_name: filebrowser
        image: filebrowser/filebrowser:s6
        restart: always
        volumes:
          - /mnt/filebrowser/srv:/srv
          - /home/leo/docker/filebrowser/filebrowser.db:/database/filebrowser.db
          - /home/leo/docker/filebrowser/settings.json:/config/settings.json
        environment:
          - PUID=$(id -u)
          - PGID=$(id -g)
        networks:
          - proxy
        ports:
          - 80:80
        labels:
          - "traefik.enable=true"
          - "traefik.http.routers.filebrowser.rule=Host(`filebrowser.zsl0621.cc`)"
          - "traefik.http.routers.filebrowser.entrypoints=websecure"
          - "traefik.http.routers.filebrowser.tls.certresolver=myresolver"
          - "traefik.http.routers.filebrowser.middlewares=securityHeaders"

    networks:
      proxy:
        external: true
    ```

  </TabItem>
</Tabs>

啟用方法一樣是先啟用網路 `docker network create proxy`，之後分別在兩個資料夾 `docker compose up -d` 啟動容器。與本文方式最相近的文章應該是 [Docker 下的 Traefik 上手教程](https://blog.bling.moe/post/14/) 和 [Traefik v3.0 Docker 全面使用指南：基础篇](https://soulteary.com/2023/07/18/traefik-v3-docker-comprehensive-user-guide-basics.html) ，可作為遇到問題時的參考。

## 下一步
使用 [geoblock](https://github.com/PascalMinder/GeoBlock) [fail2ban](https://github.com/tomMoulard/fail2ban) 避免對 IP 掃描的直接攻擊，完成 [OIDC](https://doc.traefik.io/traefik-enterprise/middlewares/oidc/) 設訂。

https://github.com/Tarow/traefik-geoblock-example/

## 結語
整理完結果很簡單，但是從測試到成功不知道打了幾百次 vim docker-compose.yml。