import React from 'react';
import Giscus from "@giscus/react";
import { useColorMode } from '@docusaurus/theme-common';

export default function GiscusComment() {
    const { colorMode } = useColorMode();

    return (
        <Giscus
            repo="ZhenShuo2021/docs"
            repoId="R_kgDOMuZlHQ"
            category="Announcements"
            categoryId="DIC_kwDOMuZlHc4CiStx"
            mapping="url"
            strict="0"
            reactionsEnabled="1"
            emitMetadata="0"
            inputPosition="top"
            theme={colorMode}
            lang="zh-TW"
            loading="lazy"
            crossorigin="anonymous"
            async
        />
    );
}