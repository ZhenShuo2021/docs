import { useState, useMemo, useEffect } from "react";

import { articlesMeta, projectsMeta } from "./config";

export const useDocsData = () => {
  const [docsMap, setDocsMap] = useState(new Map());

  useEffect(() => {
    const requireContext = require.context(
      "../../.docusaurus/docusaurus-plugin-content-docs",
      true,
      /site-docs-.*\.json$/
    );

    const loadAllDocs = async () => {
      const filePromises = requireContext.keys().map(async (file) => {
        try {
          const { permalink, frontMatter } = requireContext(file);
          return permalink ? [permalink, frontMatter] : null;
        } catch (error) {
          console.error(`無法讀取 ${file}:`, error);
          return null;
        }
      });

      const entries = await Promise.all(filePromises);
      const validEntries = entries.filter(Boolean);
      setDocsMap(new Map(validEntries));
    };

    loadAllDocs();
  }, []);

  return docsMap;
};

export const processFrontmatter = (item, docsMap) => {
  if (item.link?.startsWith("/")) {
    const frontmatter = docsMap.get(item.link) || {};
    return {
      ...item,
      title: item.title?.trim() || frontmatter.title || "找不到標題",
      description: item.description?.trim() || frontmatter.description || "",
      image: item.image?.trim() || frontmatter.image,
      tags: item.tags || frontmatter.tags || [""],
    };
  }

  return item;
};

export function getPortfolioContent(docsMap) {
  return useMemo(() => {
    const displayedArticles = articlesMeta.map((item) =>
      processFrontmatter(item, docsMap)
    );
    const displayedProjects = projectsMeta.map((item) =>
      processFrontmatter(item, docsMap)
    );

    return {
      articlesID: displayedArticles,
      projectsID: displayedProjects,
    };
  }, [docsMap]);
}
