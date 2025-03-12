import { useState, useMemo, useEffect } from "react";

import { articlesMeta, projectsMeta } from "./config";

export const useDocsData = () => {
  const [docsMap, setDocsMap] = useState(new Map());

  useEffect(() => {
    const loadDocs = async () => {
      const latestPosts = require("../components/LatestPosts/latest-posts.json");
      const entries = latestPosts.map((item) => [item.permalink, item]);
      setDocsMap(new Map(entries));
    };

    loadDocs();
  }, []);

  return docsMap;
};

export const processFrontmatter = (item, docsMap) => {
  if (item.link?.startsWith("/")) {
    const frontmatter = docsMap.get(item.link) || {};
    const tags = frontmatter.tags ? frontmatter.tags.map(tag => tag.label) : (item.tags || [""]);
    return {
      ...item,
      title: item.title?.trim() || frontmatter.title || "找不到標題",
      description: item.description?.trim() || frontmatter.description || "",
      image: item.image?.trim() || frontmatter.image,
      tags,
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
