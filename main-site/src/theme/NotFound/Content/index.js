import React, { useEffect, useState } from "react";
import clsx from "clsx";
import Translate from "@docusaurus/Translate";
import Heading from "@theme/Heading";
import stringSimilarity from "string-similarity";

const SITE_URLS_PATH = "/site-urls.json";

export default function NotFoundContent({ className }) {
  const [isChecking, setIsChecking] = useState(true);
  const [similarPages, setSimilarPages] = useState([]);
  const [redirectUrl, setRedirectUrl] = useState(null);
  const [redirectTitle, setRedirectTitle] = useState(null);

  useEffect(() => {
    async function findMatchingUrls() {
      try {
        const currentPath = window.location.pathname;
        const segments = currentPath.split("/");
        const lastSegment = segments[segments.length - 1];

        if (!lastSegment) {
          setIsChecking(false);
          return;
        }

        const response = await fetch(SITE_URLS_PATH);
        const sitePages = await response.json();

        const exactMatchPage = sitePages.find((page) => {
          const urlSegments = new URL(
            page.permalink,
            window.location.origin
          ).pathname.split("/");
          const urlLastSegment = urlSegments[urlSegments.length - 1];
          return lastSegment === urlLastSegment;
        });

        if (exactMatchPage) {
          console.log(
            `找到完全匹配頁面: ${exactMatchPage.title} - ${exactMatchPage.permalink}`
          );
          setRedirectUrl(exactMatchPage.permalink);
          setRedirectTitle(exactMatchPage.title);

          setTimeout(() => {
            window.location.href = exactMatchPage.permalink;
          }, 500);

          return;
        }

        // 計算相似度
        const pagesWithLastSegment = sitePages.map((page) => {
          const urlSegments = new URL(
            page.permalink,
            window.location.origin
          ).pathname.split("/");
          return {
            ...page,
            lastSegment: urlSegments[urlSegments.length - 1],
          };
        });

        const similarities = pagesWithLastSegment.map((page) => ({
          ...page,
          similarity: stringSimilarity.compareTwoStrings(
            lastSegment,
            page.lastSegment
          ),
        }));

        similarities.sort((a, b) => b.similarity - a.similarity);

        const topThree = similarities
          .slice(0, 3)
          .filter((page) => page.similarity > 0.3);
        setSimilarPages(topThree);
        setIsChecking(false);
      } catch (error) {
        console.error("檢查頁面時出錯:", error);
        setIsChecking(false);
      }
    }

    findMatchingUrls();
  }, []);

  if (isChecking) {
    return (
      <main className={clsx("container margin-vert--xl", className)}>
        <div className="row">
          <div className="col col--6 col--offset-3">
            <p>正在檢查是否有匹配頁面...</p>
          </div>
        </div>
      </main>
    );
  }

  if (redirectUrl) {
    return (
      <main className={clsx("container margin-vert--xl", className)}>
        <div className="row">
          <div className="col col--6 col--offset-3">
            <p>找到匹配頁面，正在跳轉到「{redirectTitle}」...</p>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className={clsx("container margin-vert--xl", className)}>
      <div className="row">
        <div className="col col--6 col--offset-3">
          <Heading as="h1" className="hero__title">
            <Translate
              id="theme.NotFound.title"
              description="The title of the 404 page"
            >
              Page Not Found
            </Translate>
          </Heading>
          <p>
            <Translate
              id="theme.NotFound.p1"
              description="The first paragraph of the 404 page"
            >
              We could not find what you were looking for.
            </Translate>
          </p>

          {similarPages.length > 0 && (
            <>
              <p>您可能想找的是以下頁面：</p>
              <ul>
                {similarPages.map((page, index) => (
                  <li key={index}>
                    <a href={page.permalink}>{page.title}</a>
                  </li>
                ))}
              </ul>
            </>
          )}

          <p>
            <Translate
              id="theme.NotFound.p2"
              description="The 2nd paragraph of the 404 page"
            >
              Please contact the owner of the site that linked you to the
              original URL and let them know their link is broken.
            </Translate>
          </p>
        </div>
      </div>
    </main>
  );
}
