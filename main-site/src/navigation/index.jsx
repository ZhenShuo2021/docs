import React, { useState, useEffect } from "react";
import Layout from "@theme/Layout";
import BackToTopButton from "@theme/BackToTopButton";
import clsx from "clsx";
import Link from "@docusaurus/Link";

import styles from "./navigation.module.css";
import { getNavigationData } from "./content";
import { sectionsMeta } from "./config";
import { Card } from "./components";
import { Hero } from "../components/Hero";

function Section({ section, content, isListView }) {
  return (
    <section id={section.id} className="padding-top--lg">
      <h2 className="text--center margin-bottom--md">{section.title}</h2>
      <hr className="margin-bottom--lg" />
      {isListView ? (
        <ul className="padding--none">
          {content.map((item, itemIndex) => (
            <li key={`${section.id}-${itemIndex}`} className="margin-bottom--sm">
              <Link to={item.link} className="text--primary">
                {item.title}
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <div className={styles.cardGrid}>
          {content.map((item, itemIndex) => (
            <Card
              key={`${section.id}-${itemIndex}`}
              item={item}
              type={section.contentType}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function NavigationContent({ sectionContents, isListView }) {
  return (
    <div className={styles.navigationContainer}>
      {sectionsMeta.map((section) => (
        <Section
          key={section.id}
          section={section}
          content={sectionContents[section.id]}
          isListView={isListView}
        />
      ))}
    </div>
  );
}

export function Navigation() {
  const sectionContents = getNavigationData();
  const [isListView, setIsListView] = useState(false);
  
  useEffect(() => {
    const savedViewMode = localStorage.getItem("viewMode");
    if (savedViewMode) {
      setIsListView(savedViewMode === "list");
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("viewMode", isListView ? "list" : "card");
  }, [isListView]);

  const toggleView = () => {
    setIsListView((prev) => !prev);
  };

  return (
    <Layout title="導航" description="最近的文章和專案">
      <Hero
        title="導航"
        tagline="最近的文章和專案"
        className="hero--primary"
      />
      <main>
        <button
          onClick={toggleView}
          className={clsx(styles.floatingButton, "button button--secondary")}
          title={isListView ? "切換到卡片檢視" : "切換到列表檢視"}
        >
          {isListView ? "卡片" : "列表"}
        </button>
        
        <NavigationContent 
          sectionContents={sectionContents} 
          isListView={isListView} 
        />
      </main>
      <BackToTopButton />
    </Layout>
  );
}