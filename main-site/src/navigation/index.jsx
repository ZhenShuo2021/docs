import React from "react";
import Layout from "@theme/Layout";
import BackToTopButton from "@theme/BackToTopButton";

import styles from "./navigation.module.css";
import { getNavigationData } from "./content";
import { sectionsMeta } from "./config";
import { Card } from "./components";
import { Hero } from "../components/Hero";

function Section({ section, content }) {
  return (
    <section id={section.id} className="padding-top--lg">
      <h2 className="text--center margin-bottom--md">{section.title}</h2>
      <hr className="margin-bottom--lg" />
      <div className={styles.cardGrid}>
        {content.map((item, itemIndex) => (
          <Card
            key={`${section.id}-${itemIndex}`}
            item={item}
            type={section.contentType}
          />
        ))}
      </div>
    </section>
  );
}

function NavigationContent({ sectionContents }) {
  return (
    <div className={styles.navigationContainer}>
      {sectionsMeta.map((section) => (
        <Section
          key={section.id}
          section={section}
          content={sectionContents[section.id]}
        />
      ))}
    </div>
  );
}

export function Navigation() {
  const sectionContents = getNavigationData();

  return (
    <Layout title="導航" description="最近的文章和專案">
      <Hero
        title="導航"
        tagline="最近的文章和專案"
        className="hero--primary"
      />
      <main>
        <NavigationContent sectionContents={sectionContents} />
      </main>
      <BackToTopButton />
    </Layout>
  );
}