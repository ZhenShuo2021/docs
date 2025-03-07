import React from "react";
import Link from "@docusaurus/Link";
import ThemedImage from "@theme/ThemedImage";

import styles from "./portfolio.module.css";

// import defaultImage from "../../../docs/docs/git/10-preliminaries/data/branch.webp";
const defaultImage = "https://picsum.photos/900/675?grayscale";
const maxTagsShow = 3;

const Tag = ({ children }) => <span className={styles.tag}>{children}</span>;

const Card = React.memo(({ item, type }) => (
  <div className={styles.card}>
    <div className={styles.cardImageContainer}>
      <Link to={item.link}>
        <ThemedImage
          sources={{
            light: item.image || defaultImage,
            dark: item.image || defaultImage,
          }}
          alt={item.title}
          className={styles.cardImage}
          loading="lazy"
        />
      </Link>
      {/* <div className={styles.cardOverlay}>
        <span className={styles.cardType}>{type}</span>
      </div> */}
    </div>

    <div className={styles.cardContent}>
      <h3 className={styles.cardTitle}>
        <Link to={item.link} className={styles.cardTitleLink}>
          {item.title}
        </Link>
      </h3>
      <p className={styles.description}>{item.description || ""}</p>
      <div className={styles.cardFooter}>
        {item.tags && item.tags.length > 0 && (
          <div className={styles.tagsContainer}>
            {item.tags.slice(0, maxTagsShow).map((tag, index) => (
              <Tag key={index}>{tag}</Tag>
            ))}
          </div>
        )}
      </div>
    </div>
  </div>
));

export { Card, Tag };
