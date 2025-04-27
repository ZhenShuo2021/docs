import React from "react";
import Link from "@docusaurus/Link";
import ThemedImage from "@theme/ThemedImage";
import clsx from "clsx";

import styles from "./navigation.module.css";

// Default image if none provided
const defaultImage = "https://picsum.photos/900/675?grayscale";
const maxTagsShow = 3;

const Tag = ({ children }) => <span className={styles.tag}>{children}</span>;

const Card = ({ item, type }) => (
  <div className={clsx("card", styles.card)}>
    <Link to={item.link} className="card__link">
      <div className={styles.cardImageContainer}>
        <ThemedImage
          sources={{
            light: item.image || defaultImage,
            dark: item.image || defaultImage,
          }}
          alt={item.title}
          className={styles.cardImage}
          loading="lazy"
        />
      </div>
    </Link>

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
);

export { Card, Tag };