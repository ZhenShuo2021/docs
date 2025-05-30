import Link from "@docusaurus/Link";
import styles from "./LatestPosts.module.css";
import postsData from "@site/src/data/latest-posts.json";

function Post({ title, permalink, tags, yearMonth, day }) {
  return (
    <div className={styles.latest_post_row_item} key={permalink}>
      <div className={styles.post_list_date_container}>
        <div className={styles.post_list_date_yearmonth}>{yearMonth}</div>
        <div>
          <div className={styles.post_list_date_day}>{day}</div>
        </div>
      </div>
      <div className={styles.latest_post_row_item_title}>
        <Link to={permalink} key={permalink}>
          {title}
        </Link>
      </div>
      <div className={styles.latest_post_row_item_tags}>
        {tags.length > 0 &&
          tags.slice(0, 2).map(({ label, permalink: tagPermalink }, index) => (
            <Link
              key={tagPermalink}
              className={`${styles.post__tags} ${index < tags.length ? "margin-right--sm" : ""}`}
              to={tagPermalink}
              style={{
                fontSize: "0.75em",
                fontWeight: 500,
              }}
            >
              {label}
            </Link>
          ))}
      </div>
    </div>
  );
}

export default function LatestPosts() {
  const latestPosts = postsData.slice(0, Math.min(postsData.length, 10));

  return (
    <section className={styles.latestPosts}>
      <div className="container">
        <div className={styles.latest_post_row}>
          {latestPosts.map((props, idx) => (
            <Post key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
