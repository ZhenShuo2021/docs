import React from "react";
import styles from "./Contact.module.css";

// credit: https://from8to8.com/docs/Website/other/googleform/

export default function Contact() {
  return (
    <section style={{ padding: '0px', margin: '0px' }}>
      <div>
        <div className={styles.ContactForm}>
          <h3 style={{ textAlign: 'center' }}>這是一個 Google 表單</h3>
          <form
            className="form"
            target="_blank"
            rel="noreferrer noopenner"
            action="https://docs.google.com/forms/u/0/d/e/1FAIpQLSetpQ2m8BFZ8wq23r0hyIKFmpuf9jG5ivmfmAbJdjhhVQJl2A/formResponse"
            method="POST"
          >
            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                className="form-control"
                name="entry.416905759"
                placeholder="稱呼（選填）"
                style={{ flex: 1 }}
              />
              <input
                type="text"
                className="form-control"
                name="entry.111119796"
                placeholder="聯絡方式（選填）"
                style={{ flex: 1 }}
              />
            </div>
            <div className="form-group">
              <textarea
                name="entry.370110025"
                placeholder="內容"
                required
                style={{ width: '100%', marginTop: '10px' }}
              />
            </div>
            <input type="submit" value="送出" className={styles.submitButton} />
          </form>
        </div>
      </div>
    </section>
  );
}
