import React, { useState, useEffect, useCallback } from 'react';
import useEmblaCarousel from 'embla-carousel-react';
import styles from './EmblaCarousel.module.css';

export default function EmblaCarousel({ 
  images, 
  captions = [],
  options = { 
    slidesToScroll: 1,
    align: 'center',
    containScroll: 'trimSnaps'
  } 
}) {
  const [emblaRef, emblaApi] = useEmblaCarousel(options);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const onSelect = useCallback(() => {
    if (!emblaApi) return;
    setSelectedIndex(emblaApi.selectedScrollSnap());
  }, [emblaApi]);

  useEffect(() => {
    if (!emblaApi) return;
    onSelect();
    emblaApi.on('select', onSelect);
    emblaApi.on('reInit', onSelect);
  }, [emblaApi, onSelect]);

  const scrollTo = useCallback((index) => {
    if (emblaApi) emblaApi.scrollTo(index);
  }, [emblaApi]);

  const getImageSrc = (image) => {
    if (typeof image === 'string') {
      return image;
    }
    return image.src;
  };
  
  const getImageKey = (image, index) => {
    const src = getImageSrc(image);
    return `slide-${src.split('/').pop()}-${index}`;
  };

  return (
    <div className={styles.embla}>
      <div 
        className={styles.emblaViewport} 
        ref={emblaRef}
      >
        <div className={styles.emblaContainer}>
          {images.map((image, index) => (
            <div 
              key={getImageKey(image, index)}
              className={styles.emblaSlide}
            >
              <div className={styles.emblaSlideInner}>
                <img 
                  src={getImageSrc(image)} 
                  alt={captions[index] || `Slide ${index + 1}`} 
                  className={styles.emblaSlideImg}
                />
                {captions[index] && (
                  <div className={styles.emblaCaption}>
                    {captions[index]}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className={styles.emblaDots}>
        {images.map((image, index) => (
          <button
            key={getImageKey(image, index)}
            type="button"
            onClick={() => scrollTo(index)}
            className={`${styles.emblaDot} ${
              index === selectedIndex ? styles.emblaDotSelected : ''
            }`}
            aria-label={`Go to slide ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
}
