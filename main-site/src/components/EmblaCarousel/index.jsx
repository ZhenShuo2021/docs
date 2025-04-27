import React, { useState, useEffect, useCallback } from 'react';
import useEmblaCarousel from 'embla-carousel-react';
import styles from './EmblaCarousel.module.css';

const EmblaCarousel = ({ 
  images, 
  captions = [],
  width = '100%',
  options = { 
    loop: true,
    slidesToScroll: 1,
    align: 'center',
    containScroll: 'trimSnaps'
  } 
}) => {
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
    
    return () => {
      emblaApi.off('select', onSelect);
      emblaApi.off('reInit', onSelect);
    };
  }, [emblaApi, onSelect]);

  const scrollTo = useCallback((index) => {
    if (emblaApi) emblaApi.scrollTo(index);
  }, [emblaApi]);

  const normalizeImageSource = (image) => {
    return typeof image === 'string' ? image : image.src;
  };
  
  const generateSlideKey = (image, index) => {
    const src = normalizeImageSource(image);
    const filename = src.split('/').pop();
    return `slide-${filename}-${index}`;
  };

  return (
    <div className={styles.embla} style={{ width }}>
      <div className={styles.emblaViewport} ref={emblaRef}>
        <div className={styles.emblaContainer}>
          {images.map((image, index) => (
            <div 
              key={generateSlideKey(image, index)}
              className={styles.emblaSlide}
            >
              <div className={styles.emblaSlideInner}>
                <img 
                  src={normalizeImageSource(image)} 
                  alt={captions[index] || `Slide ${index + 1}`} 
                  className={styles.emblaSlideImg}
                />
              </div>
              
              {captions[index] && (
                <div className={styles.emblaCaption}>
                  {captions[index]}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      
      <div className={styles.emblaDots}>
        {images.map((image, index) => (
          <button
            key={generateSlideKey(image, index)}
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
};

export default EmblaCarousel;