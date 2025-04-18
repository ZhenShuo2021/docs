import React, { useState, useRef } from 'react';
import ReactPlayer from 'react-player';

const VideoPlayer = ({ 
  url, 
  maxWidth = '100%', 
  borderRadius = '0px', 
  customStyles = {}, 
  autoPlay = false,
  marginTop = '20px',    // 新增上边距参数
  marginBottom = '20px'  // 新增下边距参数
}) => {
  // 默认使用16:9的比例（56.25%），但会在视频加载后更新
  const [aspectRatio, setAspectRatio] = useState(56.25);
  const [isLoaded, setIsLoaded] = useState(false);
  const playerRef = useRef(null);

  const handleReady = () => {
    if (playerRef.current) {
      const player = playerRef.current.getInternalPlayer();
      if (player && player.videoWidth && player.videoHeight) {
        const videoRatio = (player.videoHeight / player.videoWidth) * 100;
        setAspectRatio(videoRatio);
      }
      setIsLoaded(true);
    }
  };

  return (
    <div style={{ 
      maxWidth: maxWidth, 
      width: '100%', 
      margin: '0 auto', 
      marginTop: marginTop,     // 添加上边距
      marginBottom: marginBottom, // 添加下边距
      borderRadius: borderRadius, 
      overflow: 'hidden', 
      ...customStyles 
    }}>
      <div style={{
        position: 'relative',
        paddingTop: `${aspectRatio}%`, // 基于实际视频比例
        width: '100%',
        transition: 'padding-top 0.3s ease',
        borderRadius: borderRadius,
        overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          borderRadius: borderRadius,
          overflow: 'hidden'
        }}>
          <ReactPlayer
            ref={playerRef}
            url={url}
            controls
            width='100%'
            height='100%'
            style={{ 
              position: 'absolute', 
              top: 0, 
              left: 0,
              opacity: isLoaded ? 1 : 0, 
              transition: 'opacity 0.3s ease' 
            }}
            playing={autoPlay}
            onReady={handleReady}
            onError={() => setIsLoaded(true)}
          />
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;
