const ShinyText = ({ text, disabled = false, speed = 5, className = '' }) => {
  const animationDuration = `${speed}s`;

  return (
    <span
      className={`inline-block ${className}`}
      style={{
        backgroundImage:
          'linear-gradient(120deg, #b5b5b5a4 40%, rgba(255, 255, 255, 0.9) 50%, #b5b5b5a4 60%)',
        backgroundSize: '200% 100%',
        backgroundPosition: disabled ? '0% 0%' : '100% 0%',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        backgroundClip: 'text',
        color: '#b5b5b5a4',
        ...(disabled ? {} : {
          animation: `shine ${animationDuration} linear infinite`
        })
      }}
    >
      {text}
    </span>
  );
};

export default ShinyText;

