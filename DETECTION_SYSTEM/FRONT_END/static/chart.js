const ctx = document.getElementById('emotionChart').getContext('2d');

const emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const emojis = ['😠', '🤢', '😱', '😄', '😢', '😲', '😐'];

let emotionCounts = {};
emotions.forEach(emotion => emotionCounts[emotion] = 0);

const barColors = [
  'rgba(107, 144, 128, 0.8)',
  'rgba(107, 144, 128, 0.7)',
  'rgba(107, 144, 128, 0.6)',
  'rgba(107, 144, 128, 0.5)',
  'rgba(107, 144, 128, 0.4)',
  'rgba(107, 144, 128, 0.3)',
  'rgba(107, 144, 128, 0.2)'
];

// Draw emoji above bars and percentage inside bars
const emojiAndPercentPlugin = {
  id: 'emojiAndPercentPlugin',
  afterDatasetsDraw(chart) {
    const { ctx, data, chartArea: { top, bottom, left, right }, scales: { x, y } } = chart;

    ctx.save();
    ctx.textAlign = 'center';
    ctx.fillStyle = '#f6fff8';
    ctx.font = '20px Arial';

    data.datasets[0].data.forEach((value, index) => {
      const xPos = x.getPixelForValue(index);
      const yPos = y.getPixelForValue(value);

      // Draw emoji above bar
      ctx.fillText(emojis[index], xPos, yPos - 10);

      // Draw percentage inside the bar
      const barTop = y.getPixelForValue(value);
      const barBottom = y.getPixelForValue(0);
      const barHeight = barBottom - barTop;

      ctx.textBaseline = 'middle';
      ctx.font = '14px Arial';
      ctx.fillText(`${value.toFixed(1)}%`, xPos, barTop + barHeight / 2);
    });

    ctx.restore();
  }
};

const emotionChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: emotions,
    datasets: [{
      label: 'Emotion Percentage',
      data: emotions.map(() => 0),
      backgroundColor: barColors,
      borderRadius: 6,
      borderSkipped: false,
    }]
  },
  options: {
    responsive: true,
    animation: { duration: 500 },
    scales: {
      y: {
        beginAtZero: true,
        max: 100, // Lock to 100%
        ticks: {
          color: '#f6fff8',
          stepSize: 20,
          callback: val => `${val}%` // Show percentage ticks
        },
        grid: { color: 'rgba(255,255,255,0.1)' }
      },
      x: {
        ticks: { color: '#f6fff8', font: { size: 14 } },
        grid: { display: false }
      }
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true,
        callbacks: {
          label: context => `${context.parsed.y.toFixed(1)}%`
        }
      }
    }
  },
  plugins: [emojiAndPercentPlugin]
});

let lastSpokenEmotion = null;

function updateEmotionData(emotion) {
  emotionCounts[emotion] += 1;

  const totalCount = Object.values(emotionCounts).reduce((a, b) => a + b, 0);
  const percentages = emotions.map(em => (emotionCounts[em] / totalCount) * 100);

  emotionChart.data.datasets[0].data = percentages;
  emotionChart.update();

  // Find most dominant emotion
  const maxPercentage = Math.max(...percentages);
  const maxIndex = percentages.indexOf(maxPercentage);

  const dominantEmotion = emotions[maxIndex];
  const dominantEmoji = emojis[maxIndex];

  // ✅ Update Prediction Box (Emotion + Emoji)
  const predictionText = document.getElementById('predictionText');
  predictionText.textContent = `You look ${dominantEmotion} ${dominantEmoji}`;

  // ✅ Change Prediction Box Color Based on Emotion
  const box = document.querySelector('.prediction-box');
  switch (dominantEmotion) {
    case 'Happy':
      box.style.backgroundColor = '#a4c3b2'; // Light green
      break;
    case 'Sad':
      box.style.backgroundColor = '#9a8c98'; // Purple
      break;
    case 'Angry':
      box.style.backgroundColor = '#e76f51'; // Red
      break;
    case 'Surprise':
      box.style.backgroundColor = '#f4a261'; // Orange
      break;
    case 'Fear':
      box.style.backgroundColor = '#264653'; // Dark blue
      break;
    case 'Disgust':
      box.style.backgroundColor = '#2a9d8f'; // Teal
      break;
    case 'Neutral':
      box.style.backgroundColor = '#cce3de'; // Grey
      break;
    default:
      box.style.backgroundColor = '#a4c3b2'; // Default
  }

  // ✅ Speak the Emotion using Web Speech API
  if (dominantEmotion !== lastSpokenEmotion) {
    speakEmotion(dominantEmotion);
    lastSpokenEmotion = dominantEmotion;
  }
}

// ✅ Function to Speak the Emotion
function speakEmotion(emotion) {
  const msg = new SpeechSynthesisUtterance(`You look ${emotion}`);
  msg.lang = 'en-US';
  msg.rate = 1; // Speed of speech
  window.speechSynthesis.cancel(); // Stop any previous speech
  window.speechSynthesis.speak(msg);
}
