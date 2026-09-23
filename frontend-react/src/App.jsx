import React, { useState } from 'react';
import { 
  Sparkles, 
  Film, 
  AlertCircle, 
  CheckCircle2, 
  XCircle, 
  MinusCircle, 
  ArrowRight, 
  RotateCcw,
  Layers,
  Eye
} from 'lucide-react';

const API_ENDPOINT = "http://127.0.0.1:8000/analyze";

const SAMPLES = [
  {
    label: "अभिनय + संगीत (Positive)",
    text: "चित्रपटात कलाकारांचा अभिनय अप्रतिम आहे आणि संगीत सुद्धा खूप सुंदर आहे."
  },
  {
    label: "मिश्र मत (Mixed Aspects)",
    text: "कथा खूपच कंटाळवाणी होती, पण गाणी छान वाटली."
  },
  {
    label: "नकारात्मक (Negative)",
    text: "एकदम बकवास चित्रपट, वेळ आणि पैसा दोन्ही वाया गेले."
  }
];

export default function App() {
  const [review, setReview] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!review.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: review.trim() })
      });

      if (!response.ok) {
        throw new Error('Backend inference failed. Ensure FastAPI is running.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentTheme = (sentiment) => {
    switch (sentiment) {
      case 'Positive':
        return {
          badgeBg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
          glow: 'from-emerald-500/20 to-emerald-500/0',
          barColor: 'bg-emerald-500',
          icon: <CheckCircle2 className="w-5 h-5 text-emerald-400" />
        };
      case 'Negative':
        return {
          badgeBg: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
          glow: 'from-rose-500/20 to-rose-500/0',
          barColor: 'bg-rose-500',
          icon: <XCircle className="w-5 h-5 text-rose-400" />
        };
      default:
        return {
          badgeBg: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
          glow: 'from-amber-500/20 to-amber-500/0',
          barColor: 'bg-amber-500',
          icon: <MinusCircle className="w-5 h-5 text-amber-400" />
        };
    }
  };

  const theme = result ? getSentimentTheme(result.sentiment) : null;

  return (
    <div className="min-h-screen py-10 px-4 sm:px-6 flex flex-col items-center justify-center">
      <div className="w-full max-w-3xl space-y-6">
        
        {/* Header */}
        <header className="text-center space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-medium tracking-wide">
            <Sparkles className="w-3.5 h-3.5" />
            <span>L3Cube MahaSent-MR • NLP Intelligence</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight font-mukta text-transparent bg-clip-text bg-gradient-to-r from-slate-100 via-slate-200 to-indigo-200">
            मराठी चित्रपट समीक्षा विश्लेषण
          </h1>
          <p className="text-slate-400 text-sm max-w-lg mx-auto">
            Deep Aspect-Based Sentiment Mining and Word-Level Explainability (XAI)
          </p>
        </header>

        {/* Input Card */}
        <div className="backdrop-blur-xl bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 sm:p-7 shadow-2xl relative overflow-hidden">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm font-semibold text-slate-300 flex items-center gap-2 font-mukta text-base">
                <Film className="w-4 h-4 text-indigo-400" />
                समीक्षा प्रविष्ट करा (Enter Review):
              </label>
              {review && (
                <button
                  onClick={() => { setReview(''); setResult(null); }}
                  className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 transition"
                >
                  <RotateCcw className="w-3 h-3" /> हटवा (Clear)
                </button>
              )}
            </div>

            <textarea
              value={review}
              onChange={(e) => setReview(e.target.value)}
              rows="4"
              placeholder="उदा. चित्रपटाची कथा आणि गाणी अप्रतिम आहेत, पण दिग्दर्शन कंटाळवाणे वाटले..."
              className="w-full rounded-xl bg-slate-950/70 border border-slate-800 p-4 text-slate-100 font-mukta text-lg placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/60 transition resize-none"
            />

            {/* Test Samples */}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="text-xs text-slate-500 font-medium">चाचणी उदाहरणे:</span>
              {SAMPLES.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => setReview(sample.text)}
                  className="text-xs px-2.5 py-1 rounded-lg bg-slate-800/70 hover:bg-slate-800 text-slate-300 border border-slate-700/50 transition active:scale-95"
                >
                  {sample.label}
                </button>
              ))}
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Analyze Action */}
            <button
              onClick={handleAnalyze}
              disabled={loading || !review.trim()}
              className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 active:scale-[0.99] text-white font-medium text-sm shadow-lg shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 transition duration-200 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>विश्लेषण सुरू आहे (Processing)...</span>
                </>
              ) : (
                <>
                  <span>विश्लेषण करा (Analyze Sentiment)</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        {result && (
          <div className="space-y-4 animate-fade-in">
            
            {/* Main Score Banner */}
            <div className={`p-5 rounded-2xl border backdrop-blur-xl bg-slate-900/80 relative overflow-hidden flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${theme.badgeBg}`}>
              <div className="flex items-center gap-3.5 z-10">
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/50 shadow-inner">
                  {theme.icon}
                </div>
                <div>
                  <span className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
                    एकूण कल (Overall Sentiment)
                  </span>
                  <h2 className="text-2xl font-bold font-mukta tracking-wide">
                    {result.label_mr} <span className="text-sm font-inter font-medium opacity-80">({result.sentiment})</span>
                  </h2>
                </div>
              </div>

              <div className="z-10 flex flex-col sm:items-end">
                <span className="text-xs text-slate-400">Confidence Score</span>
                <span className="text-2xl font-black font-inter tracking-tight">
                  {result.confidence}%
                </span>
              </div>
            </div>

            {/* Feature 1: Word-Level Explainability (XAI) */}
            <div className="backdrop-blur-xl bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-slate-200">
                    शब्द-स्तरीय विश्लेषण (Word Explainability / XAI)
                  </h3>
                </div>
                <div className="flex items-center gap-3 text-[11px]">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" /> Positive
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-rose-400" /> Negative
                  </span>
                </div>
              </div>

              <p className="text-xs text-slate-500">
                मॉडेलने घेतलेल्या निर्णयासाठी कारणीभूत ठरलेले शब्द (Words contributing to model inference):
              </p>

              <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/60 leading-loose text-lg font-mukta">
                {result.tokens.map((t, idx) => {
                  if (t.tag === 'pos') {
                    return (
                      <span key={idx} className="mx-0.5 px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-500/30">
                        {t.word}{' '}
                      </span>
                    );
                  }
                  if (t.tag === 'neg') {
                    return (
                      <span key={idx} className="mx-0.5 px-2 py-0.5 rounded-md bg-rose-500/20 text-rose-300 font-semibold border border-rose-500/30">
                        {t.word}{' '}
                      </span>
                    );
                  }
                  return <span key={idx} className="text-slate-300">{t.word} </span>;
                })}
              </div>
            </div>

            {/* Feature 2: Aspect-Based Sentiment Mining */}
            <div className="backdrop-blur-xl bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 space-y-3">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                <h3 className="text-sm font-semibold text-slate-200">
                  घटक विश्लेषण (Aspect-Based Sentiment Mining)
                </h3>
              </div>
              <p className="text-xs text-slate-500">
                चित्रपटाच्या मुख्य घटकांनुसार स्वतंत्र अभिप्राय (Aspect breakdown across key cinematic pillars):
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                {Object.entries(result.aspects).map(([aspect, score], idx) => {
                  let pillStyle = "bg-slate-800/50 text-slate-400 border-slate-700/50";
                  if (score === "Positive") pillStyle = "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
                  if (score === "Negative") pillStyle = "bg-rose-500/10 text-rose-400 border-rose-500/30";
                  if (score === "Neutral")  pillStyle = "bg-amber-500/10 text-amber-400 border-amber-500/30";

                  return (
                    <div 
                      key={idx} 
                      className="flex items-center justify-between p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/50"
                    >
                      <span className="font-mukta font-medium text-slate-200 text-base">{aspect}</span>
                      <span className={`text-xs px-2.5 py-1 rounded-lg border font-semibold ${pillStyle}`}>
                        {score}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}