"use client"

import { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Chart {
  type: string;
  title: string;
  data: any;
  annotations?: any;
  plot_data?: string;  // Base64 encoded PNG data
}

interface ChartCarouselProps {
  charts: Chart[];
}

export function ChartCarousel({ charts }: ChartCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (charts.length <= 1) return;
      
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          setCurrentIndex((prevIndex) => 
            prevIndex === 0 ? charts.length - 1 : prevIndex - 1
          );
          break;
        case 'ArrowRight':
          event.preventDefault();
          setCurrentIndex((prevIndex) => 
            prevIndex === charts.length - 1 ? 0 : prevIndex + 1
          );
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [charts.length]);

  if (!charts || charts.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500">
        No charts available
      </div>
    );
  }

  const goToPrevious = () => {
    setCurrentIndex((prevIndex) => 
      prevIndex === 0 ? charts.length - 1 : prevIndex - 1
    );
  };

  const goToNext = () => {
    setCurrentIndex((prevIndex) => 
      prevIndex === charts.length - 1 ? 0 : prevIndex + 1
    );
  };

  const goToSlide = (index: number) => {
    setCurrentIndex(index);
  };

  const currentChart = charts[currentIndex];

  return (
    <div className="relative h-full w-full flex flex-col items-center justify-center">
      {/* Chart Display */}
      <div className="h-full w-full flex items-center justify-center">
        <div className="w-full h-full flex flex-col items-center justify-center">
          <h3 className="text-lg font-medium mb-4 mt-4 text-center">
            {currentChart.title}
          </h3>
          
          {currentChart.type === "van_westendorp_curves" && (
            <div className="w-full h-[400px] flex items-center justify-center bg-white rounded-lg border">
              {currentChart.plot_data ? (
                <img
                  src={`data:image/png;base64,${currentChart.plot_data}`}
                  alt={currentChart.title}
                  className="max-w-full max-h-[380px] object-contain"
                />
              ) : (
                <div className="flex h-full items-center justify-center text-gray-500">
                  No plot data available
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Navigation Controls */}
      {charts.length > 1 && (
        <>
          {/* Previous Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={goToPrevious}
            className="absolute left-2 top-1/2 transform -translate-y-1/2 z-10 bg-white/80 hover:bg-white"
            aria-label="Previous chart"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          {/* Next Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={goToNext}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 z-10 bg-white/80 hover:bg-white"
            aria-label="Next chart"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          {/* Dots Indicator */}
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
            {charts.map((_, index) => (
              <button
                key={index}
                onClick={() => goToSlide(index)}
                className={`w-2 h-2 rounded-full transition-colors ${
                  index === currentIndex 
                    ? 'bg-blue-600' 
                    : 'bg-gray-300 hover:bg-gray-400'
                }`}
                aria-label={`Go to chart ${index + 1}`}
              />
            ))}
          </div>

          {/* Chart Counter */}
          <div className="absolute top-4 right-4 bg-white/80 px-2 py-1 rounded text-sm text-gray-600">
            {currentIndex + 1} of {charts.length}
          </div>

          {/* Keyboard hint */}
          <div className="absolute bottom-4 right-4 bg-white/80 px-2 py-1 rounded text-xs text-gray-500">
            Use ← → keys
          </div>
        </>
      )}
    </div>
  );
} 