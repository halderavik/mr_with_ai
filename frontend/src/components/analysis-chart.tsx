"use client"

import { useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  ChartType
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface AnalysisChartProps {
  priceGrid: number[];
  tooCheap: number[];
  tooExpensive: number[];
  bargain: number[];
  gettingExpensive: number[];
  pmc: number;
  pme: number;
  opp: number;
}

export function AnalysisChart({
  priceGrid,
  tooCheap,
  tooExpensive,
  bargain,
  gettingExpensive,
  pmc,
  pme,
  opp
}: AnalysisChartProps) {
  console.log("[DEBUG] Chart Props:", {
    priceGrid,
    tooCheap,
    tooExpensive,
    bargain,
    gettingExpensive,
    pmc,
    pme,
    opp
  });

  const chartRef = useRef<ChartJS<"line">>(null);

  const data = {
    labels: priceGrid.map(price => `$${price.toFixed(2)}`),
    datasets: [
      {
        label: 'Too Cheap',
        data: tooCheap,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'Too Expensive',
        data: tooExpensive,
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
      },
      {
        label: 'Bargain',
        data: bargain,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: 'Getting Expensive',
        data: gettingExpensive,
        borderColor: 'rgb(153, 102, 255)',
        backgroundColor: 'rgba(153, 102, 255, 0.5)',
      },
    ],
  };

  console.log("[DEBUG] Chart Data:", data);

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Van Westendorp Price Sensitivity Analysis',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Price ($)'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Percentage of Respondents (%)'
        },
        min: 0,
        max: 100
      }
    }
  };

  useEffect(() => {
    console.log("[DEBUG] Chart Effect - Drawing vertical lines");
    if (chartRef.current) {
      const chart = chartRef.current;
      
      // Add vertical lines for PMC, PME, and OPP
      const addVerticalLine = (value: number, label: string, color: string) => {
        const x = chart.scales.x.getPixelForValue(value);
        const ctx = chart.ctx;
        
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, chart.chartArea.top);
        ctx.lineTo(x, chart.chartArea.bottom);
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.stroke();
        
        // Add label
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.fillText(label, x, chart.chartArea.top - 5);
        ctx.restore();
      };
      
      chart.draw();
      addVerticalLine(pmc, 'PMC', 'red');
      addVerticalLine(pme, 'PME', 'blue');
      addVerticalLine(opp, 'OPP', 'green');
    }
  }, [pmc, pme, opp]);

  return (
    <div className="w-full h-full">
      <Line ref={chartRef} data={data} options={options} />
    </div>
  );
}
