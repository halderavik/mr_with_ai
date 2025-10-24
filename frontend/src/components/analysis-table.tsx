"use client"

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface AnalysisTableProps {
  priceGrid: number[];
  tooCheap: number[];
  tooExpensive: number[];
  bargain: number[];
  gettingExpensive: number[];
  pmc: number;
  pme: number;
  opp: number;
}

export function AnalysisTable({
  priceGrid,
  tooCheap,
  tooExpensive,
  bargain,
  gettingExpensive,
  pmc,
  pme,
  opp
}: AnalysisTableProps) {
  // Create table data with key price points highlighted
  const tableData = priceGrid.map((price, index) => ({
    price,
    tooCheap: tooCheap[index],
    tooExpensive: tooExpensive[index],
    bargain: bargain[index],
    gettingExpensive: gettingExpensive[index],
    isPMC: Math.abs(price - pmc) < 0.01,
    isPME: Math.abs(price - pme) < 0.01,
    isOPP: Math.abs(price - opp) < 0.01,
  }));

  return (
    <div className="w-full h-full overflow-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Price ($)</TableHead>
            <TableHead>Too Cheap (%)</TableHead>
            <TableHead>Bargain (%)</TableHead>
            <TableHead>Getting Expensive (%)</TableHead>
            <TableHead>Too Expensive (%)</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {tableData.map((row, index) => (
            <TableRow
              key={index}
              className={
                row.isPMC
                  ? "bg-red-50"
                  : row.isPME
                  ? "bg-blue-50"
                  : row.isOPP
                  ? "bg-green-50"
                  : ""
              }
            >
              <TableCell className="font-medium">
                ${row.price.toFixed(2)}
                {row.isPMC && " (PMC)"}
                {row.isPME && " (PME)"}
                {row.isOPP && " (OPP)"}
              </TableCell>
              <TableCell>{row.tooCheap.toFixed(1)}%</TableCell>
              <TableCell>{row.bargain.toFixed(1)}%</TableCell>
              <TableCell>{row.gettingExpensive.toFixed(1)}%</TableCell>
              <TableCell>{row.tooExpensive.toFixed(1)}%</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
