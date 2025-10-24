"use client"

import React, { useRef } from "react"

import { useState } from "react"
import { Home, HelpCircle, Bell, User, Plus, BarChart3, PieChart, LineChart, Network } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

export default function DashboardPage() {
  const router = useRouter()
  const [query, setQuery] = useState("i want to run Van-Westendrop analysis")
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [previewRows, setPreviewRows] = useState<any[]>([])
  const [columns, setColumns] = useState<string[]>([])
  const [metadata, setMetadata] = useState<any | null>(null)
  const [filename, setFilename] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const analysisTools = [
    {
      id: 1,
      name: "Gabor Granger",
      description: "Price sensitivity analysis to determine optimal pricing",
      icon: <BarChart3 className="h-10 w-10 text-primary" />,
      path: "/analysis/gabor-granger",
    },
    {
      id: 2,
      name: "Driver Analysis",
      description: "Identify key factors driving customer satisfaction and loyalty",
      icon: <LineChart className="h-10 w-10 text-primary" />,
      path: "/analysis/driver-analysis",
    },
    {
      id: 3,
      name: "Segmentation",
      description: "Cluster analysis to identify distinct customer segments",
      icon: <PieChart className="h-10 w-10 text-primary" />,
      path: "/analysis/segmentation",
    },
    {
      id: 4,
      name: "Choice Based Conjoint",
      description: "Analyze feature preferences and willingness to pay",
      icon: <Network className="h-10 w-10 text-primary" />,
      path: "/analysis/conjoint",
    },
  ]

  const handleFileButtonClick = (e: React.MouseEvent) => {
    e.preventDefault()
    if (fileInputRef.current) {
      fileInputRef.current.value = "" // allow re-uploading same file
      fileInputRef.current.click()
    }
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    setPreviewRows([])
    setColumns([])
    setMetadata(null)
    setFilename("")
    const file = e.target.files && e.target.files[0]
    if (!file) return
    setFilename(file.name)
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append("file", file)
      const resp = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      })
      if (!resp.ok) {
        const errBody = await resp.json()
        throw new Error(errBody.detail || "Upload failed")
      }
      const data = await resp.json()
      setPreviewRows(data.preview_rows || [])
      setColumns(data.preview_rows.length > 0 ? Object.keys(data.preview_rows[0]) : [])
      setMetadata(data.metadata || null)
      localStorage.setItem("marketpro_uploaded_data", JSON.stringify({
        dataset_id: data.dataset_id,
        filename: data.filename,
        metadata: data.metadata
      }))
      console.log("[DEBUG] Stored upload data in localStorage")
      router.push("/analysis")
    } catch (err: any) {
      setError(err.message || "Upload error")
    } finally {
      setUploading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Navigate to analysis page with the query
    router.push(`/analysis?query=${encodeURIComponent(query)}`)
  }

  const renderMetadataTable = (metadata: any) => {
    if (!metadata) return null;
    const { full_meta_dict, ...displayMetadata } = metadata;
    return (
      <div className="space-y-4">
        {Object.entries(displayMetadata).map(([key, value]) => {
          if (value == null) return null;
          let displayValue: any;
          if (typeof value === "object") {
            if (Array.isArray(value)) {
              displayValue = value.join(", ");
            } else {
              displayValue = Object.entries(value)
                .map(([k, v]) => `${k}: ${v}`)
                .join(", ");
            }
          } else {
            displayValue = value;
          }
          return (
            <Card key={key} className="mb-2">
              <CardContent>
                <div className="font-semibold capitalize mb-1">{key.replace(/_/g, " ")}</div>
                <div className="text-sm whitespace-pre-wrap">{displayValue}</div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    );
  };

  const renderDataTable = (data: any[]) => {
    if (!data.length) return null;
    const columns = Object.keys(data[0]);
    return (
      <div className="rounded-md border mb-4">
        <table className="min-w-full bg-white text-sm">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col} className="border-b px-3 py-2 text-left bg-gray-50">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx}>
                {columns.map((col) => (
                  <td key={col} className="border-b px-3 py-2">{row[col] !== null && row[col] !== undefined ? String(row[col]) : ""}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="flex items-center justify-between border-b bg-white p-4 shadow-sm">
        <Link href="/dashboard">
          <Button variant="ghost" size="icon" className="rounded-full bg-gray-900 text-white hover:bg-gray-800">
            <Home className="h-5 w-5" />
            <span className="sr-only">Home</span>
          </Button>
        </Link>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" className="rounded-full">
            <HelpCircle className="h-5 w-5" />
            <span className="sr-only">Help</span>
          </Button>
          <Button variant="ghost" size="icon" className="rounded-full">
            <Bell className="h-5 w-5" />
            <span className="sr-only">Notifications</span>
          </Button>
          <Button variant="ghost" size="icon" className="rounded-full">
            <User className="h-5 w-5" />
            <span className="sr-only">Profile</span>
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto max-w-5xl px-4 py-8">
        <h1 className="mb-8 text-3xl font-bold text-gray-900">Welcome back...</h1>

        {/* Chat Input */}
        <form onSubmit={handleSubmit} className="mb-12">
          <div className="relative">
            <Input
              type="text"
              placeholder="What analysis would you like to run today?"
              className="h-14 rounded-full border-gray-300 px-6 text-lg shadow-sm focus:border-primary focus:ring-primary"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <Button type="button" size="icon" className="absolute right-2 top-2 rounded-full" onClick={handleFileButtonClick} disabled={uploading}>
              <Plus className="h-5 w-5" />
              <span className="sr-only">Upload Data File</span>
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".sav,.csv,.xlsx,.xls"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
          </div>
        </form>

        {/* File Upload Preview/Metadata */}
        {(uploading || error || previewRows.length > 0 || metadata) && (
          <div className="mb-12">
            {uploading && <div className="text-blue-600 mb-2">Uploading…</div>}
            {error && <div className="text-red-600 mb-2">Error: {error}</div>}
            {previewRows.length > 0 && (
              <div className="mb-4">
                <div className="p-2 font-semibold">Preview of first 10 rows ({filename})</div>
                {renderDataTable(previewRows)}
              </div>
            )}
            {metadata && (
              <div className="mb-4">
                <div className="font-semibold mb-1">SPSS Metadata</div>
                {renderMetadataTable(metadata)}
                <div className="text-xs text-gray-500">(All SPSS metadata fields have been captured above.)</div>
              </div>
            )}
          </div>
        )}

        {/* Analysis Tools */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {analysisTools.map((tool) => (
            <Link key={tool.id} href={tool.path}>
              <Card className="h-full cursor-pointer transition-all hover:shadow-md">
                <CardContent className="flex h-full flex-col items-center justify-center p-6 text-center">
                  <div className="mb-4 rounded-full bg-primary/10 p-3">{tool.icon}</div>
                  <h3 className="mb-2 text-lg font-medium">{tool.name}</h3>
                  <p className="text-sm text-gray-500">{tool.description}</p>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </main>
    </div>
  )
}
