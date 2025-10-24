"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload, X, FileText, FileSpreadsheet, FileCode, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"

export interface FileUploaderProps {
  onUpload: (files: FileList | null) => void
  isUploading: boolean
  onCancel: () => void
  accept?: string
}

export function FileUploader({ onUpload, isUploading, onCancel, accept }: FileUploaderProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [progress, setProgress] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  // Handle drag events
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  // Handle drop event
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFiles(e.dataTransfer.files)
    }
  }

  // Handle file input change
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFiles(e.target.files)
    }
  }

  // Handle upload button click
  const handleUpload = () => {
    if (selectedFiles) {
      onUpload(selectedFiles)

      // Simulate progress
      const interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 10
          if (newProgress >= 100) {
            clearInterval(interval)
          }
          return newProgress
        })
      }, 200)
    }
  }

  // Get file icon based on extension
  const getFileIcon = (fileName: string) => {
    const extension = fileName.split(".").pop()?.toLowerCase()

    switch (extension) {
      case "csv":
      case "xlsx":
      case "xls":
        return <FileSpreadsheet className="h-5 w-5 text-blue-500" />
      case "sav": // SPSS
        return <FileText className="h-5 w-5 text-purple-500" />
      case "py":
      case "r":
        return <FileCode className="h-5 w-5 text-green-500" />
      default:
        return <FileText className="h-5 w-5 text-gray-500" />
    }
  }

  return (
    <Card className={`w-full ${dragActive ? "border-primary" : ""}`}>
      <CardContent className="p-6">
        <div
          className="flex flex-col items-center justify-center w-full"
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            className="hidden"
            onChange={handleChange}
            accept={accept}
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="flex flex-col items-center justify-center w-full cursor-pointer"
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              {isUploading ? (
                <Loader2 className="w-8 h-8 mb-4 text-primary animate-spin" />
              ) : (
                <Upload className="w-8 h-8 mb-4 text-primary" />
              )}
              <p className="mb-2 text-sm text-gray-500">
                <span className="font-semibold">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-gray-500">
                {accept ? `Supported formats: ${accept}` : "Any file type"}
              </p>
            </div>
          </label>
        </div>
      </CardContent>
    </Card>
  )
}
