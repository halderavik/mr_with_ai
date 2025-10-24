"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useSearchParams } from "next/navigation"
import { Home, Plus, Folder, Download, FileUp, Send, BarChart2, Table, FileText, GripVertical } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { AnalysisChart } from "@/components/analysis-chart"
import { AnalysisTable } from "@/components/analysis-table"
import { AnalysisInsights } from "@/components/analysis-insights"
import { FileUploader } from "@/components/file-uploader"
import { ChartCarousel } from "@/components/chart-carousel"
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable"
import {
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

interface MetadataTable {
  [key: string]: any;
}

interface PreviewData {
  dataset_id: string;
  filename: string;
  preview_rows: any[];
  metadata: MetadataTable | null;
}

interface AnalysisResult {
  visualizations: {
    charts: Array<{
      type: string;
      title: string;
      data: any;
      annotations?: any;
      plot_data?: string;  // Base64 encoded PNG data
    }>;
    tables: Array<{
      type: string;
      title: string;
      data: Array<{
        metric: string;
        value: string;
      }>;
    }>;
  };
  insights: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  context?: {
    analysis_type?: string;
    variables_used?: string[];
    current_visualization?: string;
  };
}

export default function AnalysisPage() {
  const searchParams = useSearchParams()
  const initialQuery = searchParams.get("query") || ""

  const [messages, setMessages] = useState<Message[]>(
    initialQuery ? [{ 
      role: "user", 
      content: initialQuery,
      timestamp: Date.now(),
      context: {}
    }] : [],
  )
  const [input, setInput] = useState("")
  const [isUploading, setIsUploading] = useState(false)
  const [activeTab, setActiveTab] = useState("chart")
  const [showUploader, setShowUploader] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<PreviewData | null>(null)
  const [debugLog, setDebugLog] = useState<string>("")
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [conversationContext, setConversationContext] = useState<{
    currentAnalysisType?: string;
    variablesUsed?: string[];
    lastQuestion?: string;
    lastAnswer?: string;
  }>({})

  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const stored = localStorage.getItem("marketpro_uploaded_data");
    if (stored) {
      try {
        setPreviewData(JSON.parse(stored));
        setDebugLog(prev => prev + "\n[DEBUG] Loaded previewData from localStorage.");
      } catch (e) {
        setDebugLog(prev => prev + "\n[DEBUG] Failed to parse previewData from localStorage.");
      }
    } else {
      setDebugLog(prev => prev + "\n[DEBUG] No previewData found in localStorage.");
    }
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem("marketpro_conversation", JSON.stringify({
        messages,
        context: conversationContext,
        timestamp: Date.now()
      }));
    }
  }, [messages, conversationContext]);

  useEffect(() => {
    const stored = localStorage.getItem("marketpro_conversation");
    if (stored) {
      try {
        const { messages: storedMessages, context: storedContext } = JSON.parse(stored);
        // Only load if we don't have any messages (initial load)
        if (messages.length === 0) {
          setMessages(storedMessages);
          setConversationContext(storedContext);
          setDebugLog(prev => prev + "\n[DEBUG] Loaded conversation from localStorage.");
        }
      } catch (e) {
        setDebugLog(prev => prev + "\n[DEBUG] Failed to parse conversation from localStorage.");
      }
    }
  }, []); // Only run on mount

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    setDebugLog(prev => prev + `\n[DEBUG] Chat handler triggered. input: '${input}', previewData: ${!!previewData}`);
    if (!input.trim()) {
      setDebugLog(prev => prev + "\n[DEBUG] Input is empty. Chat not sent.");
      setError("Please enter a message.");
      return;
    }
    if (!previewData) {
      setDebugLog(prev => prev + "\n[DEBUG] No previewData. Chat not sent.");
      setError("Please upload a file before requesting analysis.");
      return;
    }

    const userMessage = input.trim();
    
    // Check if user wants to clear chat history
    const clearCommands = ["clear", "clear chat", "reset", "reset chat", "start over", "new chat"];
    if (clearCommands.includes(userMessage.toLowerCase())) {
      clearChatHistory();
      setInput("");
      return;
    }

    const newMessage: Message = {
      role: "user",
      content: userMessage,
      timestamp: Date.now(),
      context: {
        analysis_type: conversationContext.currentAnalysisType,
        variables_used: conversationContext.variablesUsed,
        current_visualization: activeTab
      }
    };
    
    setMessages(prev => [...prev, newMessage]);
    setInput("");
    setError(null);

    try {
      setDebugLog(prev => prev + "\n[DEBUG] Sending POST to /api/chat...");
      const resp = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: previewData.dataset_id,
          message: userMessage,
          filter_column: null,
          filter_value: null,
          previous_result: analysisResult,
          conversation_context: {
            messages: messages.slice(-5),
            current_analysis_type: conversationContext.currentAnalysisType,
            variables_used: conversationContext.variablesUsed,
            last_question: conversationContext.lastQuestion,
            last_answer: conversationContext.lastAnswer
          }
        }),
      });
      setDebugLog(prev => prev + `\n[DEBUG] Response status: ${resp.status}`);
      if (!resp.ok) throw new Error("Analysis failed");
      const data = await resp.json();
      
      // Debug log the response
      setDebugLog(prev => prev + `\n[DEBUG] API Response: ${JSON.stringify(data, null, 2)}`);
      
      // Handle chat communication
      const assistantMessage: Message = {
        role: "assistant",
        content: data.reply || "I apologize, but I couldn't generate a proper response.",
        timestamp: Date.now(),
        context: {
          analysis_type: data.context?.analysis_type,
          variables_used: data.context?.variables_used,
          current_visualization: activeTab
        }
      };
      setMessages(prev => [...prev, assistantMessage]);
      
      // Handle visualizations
      if (data.visualizations) {
        setDebugLog(prev => prev + "\n[DEBUG] Received visualizations from MCP");
        console.log("[DEBUG] Setting analysis result with visualizations:", data.visualizations);
        setAnalysisResult({
          visualizations: data.visualizations,
          insights: data.insights || ''
        });
      } else {
        setDebugLog(prev => prev + "\n[DEBUG] No visualizations received, keeping previous ones");
      }
      
      // Update conversation context
      if (data.context) {
        setConversationContext(prev => ({
          ...prev,
          currentAnalysisType: data.context.analysis_type || prev.currentAnalysisType,
          variablesUsed: data.context.variables_used || prev.variablesUsed,
          lastQuestion: userMessage,
          lastAnswer: data.reply
        }));
      }
      
      setDebugLog(prev => prev + "\n[DEBUG] Chat response received and displayed.");
    } catch (err) {
      const errorMessage: Message = {
        role: "assistant",
        content: "Error: " + (err instanceof Error ? err.message : "Unknown error"),
        timestamp: Date.now(),
        context: {}
      };
      setMessages(prev => [...prev, errorMessage]);
      setDebugLog(prev => prev + `\n[DEBUG] Error: ${err instanceof Error ? err.message : err}`);
    }
  };

  // Add debug logging for visualization rendering
  useEffect(() => {
    if (analysisResult?.visualizations) {
      console.log("[DEBUG] Current visualizations:", analysisResult.visualizations);
    }
  }, [analysisResult]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", files[0]);

    try {
      const response = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("[DEBUG] Upload response:", data);
      if (!data.dataset_id) {
        setError("[DEBUG] Upload response missing dataset_id. See console for details.");
        setPreviewData(null);
        return;
      }
      setPreviewData(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setPreviewData(null);
    } finally {
      setIsUploading(false);
    }
  };

  const clearChatHistory = () => {
    // Clear messages from state
    setMessages([]);
    // Clear conversation context
    setConversationContext({});
    // Clear analysis result
    setAnalysisResult(null);
    // Clear from localStorage
    localStorage.removeItem("marketpro_conversation");
    // Clear debug log
    setDebugLog("");
    setDebugLog(prev => prev + "\n[DEBUG] Chat history cleared");
  };

  return (
    <div className="flex h-screen min-h-0 w-full max-w-full overflow-x-hidden bg-gray-50">
      {/* Sidebar */}
      <div className="flex w-16 flex-col items-center border-r bg-white py-4 max-w-full">
        <Button variant="ghost" size="icon" className="mb-6 rounded-full bg-gray-900 text-white hover:bg-gray-800">
          <Home className="h-5 w-5" />
          <span className="sr-only">Home</span>
        </Button>
        <Button variant="ghost" size="icon" className="mb-2">
          <Plus className="h-5 w-5" />
          <span className="sr-only">New Chat</span>
        </Button>
        <Button variant="ghost" size="icon">
          <Folder className="h-5 w-5" />
          <span className="sr-only">Projects</span>
        </Button>
        <div className="mt-auto">
          <Button variant="ghost" size="icon" className="rounded-full">
            <FileUp className="h-5 w-5" onClick={() => setShowUploader(true)} />
            <span className="sr-only">Upload File</span>
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 flex-col h-full min-h-0 w-full max-w-full">
        <ResizablePanelGroup direction="horizontal" className="h-full min-h-0 flex-1">
          {/* Chat Panel */}
          <ResizablePanel defaultSize={40} minSize={30} maxSize={50} className="p-4 h-full min-h-0 w-full max-w-full">
            <Card className="flex h-full flex-col min-h-0 w-full max-w-full">
              <CardHeader className="border-b px-4 py-3">
                <CardTitle className="text-lg font-medium">Chat Interface</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 min-h-0 p-4 overflow-hidden break-words">
                <div className="flex flex-col space-y-4 h-full min-h-0 overflow-y-auto">
                  {messages.map((message, index) => (
                    <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                      <div
                        className={`max-w-[80%] rounded-lg px-4 py-2 ${
                          message.role === "user" ? "bg-primary text-primary-foreground" : "bg-gray-100 text-gray-800"
                        }`}
                      >
                        {message.content}
                      </div>
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              </CardContent>
              <div className="border-t p-4">
                {error && (
                  <div className="mb-2 p-2 bg-red-100 text-red-700 rounded">{error}</div>
                )}
                {!previewData && (
                  <div className="mb-2 p-2 bg-yellow-100 text-yellow-800 rounded">[DEBUG] No file uploaded. Chat will not be sent to backend.</div>
                )}
                {debugLog && (
                  <pre className="mb-2 p-2 bg-gray-100 text-xs text-gray-700 rounded max-h-40 overflow-y-auto">{debugLog}</pre>
                )}
                <form onSubmit={handleSendMessage} className="flex gap-2">
                  <Textarea
                    placeholder="Ask about your data or request an analysis..."
                    className="min-h-[60px] flex-1 resize-none"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                  />
                  <Button type="submit" size="icon" className="h-[60px] w-[60px]">
                    <Send className="h-5 w-5" />
                    <span className="sr-only">Send</span>
                  </Button>
                </form>
              </div>
            </Card>
          </ResizablePanel>

          <ResizableHandle withHandle>
            <div className="flex h-4 w-4 items-center justify-center">
              <GripVertical className="h-4 w-4" />
            </div>
          </ResizableHandle>

          {/* Analysis Panel */}
          <ResizablePanel defaultSize={60} className="p-4 h-full min-h-0 w-full max-w-full">
            <div className="flex h-full flex-col gap-4 min-h-0 w-full max-w-full">
              {/* Visualization Panel */}
              <Card className="flex-1 flex flex-col min-h-0 w-full max-w-full">
                <CardHeader className="border-b px-4 py-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg font-medium">Graphs and Tables</CardTitle>
                    <Tabs value={activeTab} onValueChange={setActiveTab} className="w-auto">
                      <TabsList className="grid w-[180px] grid-cols-2">
                        <TabsTrigger value="chart">
                          <BarChart2 className="mr-2 h-4 w-4" />
                          Chart
                        </TabsTrigger>
                        <TabsTrigger value="table">
                          <Table className="mr-2 h-4 w-4" />
                          Table
                        </TabsTrigger>
                      </TabsList>
                    </Tabs>
                  </div>
                </CardHeader>
                <CardContent className="p-6 flex-1 min-h-0 w-full max-w-full flex flex-col justify-center items-center">
                  <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full min-h-0 w-full max-w-full">
                    <TabsContent value="chart" className="mt-0 h-full min-h-0 w-full max-w-full flex flex-col items-center justify-center">
                      <div className="relative h-full min-h-[400px] max-h-[600px] w-full max-w-3xl flex flex-col items-center justify-center">
                        {analysisResult?.visualizations?.charts ? (
                          <ChartCarousel charts={analysisResult.visualizations.charts} />
                        ) : (
                          <div className="flex h-full items-center justify-center text-gray-500">
                            No analysis results available
                          </div>
                        )}
                      </div>
                      <div className="flex justify-end w-full mt-4 gap-2">
                        <Button variant="outline" size="sm">
                          <Download className="mr-1 h-4 w-4" />
                          PNG
                        </Button>
                        <Button variant="outline" size="sm">
                          <Download className="mr-1 h-4 w-4" />
                          PPTX
                        </Button>
                      </div>
                    </TabsContent>
                    <TabsContent value="table" className="mt-0 h-full min-h-0 w-full max-w-full">
                      <div className="relative h-full min-h-[300px] max-h-[500px] overflow-y-auto w-full max-w-full flex flex-col justify-start">
                        {analysisResult?.visualizations?.tables ? (
                          <div>
                            {analysisResult.visualizations.tables.map((table, index) => (
                              <div key={index} className="mb-8">
                                <h3 className="text-lg font-medium mb-4">{table.title}</h3>
                                <div className="border rounded-lg overflow-x-auto">
                                  <div className="min-w-full divide-y divide-gray-200">
                                    <div className="bg-gray-50">
                                      <div className="grid grid-cols-2 px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        <div>Metric</div>
                                        <div>Value</div>
                                      </div>
                                    </div>
                                    <div className="bg-white divide-y divide-gray-200">
                                      {table.data.map((row, rowIndex) => (
                                        <div key={rowIndex} className="grid grid-cols-2 px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                          <div>{row.metric}</div>
                                          <div>{row.value}</div>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="flex h-full items-center justify-center text-gray-500">
                            No analysis results available
                          </div>
                        )}
                        <div className="absolute right-2 top-2 flex gap-1">
                          <Button variant="outline" size="sm">
                            <Download className="mr-1 h-4 w-4" />
                            CSV
                          </Button>
                          <Button variant="outline" size="sm">
                            <Download className="mr-1 h-4 w-4" />
                            PPTX
                          </Button>
                        </div>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>

              {/* Insights Panel */}
              <Card className="flex-1 flex flex-col min-h-0 w-full max-w-full">
                <CardHeader className="border-b px-4 py-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg font-medium">Insights and Business Decision</CardTitle>
                    <Button variant="outline" size="sm">
                      <FileText className="mr-1 h-4 w-4" />
                      Export
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="p-4 flex-1 min-h-0 w-full max-w-full">
                  <div className="prose max-w-none h-full min-h-0 overflow-y-auto break-words w-full max-w-full">
                    {analysisResult?.insights ? (
                      <div className="text-gray-700">
                        {analysisResult.insights.split("\n").map((paragraph, index) => (
                          <p key={index} className="mb-4">
                            {paragraph}
                          </p>
                        ))}
                      </div>
                    ) : (
                      <p className="text-gray-500 italic">
                        No insights available yet. Upload a file and ask for analysis to see insights.
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* File Upload Modal */}
      {showUploader && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>Upload Data File</CardTitle>
            </CardHeader>
            <CardContent>
              <FileUploader
                onUpload={handleFileUpload}
                isUploading={isUploading}
                accept=".sav,.csv,.xlsx,.xls"
                onCancel={() => setShowUploader(false)}
              />
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
