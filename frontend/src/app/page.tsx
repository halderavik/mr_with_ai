"use client"

import "./globals.css";
import { GitHubIcon } from "@/components/github-icon"
import { GoogleIcon } from "@/components/google-icon"
import Link from "next/link"

export default function LoginPage() {
  const handleUploadComplete = (dataset_id: string) => {
    // This function is no longer needed, but kept for reference if needed in the future
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-md rounded-lg border border-gray-200 bg-white p-8 shadow-sm">
        <div className="mb-6 text-center">
          <h1 className="mb-2 text-2xl font-bold text-gray-900">Market Pro</h1>
          <p className="text-sm text-gray-500">Sign in to access your market research dashboard</p>
        </div>

        <div className="space-y-4">
          <button className="flex w-full items-center justify-center gap-2 rounded border border-gray-300 px-4 py-2.5 text-sm font-medium text-gray-700 transition hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-200">
            <span className="flex items-center justify-center"><GitHubIcon className="h-5 w-5" /></span>
            Continue with GitHub
          </button>

          <button className="flex w-full items-center justify-center gap-2 rounded border border-gray-300 px-4 py-2.5 text-sm font-medium text-gray-700 transition hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-200">
            <span className="flex items-center justify-center"><GoogleIcon className="h-5 w-5" /></span>
            Continue with Google
          </button>

          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200"></div>
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-white px-2 text-gray-400">OR CONTINUE WITH</span>
            </div>
          </div>

          <div className="space-y-1">
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              id="email"
              type="email"
              placeholder="name@example.com"
              className="w-full rounded border border-gray-300 px-3 py-2.5 text-sm focus:border-gray-400 focus:outline-none"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <Link href="/forgot-password" className="text-sm text-gray-600 hover:text-gray-900">
                Forgot password?
              </Link>
            </div>
            <input
              id="password"
              type="password"
              className="w-full rounded border border-gray-300 px-3 py-2.5 text-sm focus:border-gray-400 focus:outline-none"
            />
          </div>

          <button
            className="w-full rounded bg-gray-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-200"
            onClick={() => (window.location.href = "/dashboard")}
          >
            Sign In
          </button>

          <p className="mt-6 text-center text-sm text-gray-500">
            Don't have an account?{" "}
            <Link href="/signup" className="font-medium text-gray-900 hover:underline">
              Sign up
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
