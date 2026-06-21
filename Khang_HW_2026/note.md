## use gemini api

import { GoogleGenerativeAI } from "@google/generative-ai"
const API_KEY = process.env.VITE_GEMINI_API_KEY

if (!API_KEY) {
    console.error("❌  Missing VITE_GEMINI_API_KEY in your .env file.")
    process.exit(1)
}

const genAI = new GoogleGenerativeAI(API_KEY)
const model = genAI.getGenerativeModel({
    model: "gemini-3.1-flash-lite",
    generationConfig: { responseMimeType: "application/json" }
})
