"use client";

import { useState } from "react";

export default function Home() {
  const [count, setCount] = useState(0);
  const [inputValue, setInputValue] = useState("");
  const [submittedValue, setSubmittedValue] = useState("");
  const [selectedOption, setSelectedOption] = useState("");

  return (
    <main style={{ padding: "40px", maxWidth: "800px", margin: "0 auto" }}>
      <h1 id="main-title" style={{ color: "#333", marginBottom: "30px" }}>
        CUA Local Test Application
      </h1>

      <p id="description" style={{ color: "#666", marginBottom: "40px" }}>
        This is a simple Next.js application designed for testing browser automation
        with the CUA local server. It includes various interactive elements.
      </p>

      {/* Counter Section */}
      <section style={{ marginBottom: "40px", padding: "20px", background: "#f5f5f5", borderRadius: "8px" }}>
        <h2>Counter Test</h2>
        <p id="counter-value" style={{ fontSize: "24px", fontWeight: "bold" }}>
          Count: {count}
        </p>
        <div style={{ display: "flex", gap: "10px" }}>
          <button
            id="increment-btn"
            onClick={() => setCount(c => c + 1)}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              cursor: "pointer",
              background: "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
            }}
          >
            Increment
          </button>
          <button
            id="decrement-btn"
            onClick={() => setCount(c => c - 1)}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              cursor: "pointer",
              background: "#ff4444",
              color: "white",
              border: "none",
              borderRadius: "4px",
            }}
          >
            Decrement
          </button>
          <button
            id="reset-btn"
            onClick={() => setCount(0)}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              cursor: "pointer",
              background: "#666",
              color: "white",
              border: "none",
              borderRadius: "4px",
            }}
          >
            Reset
          </button>
        </div>
      </section>

      {/* Text Input Section */}
      <section style={{ marginBottom: "40px", padding: "20px", background: "#f5f5f5", borderRadius: "8px" }}>
        <h2>Text Input Test</h2>
        <div style={{ marginBottom: "10px" }}>
          <input
            id="text-input"
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type something here..."
            style={{
              padding: "10px",
              fontSize: "16px",
              width: "300px",
              border: "1px solid #ccc",
              borderRadius: "4px",
            }}
          />
          <button
            id="submit-btn"
            onClick={() => setSubmittedValue(inputValue)}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              cursor: "pointer",
              background: "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              marginLeft: "10px",
            }}
          >
            Submit
          </button>
        </div>
        {submittedValue && (
          <p id="submitted-value" style={{ color: "#0070f3" }}>
            Submitted: {submittedValue}
          </p>
        )}
      </section>

      {/* Selection Section */}
      <section style={{ marginBottom: "40px", padding: "20px", background: "#f5f5f5", borderRadius: "8px" }}>
        <h2>Selection Test</h2>
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
          {["Option A", "Option B", "Option C"].map((option) => (
            <button
              key={option}
              id={`select-${option.toLowerCase().replace(" ", "-")}`}
              onClick={() => setSelectedOption(option)}
              style={{
                padding: "10px 20px",
                fontSize: "16px",
                cursor: "pointer",
                background: selectedOption === option ? "#0070f3" : "#ddd",
                color: selectedOption === option ? "white" : "#333",
                border: "none",
                borderRadius: "4px",
              }}
            >
              {option}
            </button>
          ))}
        </div>
        {selectedOption && (
          <p id="selected-option" style={{ marginTop: "10px", color: "#0070f3" }}>
            Selected: {selectedOption}
          </p>
        )}
      </section>

      {/* Navigation Link */}
      <section style={{ marginBottom: "40px", padding: "20px", background: "#f5f5f5", borderRadius: "8px" }}>
        <h2>Navigation Test</h2>
        <a
          id="about-link"
          href="/about"
          style={{
            color: "#0070f3",
            textDecoration: "underline",
            fontSize: "16px",
          }}
        >
          Go to About Page
        </a>
      </section>

      {/* Status Section */}
      <section style={{ padding: "20px", background: "#e8f5e9", borderRadius: "8px" }}>
        <h2>Current State</h2>
        <ul id="state-summary" style={{ listStyle: "none", padding: 0 }}>
          <li>Counter: <strong>{count}</strong></li>
          <li>Input Value: <strong>{inputValue || "(empty)"}</strong></li>
          <li>Submitted: <strong>{submittedValue || "(none)"}</strong></li>
          <li>Selected: <strong>{selectedOption || "(none)"}</strong></li>
        </ul>
      </section>
    </main>
  );
}
