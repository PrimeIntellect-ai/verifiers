"use client";

export default function About() {
  return (
    <main style={{ padding: "40px", maxWidth: "800px", margin: "0 auto" }}>
      <h1 id="about-title" style={{ color: "#333", marginBottom: "30px" }}>
        About This App
      </h1>

      <p id="about-description" style={{ color: "#666", marginBottom: "20px" }}>
        This is a simple test application for the CUA Local browser automation system.
        It demonstrates various interactive elements that can be controlled via the
        CUA primitives API.
      </p>

      <h2>Features Tested</h2>
      <ul id="features-list" style={{ color: "#666", lineHeight: "1.8" }}>
        <li>Button clicks (increment, decrement, reset)</li>
        <li>Text input and form submission</li>
        <li>Option selection</li>
        <li>Page navigation</li>
        <li>State display and verification</li>
      </ul>

      <h2>Technical Details</h2>
      <p style={{ color: "#666" }}>
        Built with Next.js 14 using the App Router. The app runs on port 3000
        inside the sandbox container, with the CUA server on port 3001.
      </p>

      <a
        id="home-link"
        href="/"
        style={{
          display: "inline-block",
          marginTop: "20px",
          padding: "10px 20px",
          background: "#0070f3",
          color: "white",
          textDecoration: "none",
          borderRadius: "4px",
        }}
      >
        Back to Home
      </a>
    </main>
  );
}
