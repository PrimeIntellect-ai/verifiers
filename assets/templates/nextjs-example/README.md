# Next.js Example App

A simple Next.js application designed for testing the CUA Local browser automation system.

## Features

- **Counter**: Increment, decrement, and reset buttons
- **Text Input**: Input field with submit functionality
- **Selection**: Multiple option buttons
- **Navigation**: Links between pages
- **State Display**: Real-time state summary

## Local Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Used in CUA Local Testing

This app is deployed alongside the CUA Local server in a sandbox container. The CUA Local environment will:

1. Upload this app to the sandbox
2. Install dependencies and build it
3. Start the production server on port 3000
4. Start the CUA server on port 3001
5. Create a browser session pointing to `http://localhost:3000`

## Interactive Elements

| Element | ID | Description |
|---------|------|-------------|
| Main Title | `main-title` | Page heading |
| Increment Button | `increment-btn` | Increases counter |
| Decrement Button | `decrement-btn` | Decreases counter |
| Reset Button | `reset-btn` | Resets counter to 0 |
| Text Input | `text-input` | Text input field |
| Submit Button | `submit-btn` | Submits input value |
| Option A/B/C | `select-option-a/b/c` | Selection buttons |
| About Link | `about-link` | Navigation to about page |
| Home Link | `home-link` | Navigation back to home |
