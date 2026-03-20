import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "CUA Local Test App",
  description: "A simple Next.js app for testing CUA local browser automation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: 'system-ui, sans-serif' }}>
        {children}
      </body>
    </html>
  );
}
