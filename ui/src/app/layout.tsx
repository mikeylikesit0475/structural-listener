export const metadata = {
  title: "Structural Listener",
  description: "Simulated acoustic-emission monitoring demo",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0 }}>{children}</body>
    </html>
  );
}