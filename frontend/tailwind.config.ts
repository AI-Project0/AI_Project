import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: {
          50: "#faf9f6", // Warm Cream Background
          100: "#f7f3e8",
          200: "#efe5d0",
          300: "#e3d0b0",
          400: "#d3b488",
        },
        "stone-dark": "#5c4d43", // Warm Brown Text
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
      },
      boxShadow: {
        "soft-xl": "0 20px 25px -5px rgba(211, 180, 136, 0.1), 0 10px 10px -5px rgba(211, 180, 136, 0.04)",
      },
    },
  },
  plugins: [],
};
export default config;
